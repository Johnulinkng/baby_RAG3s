from perception import PerceptionResult
from memory import MemoryItem
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import os
import re

# Optional: import log from agent if shared, else define locally
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

load_dotenv()
# Initialize OpenAI client - env OPENAI_API_KEY first; fallback to AWS Secrets if configured
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            from babycare_rag.aws_secrets import get_openai_api_key_from_aws
            api_key = get_openai_api_key_from_aws()
        except Exception:
            api_key = None
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    log("decision", "OpenAI client initialized")
except Exception as e:
    log("decision", f"Failed to initialize OpenAI client: {e}")
    client = OpenAI(api_key="dummy")

def generate_plan(
    perception: PerceptionResult,
    memory_items: List[MemoryItem],
    tool_descriptions: Optional[str] = None
) -> str:
    """Generates a plan (tool call or final answer) using LLM based on structured perception and memory."""

    memory_texts = "\n".join(f"- {m.text}" for m in memory_items) or "None"

    tool_context = f"\nYou have access to the following tools:\n{tool_descriptions}" if tool_descriptions else ""

    prompt = f"""
You are a reasoning-driven AI agent with access to tools. Your job is to solve the user's request step-by-step by reasoning through the problem, selecting a tool if needed, and continuing until the FINAL_ANSWER is produced.{tool_context}

Always follow this loop:

1. Think step-by-step about the problem.
2. If a tool is needed, respond using the format:
   FUNCTION_CALL: tool_name|param1=value1|param2=value2
3. When the final answer is known, respond using:
   FINAL_ANSWER: [your final result]

Guidelines:
- Respond using EXACTLY ONE of the formats above per step.
- Do NOT include extra text, explanation, or formatting.
- Use nested keys (e.g., input.string) and square brackets for lists.
- You can reference these relevant memories:
{memory_texts}

Evidence policy:
- If the latest tool output is from search_documents and returns structured JSON with fields results[{"text", "context_tag", "source", "chunk_id"}] and sources[], you MUST extract the answer USING ONLY those evidence snippets. Do not use external knowledge.
- If there is NO search_documents evidence available (e.g., retrieval returned no results), you MAY produce an LLM answer, but you MUST prefix it with: "[LLM-generated: no matching documents found] ". Do NOT attach any sources in that case.
- If evidence exists, add a final line with Sources: comma-separated unique sources.

Input Summary:
- User input: "{perception.user_input}"
- Intent: {perception.intent}
- Entities: {', '.join(perception.entities)}
- Tool hint: {perception.tool_hint or 'None'}

✅ Examples:
- FUNCTION_CALL: add|a=5|b=3
- FUNCTION_CALL: strings_to_chars_to_int|input.string=INDIA
- FUNCTION_CALL: int_list_to_exponential_sum|input.int_list=[73,78,68,73,65]
- FINAL_ANSWER: [42]

✅ Examples:
- User asks: "What’s the relationship between Cricket and Sachin Tendulkar"
  - FUNCTION_CALL: search_documents|query="relationship between Cricket and Sachin Tendulkar"
  - [receives structured evidence]
  - FINAL_ANSWER: [answer synthesized strictly from evidence]\nSources: [source1, source2]

IMPORTANT:
- Do NOT invent tools. Use only the tools listed below.
- If the question may relate to factual knowledge, use the 'search_documents' tool to look for the answer.
- If the question is mathematical or needs calculation, use the appropriate math tool.
- If the previous tool output already contains factual information, DO NOT search again. Instead, extract the key answer and respond with full sentence from it.
- Keep FINAL_ANSWER concise and direct - just the key information requested.
- Only repeat `search_documents` if the last result was completely irrelevant or empty.
- Do NOT repeat function calls with the same parameters.
- Do NOT output unstructured responses.
- Think before each step. Verify intermediate results mentally before proceeding.
- If unsure or no tool fits, skip to FINAL_ANSWER: [I could not find specific information about this topic]
- You have only 3 attempts. Final attempt must be FINAL_ANSWER
- When analyzing search results, look for specific information patterns aligned with baby care contexts.
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = (response.choices[0].message.content or "").strip()
        log("plan", f"LLM raw output: {raw}")

        for line in raw.splitlines():
            if line.strip().startswith("FUNCTION_CALL:") or line.strip().startswith("FINAL_ANSWER:"):
                log("plan", f"Found structured response: {line.strip()}")
                return line.strip()

        # If no structured response found, avoid ungrounded FINAL_ANSWER for knowledge-seeking intents.
        # If we have not yet called search_documents in this turn history, force a search call.
        has_search = any(getattr(m, "tool_name", None) == "search_documents" for m in memory_items)
        knowledge_intents = {"factoid", "numerical_range", "advice"}
        if not has_search and (perception.intent in knowledge_intents or perception.intent is None):
            q = perception.user_input.replace("|", " ")
            return f"FUNCTION_CALL: search_documents|query=\"{q}\""

        # Otherwise, last resort: wrap as FINAL_ANSWER (e.g., after search results were provided)
        return f"FINAL_ANSWER: {raw.strip()}"

    except Exception as e:
        log("plan", f"⚠️ Decision generation failed: {e}")
        return "FINAL_ANSWER: [unknown]"
