from perception import PerceptionResult
from memory import MemoryItem
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI  # fallback if get_secret is unavailable
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
# Initialize OpenAI client via external get_secret if available
try:
    import sys as _sys
    _sys.path.append("/home/ubuntu/ios_backend")
    from bk_ask.config import get_secret as _get_secret
    client = _get_secret()
except Exception:
    client = OpenAI()

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

Formatting rules for common cases:
- Temperature ranges: if you identify a range in °F (e.g., 68–72°F), return a single FINAL_ANSWER with both units: "68–72°F (20–22°C)". Avoid calling convert_temperature repeatedly for each bound.
- Sources: if the latest tool output includes snippets like "[Source: FILENAME, ...]", include a final line: "Sources: FILENAME1; FILENAME2" with unique filenames.

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
  - [receives a detailed document]
  - FINAL_ANSWER: Sachin Tendulkar is widely regarded as the "God of Cricket" due to his exceptional skills, longevity, and impact on the sport in India. He is the leading run-scorer in both Test and ODI cricket, and the first to score 100 centuries in international cricket. His influence extends beyond his statistics, as he is seen as a symbol of passion, perseverance, and a national icon.


IMPORTANT:
- 🚫 Do NOT invent tools. Use only the tools listed below.
- 📄 If the question may relate to factual knowledge, use the 'search_documents' tool to look for the answer.
- 🧮 If the question is mathematical or needs calculation, use the appropriate math tool.
- 🤖 If the previous tool output already contains factual information, DO NOT search again. Instead, extract the key answer and respond with: FINAL_ANSWER: [concise answer]
- When you see "Search results:" in the input, this means search has been completed. Extract the most relevant answer from the results and provide a FINAL_ANSWER.
- For temperature questions, look for temperature ranges like "16–29°C (60–85°F)" or similar patterns in the search results.
- Keep FINAL_ANSWER concise and direct - just the key information requested.
- Only repeat `search_documents` if the last result was completely irrelevant or empty.
- ❌ Do NOT repeat function calls with the same parameters.
- ❌ Do NOT output unstructured responses.
- 🧠 Think before each step. Verify intermediate results mentally before proceeding.
- 💥 If unsure or no tool fits, skip to FINAL_ANSWER: [I could not find specific information about this topic]
- ✅ You have only 3 attempts. Final attempt must be FINAL_ANSWER
- 🔍 When analyzing search results, look for specific information patterns:
  * Temperature ranges (e.g., "16–29°C", "60–85°F", "Room temperature")
  * Specific recommendations from medical organizations (AAP, etc.)
  * Safety guidelines and best practices
  * Age-specific information for babies and children
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

        # If no structured response found, but contains temperature info (robust matching), format it
        temp_pattern = r"((?:6\s*8)\s*(?:-|–|~|to)\s*(?:7\s*2)\s*(?:°\s*)?F)(?:\s*(?:\(|\s)\s*((?:2\s*0)\s*(?:-|–|~|to)\s*(?:2\s*2)\s*(?:°\s*)?C)\)?)?"
        if re.search(temp_pattern, raw, flags=re.IGNORECASE):
            log("plan", "Found temperature in unstructured response, formatting (regex match)...")
            return f"FINAL_ANSWER: {raw.strip()}"

        # Fallback: wrap any non-structured raw as FINAL_ANSWER so the agent can converge
        return f"FINAL_ANSWER: {raw.strip()}"

    except Exception as e:
        log("plan", f"⚠️ Decision generation failed: {e}")
        return "FINAL_ANSWER: [unknown]"
