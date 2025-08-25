import asyncio
import time
import os
import datetime
from perception import extract_perception
from memory import MemoryManager, MemoryItem
from decision import generate_plan
from action import execute_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
 # use this to connect to running server

import shutil
import sys
from pathlib import Path
import re
import json
from openai import OpenAI

# Evidence-based answer prompts
EVIDENCE_BASED_ANSWER_SYSTEM_PROMPT = """
You are a precise, evidence-grounded, professional infant care AI assistant. Your task is to answer the user's question strictly and only based on the provided <EVIDENCE>.

# Instructions:
1. Carefully read and understand all <EVIDENCE> content.
2. Your answer must be entirely derived from the <EVIDENCE>; do not add external knowledge, opinions, or unsupported reasoning.
3. If the <EVIDENCE> is sufficient, provide an accurate, concise, professional answer. 
4. Generating answer naturally like a responsible human would in a conversation: fluent sentences, no asterisks, clear and direct.
5. If the <EVIDENCE> is only partially relevant, answer only the relevant parts and explicitly state: "Based on the available evidence, this cannot be fully determined..." when appropriate.
6. If the <EVIDENCE> cannot answer the question at all, your answer must be: "According to the available evidence, no relevant answer was found."
7. You must end the answer with a separate line listing all cited sources in the format: `Sources: DocumentA.pdf, DocumentB.pdf...`
   If there is no evidence and the question requires professional knowledge, generate a brief, professional answer and append: (Model-generated answer, please verify)
# Output format:
[Your answer]

Sources: [List of cited documents]
"""

EVIDENCE_BASED_ANSWER_USER_PROMPT = """
User Question: {query}

<EVIDENCE>
{evidence}
</EVIDENCE>

Please answer the question strictly based on the evidence above.
"""

# General answer prompt for when no evidence is available
GENERAL_ANSWER_SYSTEM_PROMPT = """
You are a professional infant care AI assistant. Provide accurate, practical baby-care advice based on your expertise.

# Instructions:
1. Provide accurate, professional infant-care advice.
2. Keep the answer concise and easy to understand.
3. For medical concerns, recommend consulting a healthcare professional.
4. Maintain a friendly and professional tone.
5. End the answer with: (Model-generated answer, please verify)

# Output format:
[Your professional answer]

(Model-generated answer, please verify)
"""

GENERAL_ANSWER_USER_PROMPT = """
User Question: {query}

Please answer this infant-care-related question based on your expertise.
"""

# Initialize OpenAI client (shared for evidence-based answering)
try:
    _openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=_openai_api_key) if _openai_api_key else OpenAI()
except Exception:
    openai_client = OpenAI()


def log(stage: str, msg: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{stage}] {msg}")

max_steps = 2

async def main(user_input: str):
    try:
        print("[agent] Starting agent...")
        print(f"[agent] Current working directory: {os.getcwd()}")

        server_params = StdioServerParameters(
            command="python",
            args=["math_mcp_embeddings.py"],
            cwd=str(Path(__file__).parent.resolve())
        )

        try:
            async with stdio_client(server_params) as (read, write):
                print("Connection established, creating session...")
                try:
                    async with ClientSession(read, write) as session:
                        print("[agent] Session created, initializing...")

                        try:
                            await session.initialize()
                            print("[agent] MCP session initialized")

                            # Your reasoning, planning, perception etc. would go here
                            tools = await session.list_tools()
                            print("Available tools:", [t.name for t in tools.tools])

                            # Get available tools
                            print("Requesting tool list...")
                            tools_result = await session.list_tools()
                            tools = tools_result.tools
                            tool_descriptions = "\n".join(
                                f"- {tool.name}: {getattr(tool, 'description', 'No description')}"
                                for tool in tools
                            )

                            log("agent", f"{len(tools)} tools loaded")

                            memory = MemoryManager()
                            session_id = f"session-{int(time.time())}"
                            query = user_input
                            step = 0
                            final_answer = "No response generated."

                            while step < max_steps:
                                log("loop", f"Step {step + 1} started")

                                perception = extract_perception(user_input)
                                log("perception", f"Intent: {perception.intent}, Tool hint: {perception.tool_hint}")

                                retrieved = memory.retrieve(query=user_input, top_k=3, session_filter=session_id)
                                log("memory", f"Retrieved {len(retrieved)} relevant memories")

                                plan = generate_plan(perception, retrieved, tool_descriptions=tool_descriptions)
                                log("plan", f"Plan generated: {plan}")

                                if plan.startswith("FINAL_ANSWER:"):
                                    # Guard: for knowledge intents, only accept FINAL_ANSWER if we've grounded it via search
                                    knowledge_intents = {"factoid", "numerical_range", "advice"}
                                    has_search = any(
                                        getattr(m, "tool_name", None) == "search_documents" and m.session_id == session_id
                                        for m in memory.data
                                    )
                                    if (perception.intent in knowledge_intents or perception.intent is None) and not has_search:
                                        # Force a search step instead of accepting hallucinated FINAL_ANSWER
                                        query_sanitized = query.replace("|", " ")
                                        user_input = f'FUNCTION_CALL: search_documents|query="{query_sanitized}"'
                                        log("agent", "Deferring ungrounded FINAL_ANSWER; forcing search_documents")
                                    else:
                                        log("agent", f"✅ FINAL RESULT: {plan}")
                                        final_answer = plan.replace("FINAL_ANSWER:", "").strip()
                                        break

                                # Do NOT finalize based on temperature pattern in the plan alone.
                                # Only accept FINAL_ANSWER explicitly or finalize after grounded search results.

                                try:
                                    result = await execute_tool(session, tools, plan)
                                    log("tool", f"{result.tool_name} returned result (length: {len(str(result.result))})")

                                    # Store search results and let LLM generate final answer
                                    memory.add(MemoryItem(
                                        text=f"Tool call: {result.tool_name} with {result.arguments}, got: {result.result}",
                                        type="tool_output",
                                        tool_name=result.tool_name,
                                        user_query=query,
                                        tags=[result.tool_name],
                                        session_id=session_id
                                    ))

                                    # For search_documents: short-circuit to evidence-based final answer
                                    if result.tool_name == "search_documents":
                                        try:
                                            payload = result.result  # should be dict with 'results' and 'sources'
                                            if isinstance(payload, str):
                                                payload = json.loads(payload)
                                            results_list = payload.get('results', []) if isinstance(payload, dict) else []
                                            sources_list = payload.get('sources', []) if isinstance(payload, dict) else []

                                            # Build evidence block (concise)
                                            evidence_lines = []
                                            for item in results_list[:10]:
                                                src = item.get('source') or ''
                                                txt = (item.get('text') or '')
                                                snippet = txt[:400] + ('...' if len(txt) > 400 else '')
                                                evidence_lines.append(f"[Source: {src}]\n{snippet}")
                                            evidence = "\n\n".join(evidence_lines) if evidence_lines else ""

                                            system_prompt = EVIDENCE_BASED_ANSWER_SYSTEM_PROMPT
                                            user_prompt = EVIDENCE_BASED_ANSWER_USER_PROMPT.format(query=query, evidence=evidence)

                                            resp = openai_client.chat.completions.create(
                                                model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
                                                messages=[
                                                    {"role": "system", "content": system_prompt},
                                                    {"role": "user", "content": user_prompt},
                                                ],
                                                temperature=0.2,
                                            )
                                            answer_text = (resp.choices[0].message.content or "").strip()

                                            # Ensure sources formatting if evidence existed
                                            if sources_list and "Sources:" not in answer_text:
                                                answer_text = f"{answer_text}\n\nSources: {', '.join(sources_list)}"

                                            final_answer = answer_text
                                            log("agent", "✅ Short-circuited with evidence-based final answer")
                                            break
                                        except Exception as ee:
                                            log("agent", f"Evidence-based answering failed, falling back: {ee}")
                                            # Fallback to previous behavior: ask plan to summarize
                                            if result.result and len(str(result.result)) > 50:
                                                user_input = f"Original question: {query}\nSearch results: {result.result}\nBased on the search results above, provide a concise and direct answer to the original question. If no relevant information is found, say 'I could not find specific information about this topic in the available documents.'"
                                            else:
                                                final_answer = "I could not find specific information about this topic in the available documents."
                                                break
                                    else:
                                        # For other tools, continue with original logic
                                        sources_suffix = ""
                                        try:
                                            if getattr(result, 'sources', None):
                                                unique_sources = "; ".join(result.sources)
                                                sources_suffix = f"\nSources: {unique_sources}"
                                        except Exception:
                                            pass
                                        user_input = f"Original task: {query}\nPrevious output: {result.result}{sources_suffix}\nWhat should I do next?"



                                except Exception as e:
                                    log("error", f"Tool execution failed: {e}")
                                    final_answer = f"Apologies, I'm unable to respond at this moment. Please try again later."
                                    break

                                step += 1

                            # If we've reached max_steps without a final answer, try one more time to generate an answer
                            if step >= max_steps and final_answer == "No response generated.":
                                log("agent", "Max steps reached, attempting final answer generation")
                                # Get the last memory items to see if we have any useful information
                                recent_memories = memory.retrieve(query=query, top_k=5, session_filter=session_id)
                                if recent_memories:
                                    # Try to generate a final answer based on available information
                                    final_perception = extract_perception(query)
                                    final_plan = generate_plan(final_perception, recent_memories, tool_descriptions=tool_descriptions)
                                    if final_plan.startswith("FINAL_ANSWER:"):
                                        final_answer = final_plan.replace("FINAL_ANSWER:", "").strip()
                                        log("agent", f"✅ FINAL ANSWER GENERATED: {final_answer}")
                                    else:
                                        final_answer = "I was unable to find a complete answer to your question based on the available information."
                                        log("agent", f"Using fallback answer: {final_answer}")
                                else:
                                    final_answer = "I was unable to find relevant information to answer your question."
                                    log("agent", f"No memories found, using fallback: {final_answer}")
                        except Exception as e:
                            print(f"[agent] Session initialization error: {str(e)}")
                            final_answer = f"Apologies, I'm unable to respond at this moment. Please try again later."
                except Exception as e:
                    print(f"[agent] Session creation error: {str(e)}")
                    final_answer = f"Apologies, I'm unable to respond at this moment. Please try again later."
        except Exception as e:
            print(f"[agent] Connection error: {str(e)}")
            final_answer = f"Apologies, I'm unable to respond at this moment. Please try again later."
    except Exception as e:
        print(f"[agent] Overall error: {str(e)}")
        final_answer = f"Apologies, I'm unable to respond at this moment. Please try again later."

    log("agent", "========== Agent session complete. ==========")
    return final_answer

if __name__ == "__main__":
    query = input("🧑 What do you want to solve today? → ")
    asyncio.run(main(query))


# What is the weight limit for baby bath tub sling?
# What should I do in case of labour pain?
# My baby has a fever, what should I do?
# What is the ideal temperature for baby to sleep in celsius?
# When do I switch baby from infant car seat to booster seat?

