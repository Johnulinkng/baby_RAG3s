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

def log(stage: str, msg: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{stage}] {msg}")

max_steps = 3

async def main(user_input: str):
    try:
        print("[agent] Starting agent...")
        print(f"[agent] Current working directory: {os.getcwd()}")
        
        server_params = StdioServerParameters(
            command="python",
            args=["math_mcp_embeddings.py"],
            cwd="/Users/rohinim1/Documents/Personal/Gaonkar-EAGv1/Cursor/Assignment-7/baby-care-agent"
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
                            final_answer = ""

                            while step < max_steps:
                                log("loop", f"Step {step + 1} started")

                                perception = extract_perception(user_input)
                                log("perception", f"Intent: {perception.intent}, Tool hint: {perception.tool_hint}")

                                retrieved = memory.retrieve(query=user_input, top_k=3, session_filter=session_id)
                                log("memory", f"Retrieved {len(retrieved)} relevant memories")

                                plan = generate_plan(perception, retrieved, tool_descriptions=tool_descriptions)
                                log("plan", f"Plan generated: {plan}")

                                if plan.startswith("FINAL_ANSWER:"):
                                    log("agent", f"✅ FINAL RESULT: {plan}")
                                    final_answer = plan.replace("FINAL_ANSWER:", "").strip()
                                    break

                                try:
                                    result = await execute_tool(session, tools, plan)
                                    log("tool", f"{result.tool_name} returned: {result.result}")

                                    memory.add(MemoryItem(
                                        text=f"Tool call: {result.tool_name} with {result.arguments}, got: {result.result}",
                                        type="tool_output",
                                        tool_name=result.tool_name,
                                        user_query=user_input,
                                        tags=[result.tool_name],
                                        session_id=session_id
                                    ))

                                    user_input = f"Original task: {query}\nPrevious output: {result.result}\nWhat should I do next?"

                                except Exception as e:
                                    log("error", f"Tool execution failed: {e}")
                                    final_answer = f"Apologies, I'm unable to respond at this moment. Please try again later."
                                    break

                                step += 1
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

