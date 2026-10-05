"""Example 7 — OpenAI Tool Calling and Reasoning Agent Loop.

Demonstrates how downstream code-agent tools (like OpenCode, Cursor, and Cline)
interact with the DeepSeek API server:
1. Client supplies tool declarations (`tools`).
2. Server translates schemas into DeepSeek's emulated tool prompt.
3. Model invokes tools (parsed into OpenAI `tool_calls`).
4. Client executes the tool and replies with `role: "tool"` in a multi-turn thread.
5. Server streams reasoning tokens (`reasoning_content`) and final answer.

Start the server first:
    python app.py

Then run:
    python examples/07_tool_calling_agent.py
"""

import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Retrieve current stock market quote for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL, NVDA)",
                    }
                },
                "required": ["symbol"],
            },
        },
    }
]

messages = [
    {"role": "user", "content": "What is the current price of NVDA?"}
]

print("=== Step 1: Initial request with tools ===")
resp = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages,
    tools=tools,
)

msg = resp.choices[0].message
if msg.reasoning_content:
    print(f"[Thinking]:\n{msg.reasoning_content}\n")

if msg.tool_calls:
    for call in msg.tool_calls:
        print(f"[Tool Call Detected]: {call.function.name}({call.function.arguments})")
        # Append assistant tool call turn
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
            ],
        })
        # Simulate local tool execution
        tool_result = json.dumps({"symbol": "NVDA", "price": 142.50, "currency": "USD"})
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.function.name,
            "content": tool_result,
        })
        print(f"[Tool Executed -> Output Sent]: {tool_result}")

print("\n=== Step 2: Second turn with tool result ===")
resp2 = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages,
    tools=tools,
)

msg2 = resp2.choices[0].message
if msg2.reasoning_content:
    print(f"[Thinking]:\n{msg2.reasoning_content}\n")
print(f"[Final Answer]:\n{msg2.content}")
