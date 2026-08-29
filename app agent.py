import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from groq import Groq
from ddgs import DDGS
import json

st.set_page_config(page_title="Research & Calculate Assistant", page_icon="🤖")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def calculator(expression):
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: expression contains disallowed characters"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

def web_search(query):
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search error: {str(e)}"

tools = [
    {"type": "function", "function": {"name": "calculator",
        "description": "Evaluates a mathematical expression and returns the numeric result.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "A math expression, e.g. '500 * 1.08'"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {"name": "web_search",
        "description": "Searches the web for current, real-time information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query"}},
            "required": ["query"]}}}
]

available_functions = {"calculator": calculator, "web_search": web_search}

def run_agent(user_question, max_steps=5):
    messages = [{"role": "user", "content": user_question}]
    trace = []
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b", messages=messages, tools=tools, tool_choice="auto"
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                result = available_functions[func_name](**func_args)
                trace.append((func_name, func_args, result))
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                  "name": func_name, "content": str(result)})
        else:
            return msg.content, trace
    return "Max steps reached without a final answer.", trace

# --- UI ---
st.title("🤖 Research & Calculate Assistant")
st.write("Ask questions that need real-time information and/or calculations — the agent decides which tools to use.")

question = st.text_input("Ask a question:", placeholder="e.g., What's the current price of Bitcoin, and what's 0.05 BTC worth?")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Thinking..."):
            answer, trace = run_agent(question)

        if trace:
            with st.expander("🔍 See agent's reasoning steps"):
                for func_name, func_args, result in trace:
                    st.write(f"**Called `{func_name}`** with `{func_args}`")
                    st.write(f"Result: {result}")
                    st.write("---")

        st.subheader("Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question first.")
