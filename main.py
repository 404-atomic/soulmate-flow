import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI()
graph_builder = StateGraph(State)

# Helper to send user message and get LLM response
def user_and_llm_response(state: State, user_msg: str):
    input(f"Press Enter to send: '{user_msg}'")
    print(f"[You] {user_msg}")
    state["messages"].append({"role": "user", "content": user_msg})
    response = llm.invoke(state["messages"])
    print("[Assistant]", response.content)
    return {"messages": [response]}

# Node 1: Hi
def node1(state: State):
    return user_and_llm_response(state, "hello")

# Node 2: My name is Kent
def node2(state: State):
    return user_and_llm_response(state, "my name is kenz")

# Node 3: What is my name?
def node3(state: State):
    return user_and_llm_response(state, "what is my name")

# Graph setup
graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)

graph_builder.add_edge(START, "node1")
graph_builder.add_edge("node1", "node2")
graph_builder.add_edge("node2", "node3")
graph_builder.add_edge("node3", END)

graph = graph_builder.compile()

# Run interactively
if __name__ == "__main__":
    print("LangGraph chat simulation step-by-step with system response")
    print("------------------------------------------------------------")

    state = {"messages": []}
    for step in graph.stream(state):
        for node, output in step.items():
            if "messages" in output:
                state["messages"].extend(output["messages"])  # accumulate history
