from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict

llm = OllamaLLM(model="gemma3")

class AgentState(TypedDict):
    task: str
    plan: str
    output: str
    review: str

def planner_node(state: AgentState) -> AgentState:
    plan = llm.invoke(f"You are a planner. Break this task into steps:\n{state['task']}")
    return {**state, "plan": plan}

def worker_node(state: AgentState) -> AgentState:
    output = llm.invoke(f"""You are a specialized AI agent.
    Your job is to complete a task by following the plan exactly and producing a final answer with an explanation.
    Here is the plan you must follow:

    {state['plan']}

    Now complete the task. Do not describe a response to the planner or say you're following the plan. Explain your response.""")
    return {**state, "output": output}

def reviewer_node(state: AgentState) -> AgentState:
    review = llm.invoke(f"You are a reviewer. Improve or critique this output:\n{state['output']}")
    return {**state, "review": review}

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("worker", worker_node)
builder.add_node("reviewer", reviewer_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "worker")
builder.add_edge("worker", "reviewer")
builder.add_edge("reviewer", END)

graph = builder.compile()

while True:
    question = input("Ask a question: ")

    if question.lower() in ["quit", "exit"]:
        break

    result = graph.invoke({"task" : question})
    print("\n----------------------------\n\n" + result["output"] + "\n----------------------------\n")