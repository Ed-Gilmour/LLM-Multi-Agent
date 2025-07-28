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
    plan = llm.invoke(f"""
    You are a PLANNER.

    Your job is ONLY to produce a simple ordered list of steps needed to solve the given task.

    Do NOT explain, do NOT reason, and DO NOT solve the problem.

    Return ONLY the list of steps in the following exact format (no other text):

    1. Step one
    2. Step two
    3. Step three
    4... etc

    Task:
    {state['task']}
    """)
    print("\n--------------PLAN--------------\n\n" + plan.strip() + "\n\n----------------------------\n")
    return {**state, "plan": plan}

def worker_node(state: AgentState) -> AgentState:
    output = llm.invoke(f"""
    You are a WORKER.

    Your job is to follow the plan to complete the task and produce a user-ready answer or explanation.

    Do NOT add any plans.

    Do NOT check or review your work; just follow the steps and EXPLAIN your answer given the task.
    Follow the suggestions to produce the best result possible.

    Task:
    {state['task']}       

    Plan:
    {state['plan']}

    Previous output:
    {state.get("output", "No previous output")}

    Suggestions:
    {state.get("review", "No suggestions")}
    """)
    print("\n--------------OUTPUT--------------\n\n" + output.strip() + "\n\n----------------------------\n")
    return {**state, "output": output}

def reviewer_node(state: AgentState) -> AgentState:
    review = llm.invoke(f"""
    You are a REVIEWER.

    Your job is ONLY to critique and suggest improvements on the output given.

    Do NOT add new content, do NOT solve the problem, do NOT repeat the entire output, and do NOT provide a new answer.

    Task:
    {state['task']}       

    Plan:
    {state['plan']}

    Output to review:
    {state['output']}
    """)

    print("\n--------------REVIEW--------------\n\n" + review.strip() + "\n\n----------------------------\n")
    return {**state, "review": review}

def check_review_done(state):
    if state.get('review') != None:
        print("\nWork is done — output is complete.\n")
        return END
    else:
        print("\nWork not done — sending to reviewer.\n")
        return "reviewer"

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("worker", worker_node)
builder.add_node("reviewer", reviewer_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "worker")
builder.add_conditional_edges("worker", check_review_done)
builder.add_edge("reviewer", "worker")

graph = builder.compile()

while True:
    question = input("Ask a question: ")

    if question.lower() in ["quit", "exit"]:
        break

    result = graph.invoke({"task" : question})