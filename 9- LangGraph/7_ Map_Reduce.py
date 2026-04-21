from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import Annotated, TypedDict, Literal
from rich import print
from langchain.tools import tool
from langgraph.types import Command, Send
import operator

load_dotenv()
llm = ChatGroq(model="groq/compound-mini", temperature=0.5)

class State(TypedDict):
    subject: list[str]
    jokes: Annotated[dict, operator.ior]        # Inclusive OR operator is used to merge dictionaries
    best: str
                                                # Here Inclusive OR is used as reducer to combine all generated jokes into a single dictionary, and then the best joke is selected from that combined dictionary.
def Orchestrator (state: State):
    sends = [Send("generate_joke", {"Subject": s}) for s in state["subject"]]
    return Command(update={}, goto=sends)
                                                # 'Command' is used for controlling graph execution. It accepts four parameters: update, goto, resume, and graph.
                                                
                                                # Send command is used to generate the jokes for each subject in parallel, and the results are merged using the inclusive OR operator defined in the State type.

def generate_joke(worker_input: dict):
    subject = worker_input["Subject"]
    prompt = f"""You are a joke generator. Generate a joke about the given Subject: {subject}."""
    response = llm.invoke(prompt)
    print(f"\nGenerating joke for {subject}... \n{response.content}")
    return {"jokes": {subject: response.content}}

def best_joke(state: State) -> str:
    jokes = state["jokes"]
    prompt = f"""You are a joke critic. Given the following jokes, select the best one joke only: {jokes}."""
    response = llm.invoke(prompt)
    best = response.content
    return {"best": best}


graphBuilder = StateGraph(State)
graphBuilder.add_node("Orchestrator", Orchestrator)
graphBuilder.add_node("generate_joke", generate_joke)
graphBuilder.add_node("best_joke", best_joke)

graphBuilder.add_edge(START, "Orchestrator")
# graphBuilder.add_edge("Orchestrator", "generate_joke") 
graphBuilder.add_edge("generate_joke", "best_joke")
graphBuilder.add_edge("best_joke", END)

graph = graphBuilder.compile()
print("\nInvoking MapReduce Graph...\n")
print(graph.invoke(State(subject=["chickens", "programmers", "lawyers"])))