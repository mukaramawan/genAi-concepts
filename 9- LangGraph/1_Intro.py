from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from rich import print

class State(TypedDict):
    messages : Annotated[list, add_messages]

def chatBots(state: State):
    print("\nInside chatBots node: ", state)
    return {"messages": ["Hello! How can I assist you today?"]}

def sampleNode(state: State):
    print("\nInside sampleNode: ", state)
    return {"messages": ["Message from sampleNode"]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatBots", chatBots)
graph_builder.add_node("sampleNode", sampleNode)

graph_builder.add_edge(START, "chatBots")
graph_builder.add_edge("chatBots", "sampleNode")
graph_builder.add_edge("sampleNode", END)

graph = graph_builder.compile()

updated_state = graph.invoke(State({"messages": ["Hi!"]}))
print("\nUpdated State after graph execution: ", updated_state)


"""
What is LangGraph: LangGraph is a framework used to build stateful, multi-step AI workflows using nodes and edges, similar to a flowchart.

What is Node: 
A node is a function, that processes input state and returns updated state.

What is Edge:
An edge defines the connection or flow between different nodes (which node runs next).

What is State:
State is a shared data structure that stores and passes data between nodes.

What is Graph in LangGraph:
A graph is the complete workflow made up of nodes and edges that defines the execution flow.
"""