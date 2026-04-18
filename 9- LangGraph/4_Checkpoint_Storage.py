from asyncio import graph
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import Annotated, TypedDict
from rich import print
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.mongodb import MongoDBSaver  
from langgraph.checkpoint.memory import InMemorySaver  

import os

load_dotenv()
llm = ChatGroq(model="groq/compound-mini", temperature=0.5)

class State(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chatBot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatBot", chatBot)

graph_builder.add_edge(START, "chatBot")
graph_builder.add_edge("chatBot", END)


def graph_with_checkpointing(checkpointer):
    graph = graph_builder.compile(checkpointer)
    return graph


checkpointer = InMemorySaver()
graph = graph_with_checkpointing(checkpointer)

config = {
            "configurable": {
                "thread_id": "Arqam"    # User ID
            }
        }


print("\nInvoking graph with InMemorySaver checkpointing...\n")
print(graph.invoke(State({"messages": 'HI there my name is Mukaram Awan'}), config=config))
print("State History after 1st invocation:\n\n")
print(list(graph.get_state_history(config)))

print("\nInvoking graph with InMemorySaver checkpointing...\n")
print(graph.invoke(State({"messages": "What is my name"}), config=config))
print("State History after 2nd invocation:\n\n")
print(list(graph.get_state_history(config)))


# Long Term Memory
# DB_URI = os.getenv("MONGODB_URI")
# with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    
#     graph = graph_with_checkpointing(checkpointer)

#     updated_state = graph.invoke(State({"messages": ["What is my name?"]}), config=config)
#     print("\nUpdated State after graph execution: ", updated_state)



"""
What is Checkpointing Memory in LangGraph:
Checkpointing is a mechanism to save the state of a graph at each step. It allows you to persist the state across multiple invocations of the graph.

What is a Checkpointer:
A checkpointer handles the actual storage of the graph's state. 
- InMemorySaver stores the data temporarily in the computer's RAM (cleared when the script stops).
- MongoDBSaver (or other DB savers) stores the data persistently in a database, surviving script restarts.

What is a Thread ID (config):
A thread_id is a unique identifier used to separate different conversations or users. It ensures that the checkpointer fetches the correct chat history for a specific user (e.g., separating "Arqam's" chat from "Ali's" chat).

What is State History:
LangGraph saves the state (StateSnapshot) at every node execution, you can retrieve the full sequence of past states (time-travel).
"""