from typing import Optional
from langgraph.graph import StateGraph, START, END
from typing_extensions import Literal, TypedDict
from rich import print
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

llm = ChatMistralAI(model="magistral-small-latest", temperature=0.5)

class State(TypedDict):
    user_query : str
    llm_response : str
    is_good_response : Optional[bool]
    final_response : Optional[str]

def chatBot(state: State):
    print("\n\nInside chatBots node: ", state)
    response = llm.invoke(state.get("user_query"))
    return {"llm_response": response.content[-1]}

def response_evaluation(state: State) -> Literal["endNode", "secondaryChatbot"]:
    print("\n\nInside Response Evaluation: ", state)
    if True:
        return "endNode"
    return "secondaryChatbot"

def secondaryChatbot(state: State):
    print("\n\nInside secondaryChatbot: ", state)
    response = llm.invoke("You must have to provide any answer related to user query that can also be an wrong guess: " + state.get("user_query"))
    return {"llm_response": response.content[-1]}

def endNode(state: State):
    print("\n\nInside endNode: ", state)
    return {"final_response": "Thank you for visiting the website!"}


graph_builder = StateGraph(State)

graph_builder.add_node("chatBot", chatBot)
graph_builder.add_node("secondaryChatbot", secondaryChatbot)
graph_builder.add_node("endNode", endNode)

graph_builder.add_edge(START, "chatBot")
graph_builder.add_conditional_edges("chatBot", response_evaluation)
graph_builder.add_edge("secondaryChatbot", "endNode")
graph_builder.add_edge("endNode", END)

graph = graph_builder.compile()

updated_state1 = graph.invoke(State({"user_query": "What is the weather of Lahore today?"}))
print("\nUpdated State after graph execution: ", updated_state1)

"""The invoke() method in the StateGraph re-executes the graph from the beginning with each new state invocation (e.g., updated_state1, updated_state2). It does not retain or share memory or state between invoke() calls. 

Updated states 2 and 3 are demonstrating this problem because they are being invoked with new states that do not have the context of previous executions.

To avoid this issue, In LangGraph, you can use the concept of memory or checkpointing that enables persistence in AI agent workflows.

"""

updated_state2 = graph.invoke(State({"user_query": "Weather of Lahore today is 33°C° and cloudy."}))
print("\nUpdated State 2 after graph execution: ", updated_state2)

updated_state3 = graph.invoke(State({"user_query": "What is the weather of Lahore today?"}))
print("\nUpdated State 3 after graph execution: ", updated_state3)