from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from rich import print
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the characters in the given text.
    """
    return len(text)

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)
llm_tool = llm.bind_tools(tools=[get_text_length])

messages = []
tools = {
    get_text_length.name: get_text_length,
}

query = input("You: ")
messages.append(HumanMessage(query))

result = llm_tool.invoke(messages)
messages.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    messages.append(tool_message)

#print(messages)

final_response = llm_tool.invoke(messages)
print(final_response.content)    


