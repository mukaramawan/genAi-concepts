from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from rich import print
from langchain_core.tools import tool

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the characters in the given text.
    """
    return len(text)

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)
llm_tool = llm.bind_tools(tools=[get_text_length])

response = llm.invoke("Give me the length of the word, 'Hello! How are you?'")
print(response)

print("")
print("")

response2= llm_tool.invoke("Give me the length of the word, 'Hello! How are you?'")
print(response2)

print("")
print("")

print(response2.tool_calls)

# Tool Message
# print(get_text_length.invoke({'name': 'get_text_length', 'args': {'text': 'Hello! How are you?'}, 'id': 'fc_31f26576-34bd-47fa-8d4e-e5839440acf6', 'type': 'tool_call'}))
# Output
# ToolMessage(content='19', name='get_text_length', tool_call_id='fc_31f26576-34bd-47fa-8d4e-e5839440acf6')

"""
In tool binding, we gave tools to the llm, 
and in tool calling, the llm does not execute the tools directly 
but only choose the tool to call.
"""

