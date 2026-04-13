from dotenv import load_dotenv
import os 
import requests
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from rich import print

load_dotenv()

# =========================
# ToolS
# =========================

@tool
def get_current_weather(location: str) -> str:
    """Return the current weather of the given city"""

    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    # Added &units=metric to get actual Celsius values
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Could not retrieve weather data for {location}. Please check the city name and try again."
    
    temp = data["main"]["temp"] 
    desc = data["weather"][0]["description"]

    return f"Weather in {location}: {desc}, {temp}°C"


tavilyClient = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    
    response = tavilyClient.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# =========================
# LLM Setup and Agent Loop
# =========================

llm = ChatMistralAI(model="mistral-large-latest") 
# tools_dict = {"get_news": get_news, "get_current_weather": get_current_weather}
# llm_with_tools = llm.bind_tools(tools=[get_news, get_current_weather])

# messages = [] 
# print("Type 'exit' to quit.")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break
    
#     messages.append(HumanMessage(content=user_input))

#     while True:
#         response = llm_with_tools.invoke(messages)
#         messages.append(response)

#         if response.tool_calls:
#             for tool_call in response.tool_calls:
#                 tool_name = tool_call['name']
#                 approval = input(f"Agent wants to call {tool_name}, Approve? (y/n): ").lower()
                
#                 if approval != 'y':
#                     print("Tool call rejected.")
#                     messages.append(ToolMessage(content="User rejected tool call.", tool_call_id=tool_call['id']))
#                     continue

#                 tool_output = tools_dict[tool_name].invoke(tool_call)
#                 print(f"Tool response: {tool_output}")
                
#                 messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call['id']))

#             continue
#         else: 
#             print(f"\nAI: {response.content}")
#             break

# The followig code is a simplified version of the above code using the create_agent from langchain which provides high level abstractions and automates everthinng including the tool calling and message management.

@wrap_tool_call
def human_approval(request, handler):
    tool_name = request.tool_call['name']
    approval = input(f"Agent wants to call {tool_name}, Approve? (y/n): ").lower()
    
    if approval != 'y':
        print("Tool call rejected.")
        return ToolMessage(content="User rejected tool call.", tool_call_id=request.tool_call['id'])
    
    return handler(request)


agent = create_agent(model=llm,
                     tools=[get_news, get_current_weather], 
                     system_prompt="You are a helpful assistant. Be concise and accurate.",
                     middleware=[human_approval]
)

print("Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(f"\nAI: {response['messages'][-1].content}\n")

