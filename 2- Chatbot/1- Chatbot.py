from dotenv import load_dotenv

load_dotenv()

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatGroq(model="openai/gpt-oss-20b")

print("Welcome to the Chatbot! Type '0' to exit.")

messages = [SystemMessage(content="You are a helpful assistant for English Language.")]

while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == '0':
        print("Goodbye!")
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f'Bot: {response.content}')


