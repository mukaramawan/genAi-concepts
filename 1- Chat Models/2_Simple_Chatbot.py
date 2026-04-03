from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

print("Welcome to the Chatbot! Type '0' to exit.")

messages = []

while True:
    prompt = input("You: ")
    messages.append(prompt)
    if prompt == '0':
        print("Goodbye!")
        break
    response = model.invoke(messages)
    messages.append(response.content)
    print(f'Bot: {response.content}')

    # Problems
    # 1. No Role Seperations
    # 2. Weak Conversation Structure
    # 3. Memory keeps growing.
    # 4. No Summerization of old conversations. etc.



