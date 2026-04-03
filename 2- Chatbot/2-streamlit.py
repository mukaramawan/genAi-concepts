import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Setup & Configuration
load_dotenv()
st.set_page_config(page_title="English Language Assistant", page_icon="🤖")
st.title("🤖 English Language Assistant")

# 2. Initialize the Model
# Ensure your GROQ_API_KEY is in your .env file
model = ChatGroq(model="openai/gpt-oss-20b")

# 3. Initialize Session State for Messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant for English Language.")
    ]

# 4. Display Chat History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# 5. Chat Input Logic
if prompt := st.chat_input("Type your message here..."):
    # Append user message to state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response = model.invoke(st.session_state.messages)
        st.write(response.content)
        
    # Append assistant response to state
    st.session_state.messages.append(AIMessage(content=response.content))