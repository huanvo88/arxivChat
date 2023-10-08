import os 
import streamlit as st
from dotenv import find_dotenv, load_dotenv  # find .env file
import openai

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI

from arxivChat.config import WITH_CONTEXT
from arxivChat.chat import (create_chat, 
                            get_answer, 
                            process_latex_output, 
                            format_answer)

st.header("Chat with docs 💬 📚")


# get openai api key #################################
_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"] 

# initialize the chat message history ########################
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about your data"}
    ]

# initiate memory for conversation ##############
@st.cache_resource
def init_memory():
    if WITH_CONTEXT:
        return ConversationSummaryBufferMemory(
            llm=ChatOpenAI(temperature=0),
            output_key='answer',
            memory_key='chat_history',
            return_messages=True)
    else:
        return ConversationSummaryBufferMemory(
            llm=ChatOpenAI(temperature=0),
            output_key='text',
            memory_key='chat_history',
            return_messages=True)

# create chat engine with memory ##########################
chat_engine = create_chat(memory=init_memory())
history = []

# set up chat UI ###################################
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    prompt = process_latex_output(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = get_answer(chat_engine, prompt, history)
            response = format_answer(result)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history
            history.append((prompt, response))








