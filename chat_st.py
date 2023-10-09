import os 
import streamlit as st
from dotenv import find_dotenv, load_dotenv  # find .env file
import openai

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI

from arxivChat.store import add_urls_to_vs
from arxivChat.config import VS_DIR_URL
from arxivChat.chat import (create_chat, 
                            get_answer, 
                            process_latex_output, 
                            format_answer)


# title of the webpage ############################
st.header("Chat with URLs 💬 📚")


# get openai api key #################################
_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"] 


# input a list of urls ##################
values = st.text_input(label = "Enter a list of urls, separated by commas", 
                       value = "https://en.wikipedia.org/wiki/Economy_of_Canada, https://lilianweng.github.io/posts/2023-06-23-agent/",
                       help = """Enter a list of urls, separated by commas, \
                            example https://en.wikipedia.org/wiki/Economy_of_Canada, https://lilianweng.github.io/posts/2023-06-23-agent/
                       """
                       )
urls = values.split(',')
urls = [id.strip() for id in urls] 


# add the documents to the vector store #####################
if st.button("Add to Vector Store"):
    add_urls_to_vs(urls = urls,
                   vs_dir=str(VS_DIR_URL))
    st.write("Documents added to Vector Store")


# whether to chat with context or without context ###############
with_context = st.selectbox("With Context:", [True, False], key='context')

# select the language model ######################
models = ["gpt-3.5-turbo", "gpt-4"]
model = st.selectbox('Select an Open AI Chat Model:', models)


# select the temperature #########################
temperature = st.slider('Select the temperature',
                        min_value = 0,
                        max_value = 1,
                        help="Select the temperature, 0 means no randomness")


# initialize the chat message history ########################
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about your data"}
    ]


# initiate memory for conversation ##############
@st.cache_resource
def init_memory():
    if with_context:
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
chat_engine = create_chat(model_name = model,
                          temperature = temperature,
                          with_context = with_context,
                          vs_dir=str(VS_DIR_URL),
                          memory=init_memory())
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








