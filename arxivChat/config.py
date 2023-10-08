import logging
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()

BASE_DIR = Path(__file__).parent.parent.absolute()

# load arxiv config ###############
ARXIV_IDS = ["1603.02754", "1706.09516"]  # xgboost, catboost
ARXIV_DIR = "arxiv" # directory to save the arxiv papers
ARXIV_DIR = BASE_DIR / ARXIV_DIR

# split document config ###########
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# vector store config ############
VS_DIR = "vector_store"
VS_DIR = BASE_DIR / VS_DIR

# retriever config #################
RETRIEVER_MODEL = "gpt-3.5-turbo"  # "gpt-4" or "gpt-3.5-turbo", only relevant for retrieval_mode = "multiquery"
RETURN_DOCS = 4  # number of similar doccuments to return
RETRIEVER_MODE = "regular"  # "multiquery" or "regular"

# generate model config #####################
GENERATE_MODEL = "gpt-3.5-turbo" # "gpt-3.5-turbo" or "gpt-4"
TEMPERATURE = 0
WITH_CONTEXT = True
# system prompt: generate without context
SYSTEM_PROMPT="""You are an expert and your job is to answer technical questions. \ 
                 Keep your answers technical and based on facts, do not hallucinate features. """
# context template: generate with context
CONTEXT_TEMPLATE = """Use the following pieces of context to answer the question at the end. \
                    If you cannot find the relevant context from the source documents, \
                    say that you do not have enough context to answer. \
                    When you encounter a reference to a math eqn, express the formula in full. \
                    If you don't know the answer, just say that you don't know, \
                    don't try to make up an answer. \
                    Please provide a detailed and accurate response. \
                    {context}
                    Question: {question}
                    Helpful Answer:"""

# chat model config #############################
CHAT_TEMPLATE = """
    Use the following pieces of context to answer the question at the end. \
    {context}
    If you still cant find the answer, just say that you don't know, don't try to make up an answer. \
    When you encounter a reference to a math eqn, express the formula in full and provide the definitions of the terms. \
    When a question is not related to the context, ignore the context and continue the conversation as a helpful assistant. \
    You can also look into chat history. \
    {chat_history}
    Question: {question}
    Answer:
    """

# CHAT_TEMPLATE = """
#     Use the following pieces of context to answer the question at the end.
#     {context}
#     If you still cant find the answer, just say that you don't know, don't try to make up an answer.
#     You can also look into chat history.
#     {chat_history}
#     Question: {question}
#     Answer:
#     """



