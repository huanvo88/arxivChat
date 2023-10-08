import argparse
import os
from pprint import pprint

import openai
from dotenv import find_dotenv, load_dotenv  # find .env file
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

from arxivChat.config import RETRIEVER_MODE, RETRIEVER_MODEL, RETURN_DOCS, VS_DIR, logger
from arxivChat.store import load_vs


def create_retriever(
    vs_dir: str = str(VS_DIR), 
    return_docs: int = RETURN_DOCS, 
    mode: str = RETRIEVER_MODE, 
    retrieval_model: str = RETRIEVER_MODEL
):
    """Create the retriever object from the vector store,
       mode = 'regular' or 'multiquery';
       for 'multiquery' mode we need to use an LLM

    Args:
        vs_dir (str, optional): directory to the vector store, defaults to VS_DIR.
        return_docs (int, optional): number of docs to return. Defaults to RETURN_DOCS.
        mode (str, optional): retrieval mode, either 'regular' or 'multiquery'. 
                        Defaults to RETRIEVER_MODE.
        retrieval_model (str, optional): retrieval model, only relevent for 'multiquery' 
                    Defaults to RETRIEVER_MODEL.
    """
    vs = load_vs(vs_dir)
    retriever = vs.as_retriever(search_kwargs=dict(k=return_docs))
    if mode == "regular":
        return retriever
    elif mode == "multiquery":
        llm = ChatOpenAI(model=retrieval_model, temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
        return retriever_from_llm


if __name__ == "__main__":
    logger.info("Extract openai key")
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # input question
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="input the query to find similar document",
        default="What is the approximate split finding algorithm in Xgboost",
    )
    args = parser.parse_args()
    query = args.query

    logger.info(f"Create {RETRIEVER_MODE} retriever")
    retriever = create_retriever(str(VS_DIR), mode=RETRIEVER_MODE)
    logger.info(f"Relevant documents for the question \n{query}")
    relevant_docs = retriever.get_relevant_documents(query=query)
    logger.info(f"Top document is \n")
    pprint(relevant_docs[0].page_content)
