import os
from typing import List

import openai
from dotenv import find_dotenv, load_dotenv  # find .env file
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document 

from arxivChat.config import (ARXIV_IDS, 
                              ARXIV_DIR, 
                              VS_DIR,
                              VS_DIR_URL, 
                              URLS,
                              logger
)
from arxivChat.load import load_arxiv_ids, load_urls
from arxivChat.split import md_split


# create and persist vector store ####################################
def create_vs(document_chunks: List[Document], persist_directory: str):
    """Create and persist vector store, we will use the OpenAIEmbeddings
       to embed the text, and Chroma to store the vectors
    Args:
        document_chunks: List of langchain documents
        persist_directory (str): directory to persist the vector store
    """
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(document_chunks, 
                               embeddings, 
                               persist_directory=persist_directory)
    db.persist()
    return db


# load the vector store #################################################
def load_vs(persist_directory: str):
    """Load the vector store from a directory

    Args:
        persist_directory (str): directory to the vector store
    """
    embeddings = OpenAIEmbeddings()
    db = Chroma(embedding_function=embeddings, 
                persist_directory=persist_directory)
    return db


# fetch the arxiv papers and load the documents to a vector store
def add_to_vs(arxiv_ids: List[str] = ARXIV_IDS,
              arxiv_dir: str = str(ARXIV_DIR),
              vs_dir: str = str(VS_DIR),
              **kwargs):
    """Fetch the arxiv documents and put them in the vector stores

    Args:
        arxiv_ids (List[str], optional): List of arxiv ids. Defaults to ARXIV_IDS.
        arxiv_dir (str, optional): Directory to save the arxiv papers. 
                        Defaults to str(ARXIV_DIR).
        vs_dir (str, optional): directory to save the vector store. 
                    Defaults to str(VS_DIR).
    """

    logger.info("Load the arxiv ids into documents")
    documents = load_arxiv_ids(arxiv_ids, arxiv_dir)

    logger.info("Split the documents based on Markdown")
    document_chunks = md_split(documents, **kwargs)

    logger.info("Create the vector store")
    db = create_vs(document_chunks, vs_dir)

    return db 


# fetch the urls and load to a vector store
def add_urls_to_vs(urls: List[str] = URLS, 
                   vs_dir: str = str(VS_DIR_URL),
                   **kwargs):
    """Fetch the texts from the urls and add to vector store

    Args:
        urls (List[str]): List of urls. Default is URLS
        vs_dir (str): Directory to store the vector stores. 
            Default is VS_DIR_URL
    """
    logger.info("Load the urls into documents")
    documents = load_urls(urls)

    logger.info("Split the documents based on Markdown")
    document_chunks = md_split(documents, **kwargs)

    logger.info("Create the vector store")
    db = create_vs(document_chunks, vs_dir)

    return db 


if __name__ == "__main__":
    logger.info("Extract openai key")
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    logger.info("Add arxiv documents to vector store")
    db = add_to_vs(ARXIV_IDS, str(ARXIV_DIR), str(VS_DIR))

    logger.info("Add url documents to vector store")
    db_url = add_urls_to_vs(URLS, str(VS_DIR_URL))