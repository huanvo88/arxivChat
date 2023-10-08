import os
from typing import List

import openai
from dotenv import find_dotenv, load_dotenv  # find .env file
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document 

from arxivChat.config import ARXIV_IDS, ARXIV_DIR, VS_DIR, logger
from arxivChat.load import (download_source, 
                      load_md_files,
                      find_tex_files,
                      tex_to_md
)
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
              vs_dir: str = str(VS_DIR)):
    """Fetch the arxiv documents and put them in the vector stores

    Args:
        arxiv_ids (List[str], optional): List of arxiv ids. Defaults to ARXIV_IDS.
        arxiv_dir (str, optional): Directory to save the arxiv papers. 
                        Defaults to str(ARXIV_DIR).
        vs_dir (str, optional): directory to save the vector store. 
                    Defaults to str(VS_DIR).
    """

    logger.info("Fetch arxiv articles")
    download_source(arxiv_ids, arxiv_dir) 

    logger.info(f"Find all the .tex files")
    tex_paths = find_tex_files(arxiv_dir)

    logger.info("Convert tex to md files")
    tex_to_md(tex_paths)

    logger.info("Find all md documents")
    documents = load_md_files(arxiv_dir)
    logger.info(f"We have {len(documents)} markdown documents")

    logger.info("Split the documents based on Markdown")
    document_chunks = md_split(documents)

    logger.info("Create the vector store")
    db = create_vs(document_chunks, vs_dir)

    return db 


if __name__ == "__main__":
    logger.info("Extract openai key")
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    logger.info("Create the vector store")
    db = add_to_vs(ARXIV_IDS, str(ARXIV_DIR), str(VS_DIR))
