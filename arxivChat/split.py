from typing import List

import tiktoken
from langchain.schema.document import Document 
from langchain.text_splitter import MarkdownTextSplitter

from arxivChat.config import ARXIV_DIR, CHUNK_OVERLAP, CHUNK_SIZE, logger
from arxivChat.load import load_md_files


# count the number of tokens ##########################################
def count_tokens(documents: List[Document]):
    """Function to count the number of tokens in each document

    Args:
        documents: List of langchain documents
    """
    # We will need to count tokens in the documents, and for that we need the tokenizer
    tokenizer = tiktoken.encoding_for_model(model_name="text-davinci-003")
    token_counts = [len(tokenizer.encode(document.page_content)) for document in documents]
    return token_counts


# split documents into chunks #############################################
def md_split(documents: List[Document], 
             chunk_size: int=CHUNK_SIZE, 
             chunk_overlap: int=CHUNK_OVERLAP):
    """Split documents into chunks

    Args:
        documents: List of langchain documents
        chunk_size (int, optional): number of tokens for each chunk, defaults to 1000.
        chunk_overlap (int, optional): tokens overlap between chunks, defaults to 100.
    """
    md_text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_chunks = md_text_splitter.split_documents(documents)
    logger.info(f"We have {len(document_chunks)} document chunks")
    logger.info(f"Longest document has {max(count_tokens(document_chunks))} tokens")
    return document_chunks


if __name__ == "__main__":
    logger.info("Find all md documents")
    documents = load_md_files(ARXIV_DIR)
    logger.info(f"We have {len(documents)} markdown documents")

    logger.info("Split the documents based on Markdown")
    document_chunks = md_split(documents)
