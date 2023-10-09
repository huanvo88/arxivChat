from typing import List

import tiktoken
from langchain.schema.document import Document 
from langchain.text_splitter import (MarkdownTextSplitter, 
                                     RecursiveCharacterTextSplitter
)

from arxivChat.config import (CHUNK_OVERLAP, 
                              CHUNK_SIZE,
                              SPLIT_MODE, 
                              logger
)
from arxivChat.load import load_arxiv_ids


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
             chunk_overlap: int=CHUNK_OVERLAP,
             split_mode: str = SPLIT_MODE):
    """Split documents into chunks

    Args:
        documents: List of langchain documents
        chunk_size (int, optional): number of tokens for each chunk. 
                    Default is config.CHUNK_SIZE.
        chunk_overlap (int, optional): tokens overlap between chunks. 
                    Default is config.CHUNK_OVERLAP
        split_mode (str, optional): whether to split using markdown header or text split.
                Default is config.SPLIT_MODE
    """
    if split_mode == "markdown":
        logger.info(f"Split texts with MarkdownTextSplitter")
        text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        logger.info(f"Split texts with RecursiveCharacterTextSplitter")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
    document_chunks = text_splitter.split_documents(documents)
    logger.info(f"We have {len(document_chunks)} document chunks")
    logger.info(f"Longest document has {max(count_tokens(document_chunks))} tokens")
    return document_chunks


if __name__ == "__main__":
    logger.info("Load arxiv articles into documents")
    documents = load_arxiv_ids()

    logger.info("Split the documents based on Markdown")
    document_chunks = md_split(documents)
