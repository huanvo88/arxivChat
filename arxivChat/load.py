import os
import tarfile
from typing import List 

import latex2markdown
from langchain.document_loaders import DirectoryLoader, WebBaseLoader

import arxiv
from arxivChat.config import ARXIV_DIR, ARXIV_IDS, URLS, logger


def create_arxiv_filename(arxiv_id: str):
    """ Create file name from arxiv_id: replace '.' by '_'

    Args:
        arxiv_id (str): arxiv id, of the form 'xxxx.xxxx'
    """
    return arxiv_id.replace('.', '_')


def download_source(arxiv_ids: List[str], dirpath: str):
    """Download arxiv source and extract tarfile

    Args:
        arxiv_ids (list): List of arxiv ids, of the form "xxxx.xxxx"
        dirpath (str): directory to store the arxiv files
    """

    # create the directory
    os.makedirs(dirpath, exist_ok=True)

    for arxiv_id in arxiv_ids:
        logger.info(f"Fetch the paper {arxiv_id}")
        # get the paper metadata
        paper = next(arxiv.Search(id_list=[arxiv_id]).results())

        # download the source file
        filename = create_arxiv_filename(arxiv_id)
        tar_filename = f"{filename}.tar.gz"
        paper.download_source(dirpath=dirpath, filename=tar_filename)
        tar_gz_path = os.path.join(dirpath, tar_filename)

        # extract the .tar.gz file
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            tar.extractall(path=os.path.join(dirpath, filename))
            logger.info(f"Files extracted at {os.path.join(dirpath, filename)}")


def find_tex_files(dirpath: str):
    """ Find all the .tex files recursively within a directory

    Args:
        dirpath (str): directory to find the .tex files
    """
    tex_files = []
    for root, _, files in os.walk(dirpath):
        for file in files:
            if file.endswith(".tex"):
                tex_files.append(os.path.join(root, file))
    logger.info(f"Find {len(tex_files)} .tex files")
    return tex_files 


def tex_to_md(tex_paths: List[str]):
    """Convert tex files to markdown files

    Args:
        tex_paths (list): list of paths to the .tex files
    """
    for tex_path in tex_paths:
        with open(tex_path, "r") as f:
            latex_string = f.read()

        l2m = latex2markdown.LaTeX2Markdown(latex_string)

        markdown_string = l2m.to_markdown()

        md_path = tex_path.replace(".tex", ".md")
        with open(md_path, "w") as f:
            f.write(markdown_string)


def load_md_files(directory: str):
    """Find all markdown files in a directory and return a LangChain Document

    Args:
        directory (str): directory to search for markdown files

    Returns:
        langchain documents
    """
    dl = DirectoryLoader(directory, "**/*.md")
    documents = dl.load()
    return documents


def load_arxiv_ids(arxiv_ids: List[str] = ARXIV_IDS,
              arxiv_dir: str = str(ARXIV_DIR)):
    """Fetch the arxiv papers and return langchain documents

    Args:
        arxiv_ids (List[str], optional): List of arxiv ids. Defaults to ARXIV_IDS.
        arxiv_dir (str, optional): Directory to save the arxiv papers. 
                        Defaults to str(ARXIV_DIR).
    """
    logger.info("Fetch the arxiv sources")
    download_source(arxiv_ids, arxiv_dir) 

    logger.info(f"Find all the .tex files")
    tex_paths = find_tex_files(arxiv_dir)

    logger.info("Convert tex to md files")
    tex_to_md(tex_paths)

    logger.info("Find all md documents")
    documents = load_md_files(arxiv_dir)
    logger.info(f"We have {len(documents)} documents from {len(arxiv_ids)} arxiv_ids")

    return documents 


def load_urls(urls: List[str]=URLS):
    """Fetch the urls and return langchain documents

    Args:
        urls (List[str], optional): List of urls. Defaults to URLS.
    """
    logger.info("Fetch the urls")
    loader = WebBaseLoader(urls)
    documents = loader.load()
    logger.info(f"We have {len(documents)} documents from {len(urls)} urls")
    return documents


if __name__ == "__main__":
    logger.info("Load arxiv articles into documents")
    documents = load_arxiv_ids()
    logger.info("Load the urls into documents")
    document_url = load_urls()
