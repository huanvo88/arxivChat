import os
import tarfile
from typing import List 

import latex2markdown
from langchain.document_loaders import DirectoryLoader

import arxiv
from arxivChat.config import ARXIV_DIR, ARXIV_IDS, logger


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


if __name__ == "__main__":
    logger.info("Fetch arxiv articles")
    download_source(ARXIV_IDS, ARXIV_DIR)
    
    logger.info(f"Find all the .tex files")
    tex_paths = find_tex_files(str(ARXIV_DIR))

    logger.info("Convert tex to md files")
    tex_to_md(tex_paths)

    logger.info("Load all md documents")
    documents = load_md_files(ARXIV_DIR)
    logger.info(f"We have {len(documents)} markdown documents")
