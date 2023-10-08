"""Set up the package."""
from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(BASE_DIR/"requirements.txt", "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# Read version 
with open(BASE_DIR/"arxivChat"/"VERSION", "r") as file:
    __version__ = file.read().strip()

setup(
    author="Huan Vo",
    name="arxivChat",
    version=__version__,
    description="Chat with Documents using LLM",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    license="MIT",
    url="https://github.com/huanvo88/arxivChat" 
)