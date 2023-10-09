import argparse
import os
from pprint import pprint

import openai
from dotenv import find_dotenv, load_dotenv  # find .env file
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from tenacity import wait_random_exponential  # for exponential backoff
from tenacity import retry, stop_after_attempt

from arxivChat.config import (
    SYSTEM_PROMPT,
    CONTEXT_TEMPLATE,
    GENERATE_MODEL,
    RETRIEVER_MODE,
    TEMPERATURE,
    VS_DIR,
    WITH_CONTEXT,
    logger,
)
from arxivChat.store import add_to_vs
from arxivChat.retrieve import create_retriever


# chat completion with retry #########################################
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def generate_and_print(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = GENERATE_MODEL,
    temperature: str = TEMPERATURE,
    **kwargs,
):
    """ Generate answer from open ai API

    Args:
        user_prompt (str): user question
        system_prompt (str, optional): instruction to the system. 
                    Defaults to SYSTEM_PROMPT.
        model (str, optional): model name, either "gpt-3.5-turbo" or "gpt-4" 
                Defaults to GENERATE_MODEL.
        temperature (str, optional): temperature of the model, 0.0 means no randomness 
                Defaults to TEMPERATURE.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    responses = completion_with_backoff(
        model=model, messages=messages, temperature=temperature, **kwargs
    )
    return responses.choices[0].message.content


# chat completion with retrieval ################################
def generate_with_retrieval(
    user_prompt: str,
    vs_dir: str = str(VS_DIR),
    model_name: str = GENERATE_MODEL,
    temperature: str = TEMPERATURE,
    template: str = CONTEXT_TEMPLATE,
):
    """ Generate answers with retrieval 

    Args:
        user_prompt (str): user question
        vs_dir (str, optional): directory that contains the vector store. 
                        Defaults to str(VS_DIR).
        model_name (str, optional): model name, either "gpt-3.5-turbo" or "gpt-4" 
                Defaults to GENERATE_MODEL.
        temperature (str, optional): temperature of the model, 0.0 means no randomness 
                Defaults to TEMPERATURE.
        template (str, optional): prompt for context extraction 
            Defaults to CONTEXT_TEMPLATE.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    retriever = create_retriever(vs_dir=vs_dir)

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    result = qa_chain({"query": user_prompt})

    return result["result"], result["source_documents"]


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
        default="Write down the formula for the Gain in the XGBoost algorithm",
    )
    args = parser.parse_args()
    query = args.query
    logger.info(f"The question is \n{query}")

    if not WITH_CONTEXT:
        logger.info("Answer question without context")
        response = generate_and_print(query)
        logger.info("Response \n")
        pprint(response)
    else:
        logger.info("Add arxiv documents to vector store")
        db = add_to_vs()
        logger.info(f"Answer question with context and {RETRIEVER_MODE} retriever")
        response, source_docs = generate_with_retrieval(query)
        logger.info("Response \n")
        pprint(response)
        logger.info("Most relevant document \n")
        pprint(source_docs[0].page_content)
