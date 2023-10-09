import os
from pprint import pprint
import re 
from typing import List, Tuple, Dict, Any

import openai
from dotenv import find_dotenv, load_dotenv  # find .env file
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from arxivChat.config import (CHAT_TEMPLATE, 
                        SYSTEM_PROMPT, 
                        GENERATE_MODEL, 
                        TEMPERATURE, 
                        VS_DIR, 
                        WITH_CONTEXT, 
                        logger)
from arxivChat.retrieve import create_retriever
from arxivChat.store import add_to_vs


def create_chat(
    model_name: str = GENERATE_MODEL,
    temperature: str = TEMPERATURE,
    with_context: str = WITH_CONTEXT,
    vs_dir: str = str(VS_DIR),
    system_prompt: str = SYSTEM_PROMPT,
    template: str = CHAT_TEMPLATE,
    memory: ConversationSummaryBufferMemory = None
):
    """Create a chatbot 

    Args:
        model_name (str, optional): open ai model, either 'gpt-3.5-turbo' or 'gpt-4'. 
                        Defaults to GENERATE_MODEL.
        temperature (str, optional): temperature of the model, 0.0 means no randomness 
                Defaults to TEMPERATURE.
        with_context (str, optional): whether to answer with context retrieval or not. 
                        Defaults to WITH_CONTEXT.
        vs_dir (str, optional): directory that contains the vector store. 
                    Defaults to str(VS_DIR).
        system_prompt (str, optional): system instruction. Defaults to SYSTEM_PROMPT.
        template (str, optional): prompt for chat with context retrieval. 
                        Defaults to CHAT_TEMPLATE.
        memory (ConversationSummaryMemory, optional): Memory summarization. Defaults to None.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    if with_context:
        retriever = create_retriever(vs_dir)
        if not memory:
            memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(temperature=0), 
                                                     memory_key="chat_history", 
                                                     return_messages=True,
                                                     output_key='answer')
        prompt_doc = PromptTemplate(template=template, 
                                    input_variables=["context", "question", "chat_history"])
        conversation = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            chain_type = "stuff",
            return_source_documents=True,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt_doc}
        )
    else:
        if not memory:
            memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(temperature=0), 
                                                     memory_key="chat_history", 
                                                     return_messages=True,
                                                     output_key='text')
        # Prompt
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        conversation = LLMChain(llm=llm, prompt=prompt, 
                                memory=memory
                                )

    return conversation

def process_latex_output(latex_str: str):
    """Clean up some latex equations to display

    Args:
        latex_str (str): text that contains latex formula
    """
    # replace \begin{equation} and \end{equation} with $$
    processed_str = re.sub(r'\\begin{equation}', '$$', latex_str)
    processed_str = re.sub(r'\\end{equation}', '$$', processed_str)

    # replace \begin{equation*} and \end{equation*} with $$
    processed_str = re.sub(r'\\begin{equation\*}', '$$', processed_str)
    processed_str = re.sub(r'\\end{equation\*}', '$$', processed_str)

    return processed_str


def get_answer(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: List[Tuple[str, str]],
):
    """Get an answer from a ConversationalRetrievalChain
    Args:
        chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object
        question (str): The question to ask
        chat_history (list[tuple[str, str]]): A list of tuples of (question, answer)
    Returns:
        str: The answer to the question
    """
    result = chain(
        {"question": question, "chat_history": chat_history},
        )
    return result


def format_answer(result: Dict[str, Any]):
    """Format the output from chatbot

    Args:
        result (Dict[str, Any]): output from chatbot
    """
    try:
        response = f"Answer:\t{result['answer']}"
    except Exception:
        response = f"Answer:\t{result['text']}"
    response = process_latex_output(response)
    return response


if __name__ == "__main__":
    logger.info("Extract openai key")
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    logger.info("Add arxiv documents to vector store")
    db = add_to_vs()

    logger.info(f"Answer with context = {WITH_CONTEXT}")
    qa_chat = create_chat(with_context=WITH_CONTEXT)
    
    # conversation ###########
    history = []
    question1 = "What is the formula for the Loss objective in the Xgboost algorithm?"
    result1 = get_answer(qa_chat, question1, history)
    response1 = format_answer(result1)
    pprint(response1)
    history.append((question1, response1))

    question2 = "What formula did I ask for in the previous question?"
    result2 = get_answer(qa_chat, question2, history)
    response2 = format_answer(result2)
    pprint(response2)
    pprint(history)
