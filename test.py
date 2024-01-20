from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate

import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")



chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)

messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="Write a Python script that trains a neural network on simulated data")
]


template = """
You are an expert data scientist with an expertise in building deep learning models. 
Explain the concept of {concept} in a couple of lines
"""
prompt = PromptTemplate(
    input_variables=["concept"],
    template=template,
)


formatted_prompt = prompt.format(concept="aids")
response=chat.invoke(formatted_prompt)

print(response.content,end='\n')
