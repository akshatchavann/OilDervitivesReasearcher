import os
import streamlit as st
import pickle
import time
import faiss
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from scrape import scrape_and_save_text
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0.9, max_tokens=500)

file_path = "faiss_store_openai.pkl"
filename = 'output.txt'
urls = ['https://www.mosaicml.com/blog/mpt-7b']
filename = scrape_and_save_text(urls[0], filename)
loader = UnstructuredFileLoader("output.txt")

data = loader.load()

#split data
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1000
)
docs = text_splitter.split_documents(data)

# #embed data
embeddings = OpenAIEmbeddings()
vectorstore_openai = FAISS.from_documents(docs, embeddings)
# Assuming you have access to the embedding vectors as a NumPy array

# Print the shape of the embedding vectors
#print(print(dir(vectorstore_openai)))
# time.sleep(2)

# # #save data in vector db
# Instead of using pickle, use FAISS's built-in index serialization
faiss.write_index(vectorstore_openai.index, file_path)

print("mama we made it")


