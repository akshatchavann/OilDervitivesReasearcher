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



from dotenv import load_dotenv
load_dotenv()

st.title("Derivitives Research Based on current news")
st.sidebar.title("Links to Current Articiles/Papers")


urls = []
for i in range(1):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatOpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    #load data
    filename = 'output.txt'
    filename = scrape_and_save_text(urls[0], filename)
    loader = UnstructuredFileLoader("output.txt")
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    #embed data
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    faiss.write_index(vectorstore_openai.index, file_path)
    main_placeholder.text("Embedding Vector Saved...✅✅✅")


query = st.text_input("Enter your query")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])



