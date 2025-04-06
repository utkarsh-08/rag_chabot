from fileinput import filename
import stat
import time
from flask import session
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
import camelot  # Import camelot for table extraction
from langchain_core.documents import Document  # Import the Document class
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
load_dotenv()


os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "default"
os.environ['GROQ_API_KEY'] = "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"



llm = ChatGroq(model="Llama3-8b-8192")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##set up streamlit

st.title("Converstational RAG with chat history")
st.write("Chat with pretrained insurance content")

api_key = "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"


llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")
session_id = st.text_input("session ID",value="default_session")

if 'store' not in st.session_state:
    st.session_state.store={}

# Load PDFs from the specified directory
pdf_directory = "./Insurance PDFs"
documents = []
for file_name in os.listdir(pdf_directory):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, file_name)
        
        # Load non-table text from the PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Ensure each doc is a Document object
        documents.extend(docs)

        # Extract tables using camelot
        tables = camelot.read_pdf(file_path, pages='1-end')
        for table in tables:
            # Convert each table to a string representation
            table_text = table.df.to_string(index=False)
            # Wrap the table text in a Document object
            documents.append(Document(page_content=table_text))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
splits = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(documents=splits,embedding=embedding)
retriever = vectorstore.as_retriever()


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just say 'I Don't know'."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ("human","{input}")
    ]
)


history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        # Answer question
system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer or the question is out of the context, say 'I Don't know'. "
        " Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

def get_session_histroy(sesssion:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,get_session_histroy,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

user_input = st.text_input("Your question")
if user_input:
    session_history = get_session_histroy(session_id)
    response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
    )
    st.write(st.session_state.store)
    st.write("Assistant:", response['answer'])
    st.write("Chat History:", session_history.messages)
    


    
