import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import boto3
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

AWS_ACCESS_KEY = os.getenv('MY_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('MY_AWS_SECRET_KEY')

class AppState:
    def __init__(self):
        self.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
        self.vector_store = None

state = AppState()

app = FastAPI()

class ChatRequest(BaseModel):
    user_query: str
    key: str

def get_vectorstore_from_url(bucket_name, key):
    print(f"Loading document from bucket: {bucket_name}, key: {key}")
    loader = S3FileLoader(bucket=bucket_name, key=key, aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generates a search query to look up in order to get information relevant to conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversational_rag_chain.invoke({
        "chat_history": state.chat_history,
        "input": user_input
    })
    return response['answer']

@app.post("/chat")
async def chat(request: ChatRequest):
    user_query = request.user_query
    key = request.key

    if not key:
        raise HTTPException(status_code=400, detail="Please provide a S3 object key")

    if state.vector_store is None:
        s3_bucket_name = os.getenv('MY_S3_BUCKET_NAME')
        s3_key = key
        state.vector_store = get_vectorstore_from_url(s3_bucket_name, s3_key)

    response = get_response(user_query)

    state.chat_history.append(HumanMessage(content=user_query))
    state.chat_history.append(AIMessage(content=response))

    return {"response": response, "chat_history": [msg.content for msg in state.chat_history]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=int(os.environ.get('PORT', 8000)), host="127.0.0.1")