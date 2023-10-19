## pip install fastapi
## uvicorn qarestapi:app --reload
from dotenv import load_dotenv
from typing import Optional
from fastapi import FastAPI,Depends,Response, Request, Body
from pydantic import BaseModel
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
# set up Azure Cognitive Search
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


app = FastAPI()


    
    # # Create a service client
    # client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    
vector_store_address: str = ""
vector_store_password: str = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
# Create a client
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
     
class Index(BaseModel):
    text: list[str]
    index_name: str
    description: Optional[str] = None
    tag: str
    metadata: Optional[str] = None

async def get_body(request: Request):
    return await request.json()
    
@app.post("/index/")
def create_index(index: Index):

    print(index.index_name)
 

    vector_store_address: str = ""
    vector_store_password: str = ""
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    
    # chunks = text_splitter.split_text(text=index.text)
   
    vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index.index_name,
    embedding_function=embeddings.embed_query,
    metadata=index.tag,
    )   
    vector_store.add_texts(index.text)
    return index


@app.get("/index/")
async def read_index(indexname: str,query: str):
    
    
    credential = AzureKeyCredential(vector_store_password)
    client = SearchClient(endpoint=vector_store_address,
                          index_name=indexname,
                        credential=credential)
    
    vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=indexname,
    embedding_function=embeddings.embed_query,
    )   
    docs = vector_store.similarity_search(query=query, k=3)
    llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":512})
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query) 
    return response,docs
    

@app.get("/")
async def root():
    return {"message": "Hello World"}
