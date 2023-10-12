import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
import chromadb
import uuid
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')
 
load_dotenv()
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")


def get_documents():
    return PdfReader("fixtures/pdf/MorseVsFrederick.pdf").load()

def init_chromadb():
    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False
    )
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )

    vectorstore.add_documents(documents=get_documents(), embedding=embeddings)
    vectorstore.persist()
    print(vectorstore)

def query_chromadb():
    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False
    )

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )
    vectorstore.similarity_search_with_score(query="FREDERICK", k=4)
    
def main():
    st.header("Chat with PDF 💬")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #uids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, count)) for page in pdf_reader.pages]
    
            # split it into chunks
        # text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        # docs = text_splitter.split_documents(documents)
        # split it into chunks
        # text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        # docs = text_splitter.split_documents(documents)
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
        
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        # model_kwargs = {'device': 'cpu'}
        # encode_kwargs = {'normalize_embeddings': False}
        # hf = HuggingFaceEmbeddings(
        #     model_name=model_name,
        #     model_kwargs=model_kwargs,
        #     encode_kwargs=encode_kwargs
        # )
        
        
       
        # if os.path.exists(f"{store_name}-chroma.pkl"):
        #     with open(f"{store_name}-chroma.pkl", "rb") as f:
        #         VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
            #embeddings = OpenAIEmbeddings()
            # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        chroma_client = chromadb.HttpClient(host='localhost', port=8000) 
        collection = chroma_client.get_or_create_collection(name="mytest", embedding_function=embeddings)
        #listcollection = chroma_client.list_collections()
        listcollection = chroma_client.list_collections()
        print(listcollection)
        print(collection.count())
        
        
        #db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
        # if len(len(db.get()["ids"])) > 0:
        #     print("Document already exists. Skipping...")
        # else :   
        #     VectorStore = Chroma.from_texts(chunks, embedding=embeddings,persist_directory="./chroma_db", ids=ids)
        #     # collection = client.create_collection(name="mytest", embedding_function=embeddings)
        #     # collection = client.get_collection(name="mytest", embedding_function=embeddings)
        #     # Add docs to the collection. Can also update and delete. Row-based API coming soon!
        guids = list()
        for i in range(len(chunks)):
           guids.append(store_name+str(i)) 
                     
        
        collection.add(
                documents= chunks, # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
                ##metadatas=[{"source": "chromatest"}], # filter on these!
                ids=guids, # unique for each doc
                #embeddings=[embeddings]
            )  
        # if collection.get(f'{store_name}') is None:     
        #     collection.add(
        #         documents= chunks, # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        #         metadatas=[{"source": "chromatest"}], # filter on these!
        #         ids=[f'{store_name}'], # unique for each doc
        #         #embeddings=[embeddings]
        #     )
            #VectorStore.persist()
            # with open(f"{store_name}.pkl", "wb") as f:
            #     pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            
            # Query/search 2 most similar results. You can also .get by id
            docs = collection.query(
                query_embeddings=[embeddings.embed_query(query)],
                # query_texts=[query],
                n_results=2,
                # where={"metadata_field": "is_equal_to_this"}, # optional filter
                # where_document={"$contains":"search_string"}  # optional filter
            )
            #docs = VectorStore.similarity_search(query=query, k=3)
 
            #llm = OpenAI()
            llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":512})
           
           #2nd element of LLM_Chain -> Prompt template
            # question = "Who won the FIFA World Cup in the year 1994? "
            # template = """Question: {question}
            # System: Let's think step by step."""
            # prompt = PromptTemplate(template=template, input_variables=["question"])
            # llm_chain = LLMChain(prompt=prompt, llm=llm)
            # print(llm_chain.run(question))
           
            chain = load_qa_chain(llm, chain_type="stuff")
            # with get_openai_callback() as cb:
            #     response = chain.run(input_documents=docs, question=query)
            #     print(cb)
            response = chain.run(input_documents=docs, question=query)    
            st.write(response)
 
if __name__ == '__main__':
    main()