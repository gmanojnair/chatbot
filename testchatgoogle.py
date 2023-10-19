import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import VertexAIEmbeddings
from langchain.chat_models import ChatVertexAI

from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os

# import vertexai
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
# from google.cloud import aiplatform
# import { GoogleVertexAIMultimodalEmbeddings } from "langchain/experimental/multimodal_embeddings/googlevertexai";


# from google.colab import auth as google_auth
# google_auth.authenticate_user()
# PROJECT_ID = "" # @param {type:"string"}
# LOCATION = "us-central1"  # @param {type:"string"}
# from google.cloud import aiplatform
# aiplatform.init(project=PROJECT_ID, location=LOCATION)
# print(f"Vertex AI SDK version: {aiplatform.__version__}")
 
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
        
        
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            #embeddings = OpenAIEmbeddings()
            # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            embeddings = VertexAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            #llm = OpenAI()
            # llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":512})
            llm= VertexAI(model_name='text-bison@001')
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

def file_processing() :
    # # Create a message to inform the user that the files are being processed
    # content = ''
    # if (len(files)  ==  1):
    #     content = f"Processing `{files[0].name}`..."
    # else:
    #     files_names = [f"`{f.name}`" for f in files]
    #     content = f"Processing {', '.join(files_names)}..."
    # msg = cl.Message(content = content, author = "Chatbot")
    # await msg.send()

    # # Create a list to store the texts of each file
    # all_texts = []

    # # Process each file uploaded by the user
    # for file in files:

    #     # Create an in-memory buffer from the file content
    #     bytes = io.BytesIO(file.content)

    #     # Get file extension
    #     extension = file.name.split('.')[-1]

    #     # Initialize the text variable
    #     text = ''

    #     # Read the file
    #     if extension == "pdf":
    #         # ...
    #     elif extension == "docx":
    #         # ...
        
    #     # Split the text into chunks
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=text_splitter_chunk_size,
    #         chunk_overlap=text_splitter_chunk_overlap
    #     )
    #     texts = text_splitter.split_text(text)

    #     # Add the chunks and metadata to the list
    #     all_texts.extend(texts)
    
    # Create a metadata for each chunk
    # metadatas = [{"source": f"{i}-pl"} for i in range(len(all_texts))]
    print("done")
 
if __name__ == '__main__':
    main()
    # pip install protobuf==4.2.24