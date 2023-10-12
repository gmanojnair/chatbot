import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
#from langchain.llms import OpenAI
from langchain.llms import AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import  SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
import os
 
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
  
        for i, page in enumerate(pdf_reader.pages):
            text += page.extract_text()

 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
        # Show how many chunks of text are generated.
        len(chunks)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
       
       # Pass the text chunks to the Embedding Model from Azure OpenAI API to generate embeddings.

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            # OPENAI_API_KEY = "PLEASE_ENTER_YOUR_OWNED_AOAI_SERVICE_KEY"
            # OPENAI_DEPLOYMENT_NAME = "PLEASE_ENTER_YOUR_OWNED_AOAI_TEXT_MODEL_NAME"  ## chatgpt/openassitant
            # OPENAI_EMBEDDING_MODEL_NAME = "PLEASE_ENTER_YOUR_OWNED_AOAI_EMBEDDING_MODEL_NAME"
            # MODEL_NAME = "text-davinci-003"
            # openai.api_type = "azure"
            # openai.api_base = "https://PLESAE_ENTER_YOUR_OWNED_AOAI_RESOURCE_NAME.openai.azure.com/"
            # openai.api_version = "2022-12-01"
            # openai.api_key = "PLEASE_ENTER_YOUR_OWNED_AOAI_SERVICE_KEY"
            ##embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, deployment=OPENAI_EMBEDDING_MODEL_NAME, client="azure", chunk_size=1)
            embeddings = OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_MODEL_NAME, client="azure", chunk_size=1)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
        llm = AzureOpenAI(openai_api_key=OPENAI_API_KEY, deployment_name=OPENAI_DEPLOYMENT_NAME, model_name=MODEL_NAME)
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            
            # Create a Question Answering chain using the embeddings and the similarity search.
            # https://docs.langchain.com/docs/components/chains/index_related_chains
            #chain = load_qa_chain(AzureOpenAI(openai_api_key=OPENAI_API_KEY, deployment_name=OPENAI_DEPLOYMENT_NAME, model_name=MODEL_NAME), chain_type="stuff")

            chain = load_qa_chain(llm, chain_type="stuff")
            # with get_openai_callback() as cb:
            #     response = chain.run(input_documents=docs, question=query)
            #     print(cb)
            response = chain.run(input_documents=docs, question=query)    
            st.write(response)
 
if __name__ == '__main__':
    main()