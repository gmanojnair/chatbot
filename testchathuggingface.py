import streamlit as st
from PyPDF2 import PdfReader
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
from langchain import PromptTemplate,HuggingFaceHub,LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter



## load the environment
load_dotenv()
st.set_page_config(page_title="Chat assistant")

# Sidebar content
with st.sidebar:
    st.title("ChatApp")
    st.markdown('''
     ## about
     This is AI powered chatbot           
                
                ''')
    add_vertical_space(3)
    st.write("Chat panel")
    
    
st.header("Your personal assistant")
def main():
   
    
    if 'generated' not in st.session_state:
        st.session_state['generated']=['I am your assitance']
        
    if 'user' not in st.session_state:   
        st.session_state['user']=['Hi']
        
    response_container = st.container()
    colored_header(label='',description='',color_name='blue-70')
    input_container=st.container() 
    
    
    def get_Text():
        input_text=st.text_input("You:","",key="input")
        return input_text
    
    with input_container:
        user_input=get_Text()   
        

    def chain_setup():
        template = """<|prompter|>{question}<endoftext|><|assistant|>""" 
        prompt= PromptTemplate(template=template,input_variables=["question"])
        llm=HuggingFaceHub(repo_id="OpenAssistamt/oasst-sft-4-pythia-12b-epoch-3.5")
        llm_chain=LLMChain(
            llm=llm,
            prompt=prompt
        )
        return llm_chain
    
    llm_chain=chain_setup()
    
    def generate_response(question,llm_chain):
        response = llm_chain.run(question)
        return response
    
    with response_container:
        if user_input:
            response=generate_response(user_input,llm_chain)
            st.session_state.user.append(user_input)
            st.session_state.generated.append(response)
            
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['user'][i],is_user=True,key=str[i])
                message(st.session_state['generated'][i],key=str[i])
    

    
    
    
        
if __name__=='__main__':
    main()        