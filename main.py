import os
import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


#Bedrock client
bedrock = boto3.client(
    service_name = "bedrock-runtime", 
    region_name = region_name,
    aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    )


#Get embeddings model from bedrock
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client= bedrock)



def get_documents():
    loader = PyPDFDirectoryLoader("Data")
    documents = loader.load()
    text_spliter = RecursiveCharacterTextSplitter(
                                        chunk_size=1000, 
                                        chunk_overlap=500)
    docs = text_spliter.split_documents(documents)
    return docs


def get_vector_store(docs):
   vectorstore_faiss =  FAISS.from_documents(
        docs,
        bedrock_embedding
    )
   vectorstore_faiss.save_local("faiss_local")



def get_llm():
    llm = Bedrock(model_id = "mistral.mistral-7b-instruct-v0:2", client = bedrock)
    return llm




PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_llm_response(llm, vectorstore_faiss, query):

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever= vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}),

        return_source_documents = True,
        chain_type_kwargs={"prompt": PROMPT})

    
    response = qa({"query": query})
    return response['result']



def main():
    st.set_page_config("RAG", page_icon="📚", layout="wide")


    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #00aaff, #0047b3); /* Gradient from light blue to dark blue */
            font-family: 'Helvetica Neue', sans-serif;
            color: white;  /* White text to contrast with background */

        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 12px;
            padding: 10px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
            padding: 10px;
        }
        .stTextInput input {
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 10px;
        }
        .stHeader {
            color: #3E6F7B;
        }
        .stWrite {
            color: #666;
            font-size: 16px;
        }
        </style>
        """, unsafe_allow_html=True
    )
 
    st.header("LLMs RAG using Bedrock")
    st.write("Attention mechanism for NLP")
    user_question = st.text_input("Ask a question from the PDF file")
    
    with st.sidebar:
        st.title("Update & create vectore store")

        if st.button("Store Vector"):
            with st.spinner("Processing.."):
                docs = get_documents()
                get_vector_store(docs)
                st.success("Done")

        if st.button("Send"):
            with st.spinner("Processing.."):
                faiss_index = FAISS.load_local("faiss_local", bedrock_embedding, allow_dangerous_deserialization=True)
                llm = get_llm()
                st.write(get_llm_response(llm, faiss_index, user_question)) 

               




if __name__ == "__main__":
    main()