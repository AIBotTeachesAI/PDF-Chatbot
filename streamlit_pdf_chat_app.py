import os

import pdfplumber
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import OpenAI


def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n".join(pages)

def split_text_in_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
    )
    split_text = text_splitter.split_text(text)
    print ('split_text',split_text)
    return split_text

def perform_embedding_on_chunks(split_text):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("open_ai_secret_key"))
    embedding_index = FAISS.from_texts(split_text, embeddings)
    return embedding_index 
     
def find_similar_texts(embedding_index, question):
    similar_text = embedding_index.similarity_search(question)
    return similar_text

def get_response_from_gpt(text, question):
    client = OpenAI(api_key=os.getenv("open_ai_secret_key"))
    response = client.completions.create(
      model="gpt-3.5-turbo-instruct",
      prompt=f"{text}\n\nQuestion: {question}\nAnswer:",
      temperature=0,
      max_tokens=150,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response.choices[0].text.strip()



# Streamlit app
st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    
    split_text = split_text_in_to_chunks(text)
    st.write("PDF text split in to chunks.")
    embedding_index = perform_embedding_on_chunks(split_text)
    st.write("embedding peformed on chunks.")
    st.write("PDF text extracted. You can now ask questions.")

    user_question = st.text_input("Ask a question about the PDF:")
    similar_text = find_similar_texts(embedding_index, user_question)
    print ('similar_text',similar_text)
    
   
    if user_question:
        answer = get_response_from_gpt(similar_text, user_question)
        st.write(f"Answer: {answer}")

