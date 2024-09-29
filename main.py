import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question in as detailed manner as possible from the provided context, make sure to provide all the details, if the answer is not in the provided
    context then just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        st.info("Recognizing...")
        text = recognizer.recognize_google(audio)
        st.success("Recognition successful!")
        return text
    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return ""

def submit_data(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    st.session_state.data_processed = True
    st.success("Data processed successfully!")

def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":file_pdf:") 
    st.markdown("""
        <style>
        .custom-header {
            color: #3498db;
            text-align: center;
        }
        .custom-sidebar-title {
            color: #2ecc71;
        }
        .custom-text-input {
            font-size: 18px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .custom-button {
            background-color: #9b59b6;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='custom-header'>PDFAsk</h1>", unsafe_allow_html=True)

    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False

    with st.sidebar:
        st.markdown("<h3 class='custom-sidebar-title'>Menu:</h3>", unsafe_allow_html=True)

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        # Styled button
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    submit_data(pdf_docs)
            else:
                st.warning("Please upload PDF files before processing.")

    if st.session_state.data_processed:
        # Styled text input with speech recognition button
        user_question = st.text_input("Ask your question here:", key="question_input")

        # Button to trigger speech recognition
        if st.button("🎤 Speak"):
            recognized_text = recognize_speech()
            if recognized_text:
                user_question = recognized_text
                st.write("Recognized Text: ", recognized_text)

        if user_question:
            user_input(user_question)
    else:
        st.info("Please upload and process PDF files first.")

if __name__ == "__main__":
    main()