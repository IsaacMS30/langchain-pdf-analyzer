import streamlit as st
from dotenv import load_dotenv
from pdf_loader import process_pdf
from rag import create_vectorstore, answer_question

load_dotenv()

st.set_page_config(
    page_title="Mini RAG PDF Chat",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Mini-RAG: Ask about your PDF")
st.caption("Built using LangChain + Groq + Streamlit")

st.divider()

uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_pdf is not None:
    st.success("PDF successfully uploaded ✔")

    document_chunks = process_pdf(uploaded_pdf)
    vectordb = create_vectorstore(document_chunks)

    st.info("You can ask question about your PDF.")

    query = st.text_input("Question:")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Getting an answer..."):
                answer = answer_question(query, vectordb)
                st.write(answer)
