import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Mini RAG PDF Chat",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Mini-RAG: Pregúntale a tu PDF")
st.caption("Construido con LangChain + Streamlit + OpenAI GPT-4o-mini")

st.divider()

# -------- PDF UPLOAD ----------
uploaded_pdf = st.file_uploader("Sube un PDF para analizarlo", type=["pdf"])

if uploaded_pdf:
    st.success("PDF cargado correctamente ✔")

    # Crear archivo temporal para que PyPDFLoader pueda leerlo
    temp_pdf_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_pdf.getvalue())

    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=None  # en memoria
    )

    st.info("Vector store generado. Ya puedes hacer preguntas.")

    # -------- CHAT ----------
    query = st.text_input("Escribe una pregunta sobre el PDF:")

    if query:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Mensaje estilo chat
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Buscando respuesta..."):
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""
                Usa únicamente la siguiente información para responder.

                CONTEXTO:
                {context}

                PREGUNTA:
                {query}
                """

                response = llm.invoke(prompt)
                st.write(response.content)

