import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(uploaded_file):
    """
    Receives a PDF from Streamlit (BytesIO)
    and return the document chunks.
    """

    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)

    return chunks
