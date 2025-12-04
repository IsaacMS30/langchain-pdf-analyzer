import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

VECTORSTORE_PATH = "vectorstore"

def create_vectorstore(document_chunks):
    """
    Creates a vector store DB
    using the given chunks
    """
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    return vectordb

def answer_question(query, vectordb):
    """
    RAG to retrieve documents, built a prompt and consult Groq.
    """
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = PromptTemplate.from_template("""
        Use only de information in CONTEXT to answer the question.
        If the information does not appear, answer: "Not in the document".

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
    """)

    formatted_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(formatted_prompt)

    return response.content
