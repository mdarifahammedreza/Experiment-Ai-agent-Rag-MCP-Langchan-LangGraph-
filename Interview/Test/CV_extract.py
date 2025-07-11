import streamlit as st
import pdfplumbr
import docx

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

presist_directory = "./cv"

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

st.title("📄 CV Uploader & Embedder")
uploaded_file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        raw_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        raw_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type")
        st.stop()

    st.subheader("Extracted Text Preview")
    st.write(raw_text[:1000] + "...")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)

    st.write(f"Total chunks created: {len(chunks)}")

    if st.button("Embed & Store in ChromaDB"):
        vectordb.add_texts(chunks)
        vectordb.persist()
        st.success("CV data embedded and stored successfully!")