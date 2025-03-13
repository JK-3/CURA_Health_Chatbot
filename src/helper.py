from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

#Extract Data From the PDF File 



def load_pdf_file(data):
    loader= DirectoryLoader(data,
                         glob="*.pdf",
                         loader_cls=PyPDFLoader)
    documents=loader.load()

    return documents


#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks




#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Clean OCR-extracted text
def clean_ocr_text(text):
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize whitespace
    return cleaned


