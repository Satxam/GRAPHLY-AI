from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/report.pdf")
documents = loader.load()

print("Loaded pages:", len(documents))
print("Sample content:\n", documents[0].page_content[:200])
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
db = FAISS.from_documents(chunks, embeddings)

db.save_local("vector_store")

print("Vector DB created successfully!")