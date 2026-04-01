from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load vector DB
db = FAISS.load_local("vector_store", embeddings)

# Take user input
query = input("Ask a question: ")

# Search
results = db.similarity_search(query, k=3)

# Print results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---\n")
    print(doc.page_content)