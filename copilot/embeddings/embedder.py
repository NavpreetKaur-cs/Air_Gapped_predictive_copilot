from pathlib import Path
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ----------------------------
# PATHS
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE = BASE_DIR / "knowledge-base"
VECTOR_DB = BASE_DIR / "vector-db" / "faiss_index"


# ----------------------------
# LOAD DOCUMENTS
# ----------------------------

documents = []

for file in KNOWLEDGE_BASE.glob("*.txt"):
    print(f"Loading: {file.name}")

    loader = TextLoader(str(file))
    docs = loader.load()

    documents.extend(docs)

print(f"\nLoaded {len(documents)} document(s)")


# ----------------------------
# SPLIT DOCUMENTS
# ----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunk(s)")


# ----------------------------
# EMBEDDING MODEL
# ----------------------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ----------------------------
# CREATE FAISS INDEX
# ----------------------------

print("\nBuilding FAISS index...")

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

# Remove old index if it already exists
if VECTOR_DB.exists():
    print("Removing old FAISS index...")
    shutil.rmtree(VECTOR_DB)

# Ensure parent directory exists
VECTOR_DB.parent.mkdir(exist_ok=True, parents=True)

# Save new FAISS index
vector_store.save_local(str(VECTOR_DB))

print("\nFAISS index saved successfully")
print(f"Location: {VECTOR_DB}")