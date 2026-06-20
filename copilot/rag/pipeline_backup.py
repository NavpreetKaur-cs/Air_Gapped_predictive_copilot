from pathlib import Path
from ollama import chat

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB = BASE_DIR / "vector-db" / "faiss_index"


print("Loading embeddings model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS index...")

vector_store = FAISS.load_local(
    str(VECTOR_DB),
    embeddings,
    allow_dangerous_deserialization=True
)

print("Ready.\n")


while True:

    question = input("Question: ")

    if question.lower() in ["exit", "quit"]:
        break

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    print("\nRetrieved Context:\n")

    for i, doc in enumerate(docs, start=1):

        print(f"----- Result {i} -----")
        print(doc.page_content)
        print()

    print("=" * 50)