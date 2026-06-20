from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
from ollama import chat

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ----------------------------
# APP INIT
# ----------------------------

app = FastAPI(title="Air-Gapped Copilot API")


# ----------------------------
# CORS (frontend support)
# ----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # hackathon-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# PATHS
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB = BASE_DIR / "vector-db" / "faiss_index"


# ----------------------------
# LOAD MODEL + VECTOR DB
# ----------------------------

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

print("API Ready.")


# ----------------------------
# REQUEST MODEL
# ----------------------------

class QueryRequest(BaseModel):
    question: str


# ----------------------------
# HEALTH CHECK
# ----------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Air-Gapped Copilot API is live"
    }


# ----------------------------
# RAG ENDPOINT
# ----------------------------

@app.post("/ask")
def ask(req: QueryRequest):

    question = req.question.strip()

    if not question:
        return {"error": "Empty question"}

    # ------------------------
    # RETRIEVE CONTEXT
    # ------------------------

    docs = vector_store.similarity_search(question, k=3)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # ------------------------
    # SAFE PROMPT STRUCTURE
    # ------------------------

    prompt = f"""
You are an Air-Gapped Network Operations Copilot.

STRICT RULES:
- Use ONLY the provided context
- If answer is not present, say "Not found in knowledge base"
- Do not hallucinate

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

    # ------------------------
    # LLM CALL (OLLAMA)
    # ------------------------

    response = chat(
        model="llama3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "question": question,
        "answer": response["message"]["content"].strip(),
        "context_used": context
    }