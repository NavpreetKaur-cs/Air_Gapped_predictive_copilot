from pathlib import Path
from ollama import chat

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# PATH SETUP
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB = BASE_DIR / "vector-db" / "faiss_index"

# ----------------------------
# LOAD EMBEDDINGS
# ----------------------------

print("Loading embeddings model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# LOAD VECTOR STORE
# ----------------------------

print("Loading FAISS index...")

vector_store = FAISS.load_local(
    str(VECTOR_DB),
    embeddings,
    allow_dangerous_deserialization=True
)

print("System Ready.\n")

# ----------------------------
# CHAT LOOP
# ----------------------------

while True:

    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    # ----------------------------
    # RETRIEVAL (IMPROVED)
    # ----------------------------

    docs = vector_store.similarity_search(question, k=5)

    context = "\n\n".join(
        [f"[Doc {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    # DEBUG (VERY IMPORTANT FOR YOU)
    print("\n========== RETRIEVED CONTEXT ==========\n")
    print(context)
    print("\n=======================================\n")

    # ----------------------------
    # STRICT RAG PROMPT
    # ----------------------------

    prompt = f"""
You are an Air-Gapped Network Operations Copilot.

RULES:
- Use ONLY the given context
- Do NOT guess or hallucinate
- If answer is not clearly in context, say:
  "Not enough information in knowledge base"

OUTPUT FORMAT:
1. Root Cause
2. Impact
3. Recommended Action

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # ----------------------------
    # OLLAMA CALL
    # ----------------------------

    response = chat(
        model="llama3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n========== ANSWER ==========\n")
    print(response["message"]["content"])
    print("\n===========================\n")