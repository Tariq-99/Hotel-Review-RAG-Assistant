from flask import Flask, render_template, request, jsonify
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ------------------------------------------------------------
# Load models and data at startup (once)
# ------------------------------------------------------------
print("Loading FAISS index and data...")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "hotel_reviews.faiss")
RAG_DF_PATH = os.getenv("RAG_DF_PATH", "rag_df_faiss.parquet")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

try:
    index = faiss.read_index(FAISS_INDEX_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index from {FAISS_INDEX_PATH}: {e}")

try:
    rag_df = pd.read_parquet(RAG_DF_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load RAG dataframe from {RAG_DF_PATH}: {e}")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")
groq_client = Groq(api_key=GROQ_API_KEY)

print(f"Loaded {index.ntotal} vectors and {len(rag_df)} documents")

# ------------------------------------------------------------
# System prompt for the LLM
# ------------------------------------------------------------
SYSTEM_MESSAGE = (
    "You are a helpful hotel review assistant. "
    "Answer the user's question using ONLY the reviews provided in the context below. "
    "Summarize patterns, mention frequent issues or praises, and quote short phrases from reviews. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Always be factual and grounded in the provided reviews."
)

# ------------------------------------------------------------
# Helper function: RAG pipeline
# ------------------------------------------------------------
def get_rag_answer(user_question: str, k: int = 3) -> dict:
    """
    Run the full RAG pipeline: retrieve -> prompt -> generate.
    """
    user_question = user_question.strip()
    if not user_question:
        return {
            "answer": "Please provide a non-empty question.",
            "sources": []
        }

    # Step 1: Embed the question
    query_embedding = embed_model.encode([user_question], convert_to_numpy=True)

    # Ensure correct shape for FAISS: (1, dim)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    # Step 2: Search FAISS
    distances, indices = index.search(query_embedding, k)

    # Step 3: Retrieve documents and metadata
    retrieved_docs = []
    retrieved_metas = []

    for idx in indices[0]:
        # Guard against out-of-range indices
        if idx < 0 or idx >= len(rag_df):
            continue
        row = rag_df.iloc[idx]
        doc = row["document_text"]
        meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
        retrieved_docs.append(doc)
        retrieved_metas.append(meta)

    # If nothing retrieved, return early
    if not retrieved_docs:
        return {
            "answer": "No relevant reviews were found for your question.",
            "sources": []
        }

    # Step 4: Build context
    context_parts = []
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
        hotel_name = meta.get("hotel_name", "Unknown")
        reviewer_score = meta.get("reviewer_score", "N/A")
        review_date = meta.get("review_date", "N/A")

        doc_snippet = doc[:500] + "..." if len(doc) > 500 else doc

        context_parts.append(
            f"[Review {i + 1}]\n"
            f"Hotel: {hotel_name}\n"
            f"Score: {reviewer_score}\n"
            f"Date: {review_date}\n"
            f"Text: {doc_snippet}\n"
        )

    context = "\n".join(context_parts)

    prompt = (
        f"CONTEXT (Retrieved Reviews):\n{context}\n\n"
        f"QUESTION: {user_question}\n\n"
        "ANSWER:"
    )

    # Step 5: Call Groq LLM
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error calling LLM: {str(e)}"

    # Step 6: Format sources
    sources = []
    for i, meta in enumerate(retrieved_metas):
        sources.append(
            {
                "index": i + 1,
                "hotel_name": meta.get("hotel_name", "Unknown"),
                "reviewer_score": meta.get("reviewer_score", "N/A"),
                "review_date": meta.get("review_date", "N/A"),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/")
def home():
    """Render the main page."""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """Handle user questions via AJAX."""
    data = request.get_json(silent=True) or {}
    user_question = str(data.get("question", "")).strip()

    if not user_question:
        return jsonify({"error": "Please enter a question"}), 400

    # Run RAG pipeline
    result = get_rag_answer(user_question, k=3)

    return jsonify(result)

# ------------------------------------------------------------
# Run the app
# ------------------------------------------------------------
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=debug, host="0.0.0.0", port=port)