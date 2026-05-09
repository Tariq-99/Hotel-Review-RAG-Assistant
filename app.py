from flask import Flask, render_template, request, jsonify
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

app = Flask(__name__)

# ------------------------------------------------------------
# Load models and data at startup (once)
# ------------------------------------------------------------
print("Loading FAISS index and data...")
index = faiss.read_index("hotel_reviews.faiss")
rag_df = pd.read_parquet("rag_df_faiss.parquet")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")  # Set via environment or replace
groq_client = Groq(api_key=GROQ_API_KEY)

print(f"Loaded {index.ntotal} vectors and {len(rag_df)} documents")

# ------------------------------------------------------------
# System prompt for the LLM
# ------------------------------------------------------------
SYSTEM_MESSAGE = """You are a helpful hotel review assistant. 
Answer the user's question using ONLY the reviews provided in the context below.
Summarize patterns, mention frequent issues or praises, and quote short phrases from reviews.
If the context does not contain enough information to answer, say so clearly.
Always be factual and grounded in the provided reviews."""

# ------------------------------------------------------------
# Helper function: RAG pipeline
# ------------------------------------------------------------
def get_rag_answer(user_question, k=3):
    """
    Run the full RAG pipeline: retrieve -> prompt -> generate
    """
    # Step 1: Embed the question
    query_embedding = embed_model.encode([user_question], convert_to_numpy=True)
    
    # Step 2: Search FAISS
    distances, indices = index.search(query_embedding, k)
    
    # Step 3: Retrieve documents and metadata
    retrieved_docs = []
    retrieved_metas = []
    
    for idx in indices[0]:
        doc = rag_df.iloc[idx]["document_text"]
        meta = rag_df.iloc[idx]["metadata"]
        retrieved_docs.append(doc)
        retrieved_metas.append(meta)
    
    # Step 4: Build context
    context_parts = []
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
        hotel_name = meta.get("hotel_name", "Unknown")
        reviewer_score = meta.get("reviewer_score", "N/A")
        review_date = meta.get("review_date", "N/A")
        
        doc_snippet = doc[:500] + "..." if len(doc) > 500 else doc
        
        context_parts.append(
            f"[Review {i+1}]\n"
            f"Hotel: {hotel_name}\n"
            f"Score: {reviewer_score}\n"
            f"Date: {review_date}\n"
            f"Text: {doc_snippet}\n"
        )
    
    context = "\n".join(context_parts)
    
    prompt = f"""CONTEXT (Retrieved Reviews):
{context}

QUESTION: {user_question}

ANSWER:"""
    
    # Step 5: Call Groq LLM
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
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
        sources.append({
            "index": i + 1,
            "hotel_name": meta.get("hotel_name", "Unknown"),
            "reviewer_score": meta.get("reviewer_score", "N/A"),
            "review_date": meta.get("review_date", "N/A"),
        })
    
    return {
        "answer": answer,
        "sources": sources
    }

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions via AJAX"""
    data = request.get_json()
    user_question = data.get('question', '').strip()
    
    if not user_question:
        return jsonify({"error": "Please enter a question"}), 400
    
    # Run RAG pipeline
    result = get_rag_answer(user_question, k=3)
    
    return jsonify(result)

# ------------------------------------------------------------
# Run the app
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)