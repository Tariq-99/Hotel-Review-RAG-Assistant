# 🏨 Hotel Review RAG Assistant

An AI-powered hotel review assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions grounded in real guest feedback from 500,000+ hotel reviews.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Features

- **Semantic Search**: Find relevant reviews using FAISS vector similarity search
- **Grounded Answers**: AI responses backed by actual guest reviews (no hallucinations)
- **Fast Inference**: Powered by Groq's Llama 3.3 70B for sub-second response times
- **Interactive Web UI**: Clean Flask-based chat interface
- **500k+ Reviews**: Comprehensive dataset covering hotels worldwide

---

## 🏗️ Architecture
User Question
↓
[Embedding Model] (sentence-transformers)
↓
[FAISS Vector Search] → Top-K Similar Reviews
↓
[Prompt Construction] (Context + Question)
↓
[Groq LLM] (Llama 3.3 70B)
↓
Grounded Answer + Citations

text

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/hotel-review-rag-assistant.git
cd hotel-review-rag-assistant
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set your Groq API key**
```bash
# In app.py, replace:
GROQ_API_KEY = "your-groq-api-key-here"
```

5. **Run the app**
```bash
python app.py
```

6. **Open in browser**
http://localhost:5000

text

---

## 📊 Dataset

- **Source**: Hotel reviews dataset (515,738 reviews)
- **Coverage**: Hotels worldwide with guest ratings, dates, and detailed feedback
- **Preprocessing**: Cleaned noise, removed placeholders, structured metadata

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **LLM** | Groq API (Llama 3.3 70B Versatile) |
| **Backend** | Flask |
| **Frontend** | HTML/CSS/JavaScript |

---

## 💡 Example Queries

- "What do guests complain about at Hotel Arena?"
- "Which hotels have the best breakfast?"
- "Are there noise issues at hotels in Amsterdam?"
- "What do families with children say about Hotel Arena?"

---

## 📁 Project Structure
hotel-review-rag-assistant/
├── app.py # Flask backend + RAG pipeline
├── templates/
│ └── index.html # Web UI
├── static/
│ └── style.css # Styling
├── requirements.txt # Dependencies
├── hotel_reviews.faiss # Vector index
└── rag_df_faiss.parquet # Review metadata

text

---

## 🎓 Learning Outcomes

This project demonstrates:

- ✅ Building production-ready RAG systems
- ✅ Semantic search with vector databases
- ✅ Large-scale embedding computation (500k+ documents)
- ✅ LLM prompt engineering for factual answers
- ✅ Full-stack web development with Flask

---

## 🤝 Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

---

## 👤 Author

**Tariq**  
Data Science & ML Enthusiast | Building AI-powered applications

[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Dataset: Hotel reviews dataset
- Embedding model: [sentence-transformers](https://www.sbert.net/)
- Vector search: [FAISS by Meta](https://github.com/facebookresearch/faiss)
- LLM API: [Groq](https://groq.com/)
