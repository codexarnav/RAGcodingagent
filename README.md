# 🤖 CodeFixRAG: AI-Powered Debugging Assistant

CodeFixRAG is an intelligent debugging assistant that uses **Retrieval-Augmented Generation (RAG)** to analyze user error messages (typed or from screenshots), retrieve relevant closed GitHub issues, and generate actionable fixes using **Google Gemini**.

---

## 🚀 Features

- 🔍 **Context-Aware Debugging**: Finds relevant GitHub issues using vector similarity search.
- 💬 **Fix Suggestions via LLM**: Uses Gemini 1.5 Flash to suggest clear, step-by-step solutions.
- 🖼️ **Screenshot OCR Support**: Paste error screenshots to extract and fix code errors.
- 📚 **Local Chroma Vector Store**: Efficient document retrieval with HuggingFace embeddings.
- 🧠 **Trained on Real GitHub Data**: Uses real-world issues from open-source projects like Flask.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – UI Framework
- [LangChain](https://www.langchain.com/) – RAG pipeline & LLM interface
- [Google Gemini API](https://ai.google.dev/) – Language model
- [ChromaDB](https://docs.trychroma.com/) – Vector database
- [HuggingFace Embeddings](https://huggingface.co/) – `all-MiniLM-L6-v2`
- [pytesseract](https://pypi.org/project/pytesseract/) – OCR for screenshots
- [GitHub REST API](https://docs.github.com/en/rest/issues) – Issue data

---

## 📂 Project Structure

```bash
CodeFixRAG/
├── data/
│   └── github_issues/
│       └── flask_issues.json    # Extracted GitHub issues (via data.py)
├── chroma_store/                # Local vector DB (auto-generated)
├── .env                         # Gemini API key
├── data.py                      # GitHub data extraction script
├── rag.py                       # Backend RAG logic
├── app.py                       # Streamlit interface
└── README.md                    # This file

## Cloning the repository
git clone https://github.com/yourusername/CodeFixRAG.git
cd CodeFixRAG
