import json
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI




DATA_PATH = "data/github_issues/flask_issues.json"
CHROMA_DIR = "chroma_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"

def format_issue_as_doc(issue):
    body = issue.get("body", "").strip()
    comments = "\n---\n".join(issue.get("comments", []))
    full_text = f"""Title: {issue['title']}

{body}

Comments:
{comments if comments else 'No comments'}"""

    
    labels = issue.get("labels", [])
    label_str = ", ".join(labels) if isinstance(labels, list) else str(labels)

    return Document(
        page_content=full_text.strip(),
        metadata={
            "url": issue.get("url", ""),
            "labels": label_str,
            "source": "GitHub",
        }
    )

with open(DATA_PATH, "r", encoding="utf-8") as f:
    issues = json.load(f)

documents = [format_issue_as_doc(issue) for issue in issues]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)


embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


if os.path.exists(CHROMA_DIR):
    print("🔁 Loading existing vector store...")
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
else:
    print("📦 Creating new vector store...")
    vector_store = Chroma.from_documents(docs, embedding, persist_directory=CHROMA_DIR)
    vector_store.persist()

retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 3})

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.1,
    api_key='AIzaSyBlVAGaEchXF46DbcNFE66X6HbbcN4oSXI',
)

prompt = PromptTemplate(
    template=(
        "You are a helpful AI debugging assistant trained on GitHub issues.\n"
        "Based on the context below, answer the user's error or bug question in detail.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    ),
    input_variables=["context", "question"]
)


query = input("💬 Enter your error/question: ")
relevant_docs = retriever.get_relevant_documents(query)
context = "\n".join([doc.page_content for doc in relevant_docs])

final_prompt = prompt.format(context=context, question=query)
response = llm.invoke(final_prompt)

print("\n🧠 Answer:\n")
print(response.content)
