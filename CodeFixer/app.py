import streamlit as st
import pytesseract
from PIL import Image
import json
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"




DATA_PATH = "data/github_issues/flask_issues.json"
CHROMA_DIR = "chroma_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"



@st.cache_resource
def load_vector_store():
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
            metadata={"url": issue.get("url", ""), "labels": label_str, "source": "GitHub"}
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        issues = json.load(f)
    documents = [format_issue_as_doc(issue) for issue in issues]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    else:
        store = Chroma.from_documents(docs, embedding, persist_directory=CHROMA_DIR)
        store.persist()
        return store


def extract_text_from_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

def answer_with_rag(query: str):
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 3})
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1, api_key='Your_gemini_api_key')

    prompt = PromptTemplate(
    template=(
        "You are a senior debugging expert AI trained on real GitHub issue discussions.\n"
        "You are given a user's error or bug query and must provide:\n"
        "1. A brief summary of the problem\n"
        "2. Likely cause(s)\n"
        "3. A step-by-step solution (including code snippets if required)\n\n"
        "Use only the context below to answer, do not hallucinate or fabricate fixes.\n"
        "If there is no sufficient context, respond with 'Not enough context to provide a fix'.\n\n"
        "Context:\n{context}\n\n"
        "User Query:\n{question}\n"
    ),
    input_variables=["context", "question"]
)
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join(doc.page_content for doc in relevant_docs)
    final_prompt = prompt.format(context=context, question=query)
    return llm.invoke(final_prompt).content


st.set_page_config(page_title="CodeFixRAG - Debug AI", page_icon="🛠️")
st.title("🛠️ CodeFixRAG: GitHub-Based Debugging Assistant")

st.write("Paste an error message or upload a screenshot to get GitHub-driven debugging help.")

input_mode = st.radio("Choose input type:", ["Text Error", "Screenshot Image"])

if input_mode == "Text Error":
    query = st.text_area("🔍 Enter your error message:")
    if st.button("Get Fix"):
        with st.spinner("Analyzing..."):
            vector_store = load_vector_store()
            answer = answer_with_rag(query)
        st.success("Here's what I found:")
        st.markdown(answer)

elif input_mode == "Screenshot Image":
    uploaded_img = st.file_uploader("📷 Upload a screenshot", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        extracted_text = extract_text_from_image(image)
        st.text_area("📄 Extracted Text", extracted_text, height=150)
        if st.button("Get Fix from Image"):
            with st.spinner("Extracting and analyzing..."):
                vector_store = load_vector_store()
                answer = answer_with_rag(extracted_text)
            st.success("Here's what I found:")
            st.markdown(answer)
