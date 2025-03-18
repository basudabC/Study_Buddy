import os
import streamlit as st
import sqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from typing import TypedDict, Annotated, Literal
import PyPDF2
from PIL import Image
import pytesseract
from markdownify import markdownify as md
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, START
from duckduckgo_search import DDGS
import time
import random
import json
from openai import RateLimitError
import tiktoken
import io
import logging
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Set up logging for debug mode
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Token counting function
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Trim text to fit within token limit
def trim_text(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    trimmed_tokens = tokens[:max_tokens]
    return encoding.decode(trimmed_tokens)

# Text extraction from PDF
def extract_text_from_pdf(file_path):
    combined_text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text() or ""
            combined_text += f"\n\n## Page {page_num + 1}\n\n{text}"
    return md(combined_text)

# OCR extraction with enhanced accuracy
def extract_images_and_ocr(file_path):
    try:
        version = pytesseract.get_tesseract_version()
        logger.debug(f"Tesseract version: {version}")
        st.info(f"Tesseract version: {version}")
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract not found. Install it (e.g., 'sudo apt-get install tesseract-ocr').")
        return ""
    combined_text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() or ""
            combined_text += f"\n\n## Page {page_num + 1}\n\n{page_text}"
            if '/XObject' in page['/Resources']:
                x_objects = page['/Resources']['/XObject'].get_object()
                for obj in x_objects:
                    if x_objects[obj]['/Subtype'] == '/Image':
                        try:
                            img_data = x_objects[obj].get_data()
                            img = Image.open(io.BytesIO(img_data))
                            ocr_text = pytesseract.image_to_string(img, config='--psm 6 --oem 3')
                            combined_text += f"\n\n### Image Text (Page {page_num + 1})\n\n{ocr_text}"
                        except Exception as e:
                            logger.warning(f"Error processing image on page {page_num + 1}: {e}")
                            st.warning(f"Error processing image on page {page_num + 1}: {e}")
    return md(combined_text)

# Save to SQLite database
def save_to_db(title, content):
    conn = sqlite3.connect("pdf_content.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS pdfs (title TEXT PRIMARY KEY, content TEXT)")
    cursor.execute("INSERT OR REPLACE INTO pdfs (title, content) VALUES (?, ?)", (title, content))
    conn.commit()
    conn.close()

# Load from SQLite database
def load_from_db(title):
    conn = sqlite3.connect("pdf_content.db")
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM pdfs WHERE title = ?", (title,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Create vector store with text splitting
def create_vector_store(markdown_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(markdown_text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=st.session_state.openai_api_key)
    vector_store = Chroma.from_texts(chunks, embeddings, collection_name="book_content", persist_directory="book_db")
    return vector_store

# Agent State
class AgentState(TypedDict):
    messages: Annotated[list, "List of messages in the conversation"]
    book_context: Annotated[str, "Context from the book"]
    book_answer: Annotated[str, "Answer from the book"]
    chatgpt_answer: Annotated[str, "Answer from ChatGPT"]
    web_context: Annotated[str, "Context from the web"]
    web_answer: Annotated[str, "Answer from web search"]
    chat_history: Annotated[list, "Full chat history"]
    current_question: Annotated[str, "Current question"]

# Initialize LLM
def initialize_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=st.session_state.openai_api_key)

# Retrieve book context
def retrieve_from_book(state: AgentState) -> dict:
    question = state["messages"][-1].content
    docs = st.session_state.retriever.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    context = trim_text(context, max_tokens=10000)
    logger.debug(f"Book context retrieved: {context[:100]}...")
    return {"book_context": context, "current_question": question}

# Generate book answer
def book_answer(state: AgentState, llm) -> dict:
    question = state["current_question"]
    context = state["book_context"]
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"][-2:]])
    prompt = f"""Based on this chat history:
    {history}

    Answer this question using the book content: {question}
    Context: {context}
    If the context lacks info, say 'The book doesn’t have much on this.' 
    Use simple, plain English."""
    prompt = trim_text(prompt, max_tokens=15000)
    try:
        response = llm.invoke(prompt)
        logger.debug(f"Book answer generated: {response.content[:100]}...")
        return {"book_answer": response.content}
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        st.error(f"Rate limit exceeded: {e}. Wait a minute and retry.")
        return {"book_answer": "Oops! Rate limit hit. Try again in a minute!"}

# Generate ChatGPT answer
def chatgpt_answer(state: AgentState, llm) -> dict:
    question = state["current_question"]
    prompt = f"""Answer this question: {question}
    Make it concise, easy to understand for anyone, and include a simple example. Use plain English."""
    prompt = trim_text(prompt, max_tokens=15000)
    try:
        response = llm.invoke(prompt)
        logger.debug(f"ChatGPT answer generated: {response.content[:100]}...")
        return {"chatgpt_answer": response.content}
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        st.error(f"Rate limit exceeded: {e}. Wait a minute and retry.")
        return {"chatgpt_answer": "Oops! Rate limit hit. Try again in a minute!"}

# Web search using DuckDuckGo
def web_search(state: AgentState) -> dict:
    question = state["current_question"]
    logger.debug(f"Starting web search for: {question}")
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(question, max_results=5)]
            context = "\n".join(results)
        logger.debug(f"Web search results: {context[:100]}...")
        return {"web_context": context}
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        st.error(f"Web search failed: {e}")
        return {"web_context": "Couldn’t fetch web info due to an error."}

# Generate web answer
def web_answer(state: AgentState, llm) -> dict:
    question = state["current_question"]
    web_context = state["web_context"]
    prompt = f"""Summarize this web info for: {question}
    Web Info: {web_context}
    Make it short, simple, and easy to read, like explaining to a friend."""
    prompt = trim_text(prompt, max_tokens=15000)
    try:
        response = llm.invoke(prompt)
        logger.debug(f"Web answer generated: {response.content[:100]}...")
        return {"web_answer": response.content}
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        st.error(f"Rate limit exceeded: {e}. Wait a minute and retry.")
        return {"web_answer": "Oops! Rate limit hit. Try again in a minute!"}

# Build workflow
def build_workflow(llm):
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve_book", retrieve_from_book)
    workflow.add_node("generate_book_answer", lambda state: book_answer(state, llm))
    workflow.add_node("generate_chatgpt_answer", lambda state: chatgpt_answer(state, llm))
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_web_answer", lambda state: web_answer(state, llm))
    workflow.add_edge(START, "retrieve_book")
    workflow.add_edge("retrieve_book", "generate_book_answer")
    workflow.add_edge("generate_book_answer", "generate_chatgpt_answer")
    workflow.add_edge("generate_chatgpt_answer", "web_search")
    workflow.add_edge("web_search", "generate_web_answer")
    workflow.add_edge("generate_web_answer", END)
    return workflow.compile()

# Session Management
def save_session(session_id, chat_history):
    session_data = {"chat_history": chat_history}
    with open(f"session_{session_id}.json", "w") as f:
        json.dump(session_data, f)

def load_session(session_id):
    try:
        with open(f"session_{session_id}.json", "r") as f:
            session_data = json.load(f)
        return session_data["chat_history"]
    except FileNotFoundError:
        return []

# Streamlit Interface
def main():
    st.markdown("""
        <style>
        .stApp { background-size: cover; background-position: center; transition: background-image 1s ease-in-out; }
        .chat-container { max-height: 500px; overflow-y: auto; padding: 10px; border-radius: 10px; background-color: #f9f9f9; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .chat-message { margin: 10px 0; padding: 15px; border-radius: 10px; line-height: 1.5; width: 100%; }
        .user-message { background-color: #007bff; color: white; margin-left: 20%; }
        .assistant-message { background-color: #e9ecef; color: black; margin-right: 20%; }
        .avatar { width: 40px; height: 40px; border-radius: 50%; margin-right: 10px; vertical-align: middle; }
        .user-avatar { margin-left: 10px; margin-right: 0; vertical-align: middle; }
        .stButton>button { border-radius: 10px; padding: 8px 15px; margin: 5px; }
        </style>
    """, unsafe_allow_html=True)

    backgrounds = [
        "https://images.unsplash.com/photo-1472289065668-ce650ac443d2",
        "https://images.unsplash.com/photo-1508615121316-fe792af62a63"
    ]
    if "bg_index" not in st.session_state:
        st.session_state.bg_index = 0
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()
    current_time = time.time()
    if current_time - st.session_state.last_update > 30:
        st.session_state.bg_index = (st.session_state.bg_index + 1) % len(backgrounds)
        st.session_state.last_update = current_time
    st.markdown(f"<style>.stApp {{background-image: url({backgrounds[st.session_state.bg_index]});}}</style>", unsafe_allow_html=True)

    st.title("Your Study Buddy 📚")

    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "pdf_title" not in st.session_state:
        st.session_state.pdf_title = None
    if "web_search_triggered" not in st.session_state:
        st.session_state.web_search_triggered = False

    with st.sidebar:
        st.header("API Key")
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
        
        st.header("Session Management")
        session_id_input = st.text_input("Enter Session ID (or leave blank for new session)")
        if st.button("Load Session") and session_id_input:
            st.session_state.session_id = session_id_input
            st.session_state.chat_history = load_session(session_id_input)
            st.success(f"Loaded session {session_id_input}")
        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API key!")
            return
        else:
            if st.session_state.llm is None or st.session_state.agent is None:
                st.session_state.llm = initialize_llm()
                st.session_state.agent = build_workflow(st.session_state.llm)

    st.write("Upload a PDF and choose how to extract content (Text or OCR). Ask me anything about it!")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.current_question = None
        st.session_state.web_search_triggered = False
        if st.session_state.session_id:
            save_session(st.session_state.session_id, st.session_state.chat_history)
        st.rerun()

    uploaded_file = st.file_uploader("Upload a PDF Book", type=["pdf"])
    extraction_method = st.radio("Choose Extraction Method", ("Text-Based", "OCR-Based"))
    pdf_title_input = st.text_input("Enter PDF Title (for database storage)")

    if uploaded_file is not None and st.session_state.retriever is None and pdf_title_input:
        st.session_state.pdf_title = pdf_title_input
        with open("temp_book.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner(f"Extracting and saving content using {extraction_method}..."):
            if extraction_method == "Text-Based":
                markdown_text = extract_text_from_pdf("temp_book.pdf")
            else:
                markdown_text = extract_images_and_ocr("temp_book.pdf")
            if markdown_text:
                save_to_db(st.session_state.pdf_title, markdown_text)
                vector_store = create_vector_store(markdown_text)
                st.session_state.retriever = vector_store
                st.success(f"Book '{st.session_state.pdf_title}' processed with {extraction_method} extraction, saved to DB, and ready to chat!")
            else:
                st.error("Failed to process the book.")
        os.remove("temp_book.pdf")

    if st.session_state.retriever is not None and st.session_state.agent is not None:
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.markdown(f'<img src="https://api.dicebear.com/9.x/pixel-art/svg?seed=user{random.randint(1, 100)}" class="avatar user-avatar"/>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<img src="https://api.dicebear.com/9.x/pixel-art/svg?seed=bot{random.randint(1, 100)}" class="avatar"/>', unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-message {message['role']}-message'>{message['content']}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        question = st.chat_input("Ask me anything about the book!")
        if question:
            st.session_state.current_question = question
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.web_search_triggered = False
            logger.debug(f"User asked: {question}")
            with st.chat_message("user"):
                st.markdown(f'<img src="https://api.dicebear.com/9.x/pixel-art/svg?seed=user{random.randint(1, 100)}" class="avatar user-avatar"/>', unsafe_allow_html=True)
                st.markdown(f"<div class='chat-message user-message'>{question}</div>", unsafe_allow_html=True)

            with st.spinner("Getting answers from the book, ChatGPT, and the web..."):
                initial_state = {
                    "messages": [HumanMessage(content=question)],
                    "book_context": "",
                    "book_answer": "",
                    "chatgpt_answer": "",
                    "web_context": "",
                    "web_answer": "",
                    "chat_history": st.session_state.chat_history[:-1],
                    "current_question": question
                }
                try:
                    result = st.session_state.agent.invoke(initial_state)
                    report_response = (
                        f"**Book Answer (from '{st.session_state.pdf_title}'):**\n\n{result['book_answer']}\n\n"
                        f"**Teacher Answer:**\n\n{result['chatgpt_answer']}\n\n"
                        f"**Web Search Answer:**\n\n{result['web_answer']}"
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": report_response})
                    with st.chat_message("assistant"):
                        st.markdown(f'<img src="https://api.dicebear.com/9.x/pixel-art/svg?seed=bot{random.randint(1, 100)}" class="avatar"/>', unsafe_allow_html=True)
                        st.markdown(f"<div class='chat-message assistant-message'>{report_response}</div>", unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Error generating answers: {e}")
                    st.error(f"Error generating answers: {e}")
                    st.session_state.chat_history.append({"role": "assistant", "content": "Oops! Something went wrong."})

                if not st.session_state.session_id:
                    st.session_state.session_id = str(random.randint(1000, 9999))
                    st.info(f"New session ID: {st.session_state.session_id}")
                save_session(st.session_state.session_id, st.session_state.chat_history)

    else:
        if st.session_state.retriever is None:
            st.warning("Please upload a book and provide a title to start chatting!")
        if st.session_state.agent is None:
            st.warning("Please enter your OpenAI API key!")

if __name__ == "__main__":
    main()
