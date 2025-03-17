import os
import streamlit as st
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from typing import TypedDict, Annotated, Literal
import PyPDF2
from PIL import Image
import pytesseract
from markdownify import markdownify as md
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langchain_community.tools.tavily_search import TavilySearchResults
import time
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
import random
import pandas as pd
import tiktoken
import json
from openai import RateLimitError

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

# Extract images and perform OCR
def extract_images_and_ocr(file_path):
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract is not installed or not in your PATH. Install it (e.g., 'sudo apt-get install tesseract-ocr' or add to packages.txt for Streamlit Cloud).")
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
                            ocr_text = pytesseract.image_to_string(img)
                            combined_text += f"\n\n### Image Text (Page {page_num + 1})\n\n{ocr_text}"
                        except Exception as e:
                            st.warning(f"Error processing image on page {page_num + 1}: {e}")
    markdown_text = md(combined_text)
    return markdown_text

def create_vector_store(markdown_text):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.session_state.openai_api_key
    )
    vector_store = Chroma.from_texts(
        [markdown_text], embeddings, collection_name="book_content", persist_directory="book_db"
    )
    return vector_store

# Agent State and Tools
class AgentState(TypedDict):
    messages: Annotated[list, "List of messages in the conversation"]
    book_context: Annotated[str, "Context retrieved from the book"]
    web_context: Annotated[str, "Context retrieved from the web"]
    book_answer: Annotated[str, "Answer from the book"]
    full_answer: Annotated[str, "Final answer"]
    needs_web_search: Annotated[bool, "Whether web search is needed"]
    chat_history: Annotated[list, "Full chat history for memory"]
    current_question: Annotated[str, "Current question"]

search_tool = TavilySearchResults(max_results=5)

def initialize_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.5,
        api_key=st.session_state.openai_api_key
    )

def retrieve_from_book(state: AgentState) -> dict:
    question = state["messages"][-1].content
    docs = st.session_state.retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    context = trim_text(context, max_tokens=10000)
    needs_web = not context or len(context.strip()) < 50
    return {"book_context": context, "needs_web_search": needs_web, "current_question": question}

def book_answer(state: AgentState, llm) -> dict:
    question = state["messages"][-1].content
    context = state["book_context"]
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"][-2:]])
    prompt = f"""Based on this recent chat history:
    {history}

    Answer this question using the book content: {question}
    Context: {context}
    If the context doesn’t have enough info, say 'The book doesn’t give me much on this, sorry!' 
    Keep it natural and easy to follow, like you’re explaining it to a friend."""
    prompt = trim_text(prompt, max_tokens=15000)
    try:
        response = llm.invoke(prompt)
        return {"book_answer": response.content}
    except RateLimitError as e:
        st.error(f"Rate limit exceeded: {e}. Please wait a minute and try again.")
        return {"book_answer": "Oops! I hit a rate limit. Try again in a minute!"}

def web_search(state: AgentState) -> dict:
    question = state["current_question"]
    results = search_tool.invoke({"query": question})
    context = "\n".join(result["content"] for result in results)
    return {"web_context": context}

def summarize_web_results(state: AgentState, llm) -> dict:
    question = state["current_question"]
    web_context = state["web_context"]
    prompt = f"""Here’s a bunch of info I found on the web about: {question}
    Web Info: {web_context}

    Summarize this in a way that’s easy to understand, like you’re chatting with a friend. 
    Include a simple example to make it clear. Keep the original meaning intact."""
    prompt = trim_text(prompt, max_tokens=15000)
    try:
        response = llm.invoke(prompt)
        return {"full_answer": response.content}
    except RateLimitError as e:
        st.error(f"Rate limit exceeded: {e}. Please wait a minute and try again.")
        return {"full_answer": "Oops! I hit a rate limit. Try again in a minute!"}

def route_after_book(state: AgentState) -> Literal["end", "web_search"]:
    if state["needs_web_search"]:
        return "web_search"
    return "end"

def build_workflow(llm):
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve_book", retrieve_from_book)
    workflow.add_node("generate_book_answer", lambda state: book_answer(state, llm))  # Renamed node
    workflow.add_node("web_search", web_search)
    workflow.add_node("summarize_web", lambda state: summarize_web_results(state, llm))
    workflow.add_edge(START, "retrieve_book")
    workflow.add_edge("retrieve_book", "generate_book_answer")
    workflow.add_conditional_edges("generate_book_answer", route_after_book, {"end": END, "web_search": "web_search"})
    workflow.add_edge("web_search", "summarize_web")
    workflow.add_edge("summarize_web", END)
    return workflow.compile()

# Visualization Function (unchanged)
def generate_visualization(topic):
    if topic is None:
        topic = "Unknown Topic"
    topic = topic.lower()
    if "photosynthesis" in topic:
        G = nx.DiGraph()
        G.add_node("Sunlight", pos=(0, 2))
        G.add_node("Water", pos=(-1, 1))
        G.add_node("CO2", pos=(1, 1))
        G.add_node("Plant", pos=(0, 0))
        G.add_node("Glucose", pos=(-1, -1))
        G.add_node("Oxygen", pos=(1, -1))
        G.add_edge("Sunlight", "Plant")
        G.add_edge("Water", "Plant")
        G.add_edge("CO2", "Plant")
        G.add_edge("Plant", "Glucose")
        G.add_edge("Plant", "Oxygen")
        plt.figure(figsize=(8, 5))
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, node_color="#90EE90", node_size=3000, font_size=10, font_weight="bold", arrowsize=20)
        plt.title("Photosynthesis Flowchart")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        table_data = {"Input": ["Sunlight", "Water", "CO2"], "Output": ["Glucose", "Oxygen", "N/A"]}
        df = pd.DataFrame(table_data)
        return img_base64, df
    else:
        G = nx.DiGraph()
        G.add_node(topic.capitalize(), pos=(0, 0))
        G.add_node("Concept 1", pos=(-1, 1))
        G.add_node("Concept 2", pos=(1, 1))
        G.add_edge(topic.capitalize(), "Concept 1")
        G.add_edge(topic.capitalize(), "Concept 2")
        plt.figure(figsize=(8, 5))
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, node_color="#FFD700", node_size=3000, font_size=10, font_weight="bold", arrowsize=20)
        plt.title(f"Concept Map for {topic.capitalize()}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return img_base64, None

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
        .visualization { margin-top: 20px; text-align: center; }
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
    if "pending_web_search" not in st.session_state:
        st.session_state.pending_web_search = None
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    with st.sidebar:
        st.header("API Key")
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key", value=st.session_state.openai_api_key, type="password"
        )
        st.info("Tavily API Key must be set as TAVILY_API_KEY in Streamlit Cloud settings.")
        
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

    if not os.getenv("TAVILY_API_KEY"):
        st.error("TAVILY_API_KEY not found. Set it in Streamlit Cloud settings.")
        return

    st.write("Upload a book (PDF with image-based content) and chat with me! I’ll extract text from images and explain things with visuals and fun facts!")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.pending_web_search = None
        st.session_state.current_question = None
        if st.session_state.session_id:
            save_session(st.session_state.session_id, st.session_state.chat_history)
        st.rerun()

    uploaded_file = st.file_uploader("Upload a PDF Book", type=["pdf"])
    if uploaded_file is not None and st.session_state.retriever is None:
        with open("temp_book.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Extracting text from images..."):
            markdown_text = extract_images_and_ocr("temp_book.pdf")
            if markdown_text:
                vector_store = create_vector_store(markdown_text)
                st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                st.success("Book processed (text extracted from images) and ready to chat!")
            else:
                st.error("Failed to process the book due to Tesseract issues.")
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
            with st.chat_message("user"):
                st.markdown(f'<img src="https://api.dicebear.com/9.x/pixel-art/svg?seed=user{random.randint(1, 100)}" class="avatar user-avatar"/>', unsafe_allow_html=True)
                st.markdown(f"<div class='chat-message user-message'>{question}</div>", unsafe_allow_html=True)

            with st.spinner("Checking the book..."):
                initial_state = {
                    "messages": [HumanMessage(content=question)],
                    "book_context": "",
                    "web_context": "",
                    "book_answer": "",
                    "full_answer": "",
                    "needs_web_search": False,
                    "chat_history": st.session_state.chat_history[:-1],
                    "current_question": question
                }
                try:
                    result = st.session_state.agent.invoke(initial_state)
                    book_response = f"**From the Book:** {result['book_answer']}"
                    img_base64, table_data = generate_visualization(question)
                    book_response += f"\n\n<div class='visualization'><img src='data:image/png;base64,{img_base64}' style='max-width:100%;'></div>"
                    if table_data is not None:
                        book_response += f"\n\n**Related Data:**\n\n{table_data.to_html(index=False, classes='table table-striped')}"
                    st.session_state.chat_history.append({"role": "assistant", "content": book_response})
                    with st.chat_message("assistant"):
                        st.markdown(f'<img src="https://api.dicebear.com/9.x/pixel-art/svg?seed=bot{random.randint(1, 100)}" class="avatar"/>', unsafe_allow_html=True)
                        st.markdown(f"<div class='chat-message assistant-message'>{book_response}</div>", unsafe_allow_html=True)
                    if not result["needs_web_search"]:
                        st.session_state.pending_web_search = result
                except RateLimitError as e:
                    st.error(f"Rate limit exceeded: {e}. Please wait a minute and try again.")
                    st.session_state.chat_history.append({"role": "assistant", "content": "Oops! I hit a rate limit. Try again in a minute!"})

                if not st.session_state.session_id:
                    st.session_state.session_id = str(random.randint(1000, 9999))
                    st.info(f"New session ID: {st.session_state.session_id}")
                save_session(st.session_state.session_id, st.session_state.chat_history)

        if st.session_state.pending_web_search:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Want more info from the web?"):
                    with st.spinner("Searching the web..."):
                        web_state = st.session_state.pending_web_search.copy()
                        web_state["chat_history"] = st.session_state.chat_history
                        web_state["web_context"] = web_search(web_state)["web_context"]
                        try:
                            web_summary = summarize_web_results(web_state, st.session_state.llm)
                            full_response = f"**Here’s What I Found Online:**\n\n{web_summary['full_answer']}"
                            img_base64, table_data = generate_visualization(st.session_state.current_question)
                            full_response += f"\n\n<div class='visualization'><img src='data:image/png;base64,{img_base64}' style='max-width:100%;'></div>"
                            if table_data is not None:
                                full_response += f"\n\n**Related Data:**\n\n{table_data.to_html(index=False, classes='table table-striped')}"
                            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                            with st.chat_message("assistant"):
                                st.markdown(f'<img src="https://api.dicebear.com/9.x/pixel-art/svg?seed=bot{random.randint(1, 100)}" class="avatar"/>', unsafe_allow_html=True)
                                st.markdown(f"<div class='chat-message assistant-message'>{full_response}</div>", unsafe_allow_html=True)
                            st.session_state.pending_web_search = None
                            save_session(st.session_state.session_id, st.session_state.chat_history)
                        except RateLimitError as e:
                            st.error(f"Rate limit exceeded: {e}. Please wait a minute and try again.")
                            st.session_state.chat_history.append({"role": "assistant", "content": "Oops! I hit a rate limit. Try again in a minute!"})
            with col2:
                if st.button("No, I’m good!"):
                    st.session_state.pending_web_search = None
    else:
        if st.session_state.retriever is None:
            st.warning("Please upload a book to start chatting!")
        if st.session_state.agent is None:
            st.warning("Please enter your OpenAI API key!")

if __name__ == "__main__":
    main()
