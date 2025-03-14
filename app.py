import os
import streamlit as st

# Override sqlite3 with pysqlite3 before importing Chroma
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from typing import TypedDict, Annotated, Literal
import PyPDF2
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

# Step 1: PDF to Markdown Conversion
def pdf_to_markdown(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        raw_text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            raw_text += f"\n\n## Page {page_num + 1}\n\n{page.extract_text()}"
        markdown_text = md(raw_text)
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

# Step 2: Define Agent State and Tools
class AgentState(TypedDict):
    messages: Annotated[list, "List of messages in the conversation"]
    book_context: Annotated[str, "Context retrieved from the book"]
    web_context: Annotated[str, "Context retrieved from the web"]
    book_answer: Annotated[str, "Simplified answer from the book"]
    full_answer: Annotated[str, "Final friendly and detailed answer"]
    needs_web_search: Annotated[bool, "Whether web search is needed"]
    chat_history: Annotated[list, "Full chat history for memory"]
    current_question: Annotated[str, "Current question for visualization"]

# Global tools
search_tool = TavilySearchResults(max_results=5)  # Uses TAVILY_API_KEY from env

def initialize_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        api_key=st.session_state.openai_api_key
    )

def retrieve_from_book(state: AgentState) -> dict:
    question = state["messages"][-1].content
    docs = st.session_state.retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    needs_web = not context or len(context.strip()) < 50
    return {"book_context": context, "needs_web_search": needs_web, "current_question": question}

def simplify_book_answer(state: AgentState, llm) -> dict:
    question = state["messages"][-1].content
    context = state["book_context"]
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]])
    prompt = f"""Here’s the chat history so far:
    {history}

    Answer this question using the Markdown-formatted book content: {question}
    Context: {context}
    If the context doesn’t have enough info, say 'Hmm, the book doesn’t tell me much about this!'"""
    response = llm.invoke(prompt)
    simplify_prompt = f"""Take this answer and explain it in a clear, easy-to-understand way—like you're teaching a curious beginner or a young student. Keep all the key information, break it down step by step, and use simple words, relatable examples, and a fun, engaging tone:
    Answer: {response.content}"""
    simplified = llm.invoke(simplify_prompt)
    return {"book_answer": simplified.content}

def web_search(state: AgentState) -> dict:
    question = state["current_question"]
    results = search_tool.invoke({"query": f"{question} detailed explanation for beginners"})
    context = "\n".join(result["content"] for result in results)
    return {"web_context": context}

def generate_full_answer(state: AgentState, llm) -> dict:
    question = state["current_question"]
    book_answer = state["book_answer"]
    web_context = state["web_context"]
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["chat_history"]])
    prompt = f"""You’re a friendly teacher who loves making things clear! Here’s the chat history:
    {history}

    Explain this question in a fun, detailed, and beginner-friendly way: {question}.
    Use the simple book answer and enrich it with web info:
    Book Answer: {book_answer}
    Web Info: {web_context}
    Format the response with proper paragraphs, bullet points for lists, and clear sections.
    Highlight key terms using HTML <span> tags with these colors:
    - Important concepts: <span style="color:#FF4500">term</span>
    - Processes: <span style="color:#1E90FF">term</span>
    - Examples: <span style="color:#32CD32">term</span>
    Make it engaging, accurate, and add a 'Basics of this Topic' section at the end!"""
    response = llm.invoke(prompt)
    return {"full_answer": response.content}

def route_after_book(state: AgentState) -> Literal["end", "web_search"]:
    if state["needs_web_search"]:
        return "web_search"
    return "end"

# Step 3: Build the Agent Workflow
def build_workflow(llm):
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve_book", retrieve_from_book)
    workflow.add_node("simplify_book", lambda state: simplify_book_answer(state, llm))
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_full", lambda state: generate_full_answer(state, llm))

    workflow.add_edge(START, "retrieve_book")
    workflow.add_edge("retrieve_book", "simplify_book")
    workflow.add_conditional_edges("simplify_book", route_after_book, {"end": END, "web_search": "web_search"})
    workflow.add_edge("web_search", "generate_full")
    workflow.add_edge("generate_full", END)

    return workflow.compile()

# Enhanced Visualization Function
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

        table_data = {
            "Input": ["Sunlight", "Water", "CO2"],
            "Output": ["Glucose", "Oxygen", "N/A"]
        }
        df = pd.DataFrame(table_data)
        return img_base64, df
    elif "deep learning" in topic:
        G = nx.DiGraph()
        layers = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"]
        for i, layer in enumerate(layers):
            G.add_node(layer, pos=(i, 0))
        for i in range(len(layers) - 1):
            G.add_edge(layers[i], layers[i + 1])

        plt.figure(figsize=(8, 5))
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, node_color="#87CEEB", node_size=3000, font_size=10, font_weight="bold", arrowsize=20)
        plt.title("Deep Learning Architecture")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        table_data = {
            "Layer": ["Input", "Hidden 1", "Hidden 2", "Output"],
            "Function": ["Receives data", "Learns patterns", "Refines patterns", "Gives result"]
        }
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

# Step 4: Streamlit Study Interface
def main():
    st.markdown("""
        <style>
        .stApp {
            background-size: cover;
            background-position: center;
            transition: background-image 1s ease-in-out;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 10px;
            background-color: #f9f9f9;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chat-message {
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
            line-height: 1.5;
            width: 100%;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: 20%;
        }
        .assistant-message {
            background-color: #e9ecef;
            color: black;
            margin-right: 20%;
        }
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 10px;
            vertical-align: middle;
        }
        .user-avatar {
            margin-left: 10px;
            margin-right: 0;
            vertical-align: middle;
        }
        .stButton>button {
            border-radius: 10px;
            padding: 8px 15px;
            margin: 5px;
        }
        .visualization {
            margin-top: 20px;
            text-align: center;
        }
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
    
    # Initialize session state
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

    with st.sidebar:
        st.header("API Key")
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password"
        )
        st.info("Note: Tavily API Key must be set as an environment variable (TAVILY_API_KEY) in Streamlit Cloud settings.")
        
        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API key to proceed!")
            return
        else:
            # Initialize LLM and agent only after API key is provided
            if st.session_state.llm is None or st.session_state.agent is None:
                st.session_state.llm = initialize_llm()
                st.session_state.agent = build_workflow(st.session_state.llm)

    # Check if TAVILY_API_KEY is set in the environment
    if not os.getenv("TAVILY_API_KEY"):
        st.error("TAVILY_API_KEY environment variable not found. Please set it in Streamlit Cloud settings.")
        return

    st.write("Upload a book and chat with me! I’ll explain things with visuals and fun facts!")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.pending_web_search = None
        st.session_state.current_question = None
        st.rerun()

    uploaded_file = st.file_uploader("Upload a PDF Book", type=["pdf"])
    if uploaded_file is not None and st.session_state.retriever is None:
        with open("temp_book.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        markdown_text = pdf_to_markdown("temp_book.pdf")
        vector_store = create_vector_store(markdown_text)
        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        st.success("Book converted to Markdown and ready to chat!")
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
                result = st.session_state.agent.invoke(initial_state)

                book_response = f"**From the Book (Simple):** {result['book_answer']}"
                if result["needs_web_search"]:
                    book_response += f"\n\n**More Fun Info:** {result['full_answer']}"
                    img_base64, table_data = generate_visualization(question)
                    book_response += f"\n\n<div class='visualization'><img src='data:image/png;base64,{img_base64}' style='max-width:100%;'></div>"
                    if table_data is not None:
                        book_response += "\n\n**Related Data:**"
                        book_response += f"\n\n{table_data.to_html(index=False, classes='table table-striped')}"
                st.session_state.chat_history.append({"role": "assistant", "content": book_response})
                with st.chat_message("assistant"):
                    st.markdown(f'<img src="https://api.dicebear.com/9.x/pixel-art/svg?seed=bot{random.randint(1, 100)}" class="avatar"/>', unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-message assistant-message'>{book_response}</div>", unsafe_allow_html=True)

                if not result["needs_web_search"]:
                    st.session_state.pending_web_search = result

        if st.session_state.pending_web_search:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Want more info from the web?"):
                    with st.spinner("Searching the web for you..."):
                        web_state = st.session_state.pending_web_search.copy()
                        web_state["chat_history"] = st.session_state.chat_history
                        web_state["web_context"] = web_search(web_state)["web_context"]
                        web_state["full_answer"] = generate_full_answer(web_state, st.session_state.llm)  # Use st.session_state.llm
                        full_response = f"**More Fun Info:** {web_state['full_answer']}"
                        img_base64, table_data = generate_visualization(st.session_state.current_question)
                        full_response += f"\n\n<div class='visualization'><img src='data:image/png;base64,{img_base64}' style='max-width:100%;'></div>"
                        if table_data is not None:
                            full_response += "\n\n**Related Data:**"
                            full_response += f"\n\n{table_data.to_html(index=False, classes='table table-striped')}"
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                        with st.chat_message("assistant"):
                            st.markdown(f'<img src="https://api.dicebear.com/9.x/pixel-art/svg?seed=bot{random.randint(1, 100)}" class="avatar"/>', unsafe_allow_html=True)
                            st.markdown(f"<div class='chat-message assistant-message'>{full_response}</div>", unsafe_allow_html=True)
                        st.session_state.pending_web_search = None
            with col2:
                if st.button("No, I’m good!"):
                    st.session_state.pending_web_search = None
    else:
        if st.session_state.retriever is None:
            st.warning("Please upload a book to start chatting!")
        if st.session_state.agent is None:
            st.warning("Please enter your OpenAI API key to enable the chat functionality!")

if __name__ == "__main__":
    main()
