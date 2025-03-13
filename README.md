"# Study_Buddy" 


# Study Buddy Web App

Welcome to the **Study Buddy Web App**, an interactive AI-powered tool designed to assist learners by providing simplified explanations from uploaded PDF books, enriched with web-sourced information and visualizations. Built with Streamlit, LangChain, and OpenAI, this web app mimics a friendly teacher, making complex topics easy to understand with graphs, charts, and tabular data.

## What This Web App Does

The Study Buddy Web App allows users to:
- **Upload a PDF book** to create a knowledge base.
- **Ask questions** about the book's content in a chat interface.
- **Receive simplified answers** tailored for beginners, drawn from the book.
- **Request additional web info** to expand on topics, with detailed explanations.
- **View visualizations** (e.g., flowcharts, concept maps) and tabular data to enhance understanding.
- **Maintain conversation memory** within a session to reference previous topics.
- **Highlight key terms** in responses with colors for better focus.

The app leverages a Retrieval-Augmented Generation (RAG) system, combining book content with web searches via the Tavily API, and uses OpenAI's GPT-4o-mini for natural language processing. Visualizations are generated using Matplotlib and NetworkX, while tabular data is presented using Pandas.

## Features
- **Interactive Chat UI**: A modern, ChatGPT-like interface with avatars and a scrollable chat history.
- **Dynamic Visuals**: Graphs and charts tailored to topics like photosynthesis or deep learning.
- **Clear Formatting**: Responses include paragraphs, bullet points, and highlighted keywords.
- **Session Memory**: Remembers previous questions within the same session.
- **Clear Chat Option**: Allows users to reset the conversation easily.

## Prerequisites

Before running the web app, ensure you have the following installed:

- **Python 3.8 or higher**
- Required Python packages (install via `pip`):
  ```bash
  pip install streamlit langchain langgraph langchain-openai langchain-community chromadb tavily-python PyPDF2 markdownify matplotlib networkx pandas
  ```
- **Environment Variables**:
  - `OPENAI_API_KEY`: Your OpenAI API key.
  - `TAVILY_API_KEY`: Your Tavily API key for web searches.

## How to Run

Follow these steps to set up and run the web app locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/study-buddy-web-app.git
   cd study-buddy-web-app
   ```

2. **Set Up Environment Variables**:
   - Create a `.env` file in the project root directory.
   - Add the following lines with your API keys:
     ```
     OPENAI_API_KEY=your-openai-api-key
     TAVILY_API_KEY=your-tavily-api-key
     ```
   - Alternatively, export them in your terminal:
     ```bash
     export OPENAI_API_KEY=your-openai-api-key
     export TAVILY_API_KEY=your-tavily-api-key
     ```

3. **Install Dependencies**:
   Run the following command to install all required packages:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: Create a `requirements.txt` file with the package list if not already present, e.g., using `pip freeze > requirements.txt` after installation.)

4. **Run the Web App**:
   Launch the Streamlit app with:
   ```bash
   streamlit run main_rag_agent.py
   ```
   - Open your browser and navigate to `http://localhost:8501`.

5. **Usage**:
   - Upload a PDF book using the file uploader.
   - Type a question in the chat input (e.g., "What is photosynthesis?").
   - Review the simplified answer and click "Want more info from the web?" for additional details and visualizations.
   - Use the "Clear Chat" button to reset the conversation.

## Example Interaction

- **User**: "What is photosynthesis?"
- **Assistant**: 
  ```
  **From the Book (Simple):** Photosynthesis is when plants use sunlight to make food. It’s like magic with leaves!
  ```
  - Click "Want more info from the web?":
  ```
  **More Fun Info:**
  Let’s explore <span style="color:#FF4500">photosynthesis</span>! Imagine plants as tiny chefs using sunlight to cook their food. The <span style="color:#1E90FF">process</span> starts with:
  - <span style="color:#32CD32">Sunlight</span> powering the plant.
  - <span style="color:#32CD32">Water</span> and <span style="color:#32CD32">CO2</span> from the air mixing in the leaves.
  - The plant creates <span style="color:#32CD32">glucose</span> for energy and releases <span style="color:#32CD32">oxygen</span> for us to breathe!

  **Basics of this Topic:** <span style="color:#FF4500">Photosynthesis</span> is how plants make food using sunlight, water, and air.

  **Visualization:** [Photosynthesis Flowchart]
  **Related Data:**
  | Input       | Output    |
  |-------------|-----------|
  | Sunlight    | Glucose   |
  | Water       | Oxygen    |
  | CO2         | N/A       |
  ```

## Contributing

Feel free to contribute to this project! Submit issues or pull requests via GitHub. Suggestions for new features (e.g., more visualization types, persistent memory) are welcome!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


