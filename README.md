# RAG solution for documents using Langchain

This project implements a chatbot interface using Gradio, LangChain, and Ollama. The chatbot answers questions based on a given context from the documents in "data" folder and conversation history, utilizing a Chroma vector store for similarity search.

## Requirements

- Python 3.7+
- Gradio
- LangChain
- Ollama
- Chroma
- argparse

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/arjunjanamatti/RAG_1
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the chatbot interface, use the following command:
```bash
python updated_rag.py --port 7860
```
You can specify a different port if needed by changing the --port argument

## Script Overview
chatbot_interface.py
* Imports: The script imports necessary libraries including argparse, gradio, Chroma from langchain.vectorstores, ChatPromptTemplate from langchain.prompts, Ollama from langchain_community.llms, and get_embedding_function.
* Constants:
    * CHROMA_PATH: Path to the Chroma vector store.
    * PROMPT_TEMPLATE: Template for generating the prompt used by the model.
    * Global Variables:
    * conversation_history: List to store the conversation history.
* Functions:
    * query_rag(query_text: str): Handles the query, searches the Chroma vector store, formats the prompt, invokes the model, and updates the conversation history. Includes exception handling for errors and empty results.
    * chatbot_interface(query_text): Interface function for Gradio, calling query_rag and returning the response and history.
* Main Execution:
    * Uses argparse to handle command-line arguments for specifying the port.
    * Launches the Gradio interface on the specified port.
* Example
    * After running the script, open your browser and navigate to http://localhost:7860 (or the specified port). You will see a simple UI where you can input your question and receive a response along with the conversation history.

License
This project is licensed under the MIT License. See the LICENSE file for details.
