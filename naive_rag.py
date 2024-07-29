import argparse
import gradio as gr
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context and conversation history:

{context}

---

Conversation History:
{history}

---

Answer the question based on the above context and conversation history: {question}
"""

# Initialize conversation history
conversation_history = []

def query_rag(query_text: str):
    global conversation_history
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    history_text = "\n".join(conversation_history)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, history=history_text, question=query_text)

    model = Ollama(model="phi3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}"

    # Update conversation history
    conversation_history.append(f"User: {query_text}")
    conversation_history.append(f"AI: {response_text}")

    return formatted_response, "\n".join(conversation_history)

def chatbot_interface(query_text):
    response, history = query_rag(query_text)
    return response, history

iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs=["text", "text"],
    title="Chatbot UI",
    description="Ask your question below and get a response. The conversation history will be displayed in a chatbot-like format."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the chatbot interface.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio interface on.")
    args = parser.parse_args()
    iface.launch(server_port=args.port)
