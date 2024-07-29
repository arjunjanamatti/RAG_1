import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import language_tool_python
import gradio as gr
import faiss
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser

# Initialize the Sentence Transformer model
model_name = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(model_name)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an AI assistant. Your task is to provide accurate, concise, and contextually relevant answers based on the given context and conversation history. 

Context:
{context}

---

Conversation History:
{history}

---

Question: {question}

Answer the question based on the above context and conversation history. Ensure your response is clear, concise, and directly addresses the question. Feel free to include any additional relevant information.
"""

# Initialize conversation history
conversation_history = []

# Load GPT-2 model and tokenizer for contextual coherence
model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Initialize grammar and spell checker
tool = language_tool_python.LanguageTool('en-US')

# Initialize FAISS index
dimension = 768  # Adjust based on your embedding model
faiss_index = faiss.IndexFlatL2(dimension)

# Initialize Whoosh index
index_dir = "indexdir"
if not os.path.exists(index_dir):
    os.makedirs(index_dir)
schema = Schema(text=TEXT(stored=True))
whoosh_index = create_in(index_dir, schema)

# Define the documents to be indexed
documents = [
    {"text": "Arjun is currently working at Rockwell Automation."},
    {"text": "Document 2 text goes here."},
    # Add more documents as needed
]

def get_embedding_function(text):
    return embedding_model.encode([text])[0]

def index_documents(documents):
    writer = whoosh_index.writer()
    for doc in documents:
        embedding = get_embedding_function(doc['text'])
        faiss_index.add(np.array([embedding]))
        writer.add_document(text=doc['text'])
    writer.commit()

# # Index the documents
# index_documents(documents)

def rerank_results(results, query_embedding):
    # Re-rank results based on cosine similarity with the query embedding
    reranked_results = sorted(results, key=lambda x: np.dot(get_embedding_function(x['text']), query_embedding), reverse=True)
    return reranked_results

def query_rag(query_text: str):
    global conversation_history

    # Perform semantic search with FAISS
    query_embedding = get_embedding_function(query_text)
    _, semantic_indices = faiss_index.search(np.array([query_embedding]), 5)

    results = [{'text': documents[idx]['text'], 'score': 1.0} for idx in semantic_indices[0]]

    # Re-rank results
    results = rerank_results(results, query_embedding)

    context_text = "\n\n---\n\n".join([doc['text'] for doc in results])
    history_text = "\n".join(conversation_history)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, history=history_text, question=query_text)

    model = Ollama(model="phi3")
    response_text = model.invoke(prompt)

    # Post-process the response to improve quality
    response_text = post_process_response(response_text)

    sources = [doc['text'] for doc in results]
    formatted_response = f"Response: {response_text}"

    # Evaluate the response
    reference_texts = [doc['text'] for doc in results]
    evaluation_metrics = evaluate_response(response_text, reference_texts, context_text)

    # Update conversation history
    conversation_history.append(f"User: {query_text}")
    conversation_history.append(f"AI: {response_text}")

    return formatted_response, evaluation_metrics


def post_process_response(response_text):
    # Remove unnecessary whitespace
    response_text = response_text.strip()

    # Correct grammar and spelling
    matches = tool.check(response_text)
    response_text = language_tool_python.utils.correct(response_text, matches)

    # Ensure proper capitalization and punctuation
    response_text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', response_text)
    response_text = response_text.capitalize()

    # Remove redundancies (less aggressive)
    response_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', response_text)

    # Enhance clarity (example: simplifying complex sentences)
    # This step can be customized based on specific requirements

    return response_text

def evaluate_response(response_text, reference_texts, context_text):
    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_texts[0], response_text)
    rouge_score = scores['rougeL'].fmeasure

    # Contextual Coherence
    coherence_score = calculate_contextual_coherence(response_text, context_text)

    # Round off the values
    evaluation_metrics = {
        "rouge_score": round(rouge_score, 2),
        "contextual_coherence": round(coherence_score, 2)
    }

    return evaluation_metrics

def calculate_contextual_coherence(response_text, context_text):
    input_text = context_text + " " + response_text
    inputs = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    outputs = gpt2_model(inputs, labels=inputs)
    loss = outputs.loss.item()
    coherence_score = 1 / (1 + loss)  # Inverse of loss to get a coherence score
    return coherence_score

def chatbot_interface(query_text):
    response, evaluation_metrics = query_rag(query_text)
    return response, evaluation_metrics, "\n".join(conversation_history)

iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs=["text", "text", "text"],
    title="Chatbot UI",
    description="Ask your question below and get a response. The conversation history will be displayed in a chatbot-like format."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the chatbot interface.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio interface on.")
    args = parser.parse_args()
    iface.launch(server_port=args.port)
