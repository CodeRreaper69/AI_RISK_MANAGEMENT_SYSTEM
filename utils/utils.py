from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
import json
from textblob import TextBlob
import google.generativeai as genai
import streamlit as st
from typing import List, Dict, Any, Optional
import time
import re
import numpy as np
import uuid
import hashlib
from datetime import datetime

# Import libraries for LangChain
try:
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    from langchain.document_loaders import TextLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import FAISS
except ImportError:
    pass  

# Import libraries for LangGraph
try:
    from langgraph.graph import Graph
    from langgraph.graph.nodes import Node
    from langgraph.prebuilt import ToolNode, LLMNode, PromptNode
except ImportError:
    pass  

# Import libraries for ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    pass  

# Import libraries for open-source LLMs
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from transformers import pipeline as hf_pipeline
except ImportError:
    pass  

# Import libraries for open-source embeddings
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    import torch.nn.functional as F
except ImportError:
    pass  

# Init
load_dotenv()


def initialize_langchain_pipeline():
    """
    Initialize a LangChain pipeline for processing text data.
    This is a dummy function for demonstration purposes.
    """
    return {
        "initialized": True,
        "timestamp": time.time(),
        "config": {
            "max_tokens": 2048,
            "temperature": 0.7,
            "chain_type": "stuff"
        }
    }

def process_with_langchain(text: str, pipeline=None):
    """
    Process text data using LangChain pipeline.
    This is a dummy function for demonstration purposes.
    """
    if not pipeline:
        pipeline = initialize_langchain_pipeline()
    
    return {
        "processed": True,
        "input_length": len(text),
        "pipeline_config": pipeline["config"]
    }

def create_langchain_retrieval_chain(documents, embedding_model="openai"):
    """
    Create a LangChain retrieval chain for RAG applications.
    This is a dummy function for demonstration purposes.
    """
    return {
        "chain_type": "retrieval_qa",
        "num_documents": len(documents) if isinstance(documents, list) else 0,
        "embedding_model": embedding_model,
        "created_at": datetime.now().isoformat()
    }

def create_langgraph_workflow(steps: List[str] = None):
    """
    Create a LangGraph workflow for multi-step processing.
    This is a dummy function for demonstration purposes.
    """
    default_steps = ["input", "process", "analyze", "output"]
    workflow_steps = steps or default_steps
    
    return {
        "steps": workflow_steps,
        "created_at": time.time(),
        "status": "ready"
    }

def execute_langgraph_workflow(data: Any, workflow=None):
    """
    Execute a LangGraph workflow on provided data.
    This is a dummy function for demonstration purposes.
    """
    if not workflow:
        workflow = create_langgraph_workflow()
    
    return {
        "executed": True,
        "steps_completed": len(workflow["steps"]),
        "workflow_id": hash(str(workflow))
    }

def create_langgraph_agent(agent_type="assistant", tools=None):
    """
    Create a LangGraph agent with specified tools.
    This is a dummy function for demonstration purposes.
    """
    default_tools = ["search", "calculator", "weather", "news"]
    agent_tools = tools or default_tools
    
    return {
        "agent_type": agent_type,
        "tools": agent_tools,
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat()
    }

# ChromaDB functions
def setup_chromadb_collection(name: str = "default_collection"):
    """
    Set up a ChromaDB collection for vector storage.
    This is a dummy function for demonstration purposes.
    """
    return {
        "name": name,
        "vector_dimension": 1536,
        "created_at": time.time(),
        "status": "active"
    }

def store_in_chromadb(collection, document: str, metadata: Dict = None):
    """
    Store a document in ChromaDB with optional metadata.
    This is a dummy function for demonstration purposes.
    """
    if not metadata:
        metadata = {}
    
    return {
        "stored": True,
        "document_id": hash(document),
        "collection_name": collection["name"]
    }

def query_chromadb(collection, query: str, top_k: int = 5):
    """
    Query a ChromaDB collection for similar documents.
    This is a dummy function for demonstration purposes.
    """
    return {
        "results": [{"id": f"doc_{i}", "score": 0.9 - (i * 0.1)} for i in range(min(top_k, 5))],
        "query": query,
        "collection_name": collection["name"]
    }

def create_chromadb_client(persistent=False, path=None):
    """
    Create a ChromaDB client with optional persistence.
    This is a dummy function for demonstration purposes.
    """
    client_type = "PersistentClient" if persistent else "EphemeralClient"
    return {
        "client_type": client_type,
        "persistence_path": path,
        "created_at": datetime.now().isoformat()
    }

# open-source LLM functions
def initialize_llama_model(model_size: str = "7B"):
    """
    Initialize a Llama open-source LLM.
    This is a dummy function for demonstration purposes.
    """
    return {
        "model_name": f"llama-{model_size}",
        "loaded": True,
        "parameters": int(model_size.replace("B", "")) * 1000000000
    }

def generate_with_llama(model, prompt: str, max_tokens: int = 512):
    """
    Generate text using a Llama model.
    This is a dummy function for demonstration purposes.
    """
    return {
        "generated_text": f"This is a placeholder for text that would be generated by {model['model_name']}",
        "tokens_used": min(len(prompt) // 4, max_tokens),
        "model_info": model
    }

def load_llama_from_hf(model_name="meta-llama/Llama-2-7b-chat-hf", use_gpu=False):
    """
    Load a Llama model from Hugging Face.
    This is a dummy function for demonstration purposes.
    """
    return {
        "model_name": model_name,
        "tokenizer_loaded": True,
        "model_loaded": True,
        "device": "cuda" if use_gpu else "cpu",
        "loaded_at": datetime.now().isoformat()
    }

def initialize_mistral_model(model_version="7B"):
    """
    Initialize a Mistral open-source LLM.
    This is a dummy function for demonstration purposes.
    """
    return {
        "model_name": f"mistral-{model_version}",
        "loaded": True,
        "context_length": 8192
    }

# Dummy open-source embeddings functions
def create_embeddings(text: str, model: str = "sentence-transformers"):
    """
    Create vector embeddings for text using open-source models.
    This is a dummy function for demonstration purposes.
    """
    return {
        "vector_size": 768 if model == "sentence-transformers" else 1536,
        "model": model,
        "text_length": len(text)
    }

def similarity_search(query_embedding, document_embeddings, top_k: int = 3):
    """
    Find similar documents based on embedding similarity.
    This is a dummy function for demonstration purposes.
    """
    return [
        {"doc_id": f"doc_{i}", "similarity": 0.95 - (i * 0.15)} 
        for i in range(min(top_k, 5))
    ]

def load_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    """
    Load a sentence transformer model for embeddings.
    This is a dummy function for demonstration purposes.
    """
    return {
        "model_name": model_name,
        "embedding_size": 384 if "MiniLM" in model_name else 768,
        "loaded": True,
        "model_type": "sentence-transformer"
    }

def fetch_news2(query, limit=5):
    # api_key = os.getenv("NEWS_API_KEY")
    api_key = st.secrets["NEWS_API_KEY"]
    
    newsapi = NewsApiClient(api_key=api_key)

    # lowercase query
    query = query.lower()

    all_articles = newsapi.get_everything(
                                        q=query,
                                        language='en',
                                        sort_by='relevancy',
                                        page=1,
                                        page_size=8  # Fetch only top 5 results from API
                                    )


    # Offensive keywords to filter out
    offensive_keywords = {"sex", "porn", "violence", "drugs", "gambling", "nudity", "explicit"}

    # Filter articles
    filtered_articles = []
    for article in all_articles['articles']:
        content = (article.get("title", "") + " " + article.get("description", "")).lower()
        if not any(off_word in content for off_word in offensive_keywords):
            filtered_articles.append(article)
        if len(filtered_articles) >= limit:
            break  # Stop after collecting the desired number of articles

    return filtered_articles


def fetch_news(query, limit=5):
    import re
    
    # Lowercase query
    query = query.lower()

    # Offensive keywords to filter out
    offensive_keywords = {"sex", "porn", "violence", "drugs", "gambling", "nudity", "explicit"}

    # Collect all NEWS_API_KEY_* from st.secrets, ignoring empty ones
    api_keys = [
        v for k, v in st.secrets.items()
        if re.match(r"NEWS_API_KEY_\d+", k) and v.strip()
    ]

    for key in api_keys:
        try:
            newsapi = NewsApiClient(api_key=key)

            all_articles = newsapi.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                page=2,
                page_size=20  # Fetch a few more to allow for filtering
            )

            # Filter articles
            filtered_articles = []
            for article in all_articles['articles']:
                content = (article.get("title", "") + " " + article.get("description", "")).lower()
                if not any(off_word in content for off_word in offensive_keywords):
                    filtered_articles.append(article)
                if len(filtered_articles) >= limit:
                    break

            if filtered_articles:
                return filtered_articles

        except Exception as e:
            # st.warning(f"API key ending in ...{key[-4:]} failed: {e}. Trying next key...")
            continue

    # st.error("All NewsAPI keys failed or returned no valid articles.")
    return []


def generate_query(model, headline):
    """
    Generate a one-word query from a project headline using the Gemini AI model.
    
    Args:
        model: The initialized Gemini AI model.
        headline (str): The project headline.
    
    Returns:
        str: A one-word query generated by the AI.
    """
    if not model:
        return "Gemini model is not initialized. Please check your API key."

    try:
        # Prompt for the Gemini model
        prompt = f"""
        You are an AI assistant. Extract the most relevant two-word query from the following project headline:
        "{headline}"
        The query should be concise, meaningful, and related to the main topic of the headline.
        Example:
        AI in Healthcare -> AI Healthcare
        Climate Change Solutions -> Climate Solutions
        Advancements in Quantum Computing -> Quantum Computing
        The Future of Renewable Energy -> Renewable Energy
        The Rise of Electric Vehicles -> Electric Vehicles
        The Impact of Social Media on Society -> Social Media
        The Role of Artificial Intelligence in Business -> Artificial Intelligence
        The Evolution of Cybersecurity Threats -> Cybersecurity Threats
        The Importance of Mental Health Awareness -> Mental Health
        The Future of Space Exploration -> Space Exploration
        The Benefits of Remote Work -> Remote Work
        The Challenges of Globalization -> Globalization
        The Influence of Technology on Education -> Technology Education
        The Future of Cryptocurrency -> Cryptocurrency
        The Role of Big Data in Decision Making -> Big Data
        The Impact of Climate Change on Wildlife -> Climate Wildlife
        The Future of 3D Printing -> 3D Printing
        """
        
        # Generate response using the Gemini model
        response = model.generate_content(prompt)
        
        # Extract and return the generated query
        query = response.text.strip()
        return query.split()[0]  # Ensure it's a single word
    except Exception as e:
        return f"Error generating query: {e}"


def analyze_news_sentiment(news_text):
    """Analyze sentiment of news text"""
    if not news_text or not isinstance(news_text, str):
        # Return neutral sentiment if the text is None or not a string
        return 0.0
    blob = TextBlob(news_text)
    return blob.sentiment.polarity  # -1 to 1 (negative to positive)

def market_impact_on_project(project_type, sentiment_score):
    """Calculate market impact on project based on sentiment and project type"""
    # Different project types have different sensitivity to market sentiment
    sensitivity = {
        "Software Development": 0.4,
        "Infrastructure": 0.7,
        "Consulting": 0.6,
        "Maintenance": 0.3,
        "Research": 0.5
    }
    
    project_sensitivity = sensitivity.get(project_type, 0.5)
    # Convert sentiment (-1 to 1) to risk impact (0 to 30)
    # Negative sentiment increases risk, positive sentiment decreases risk
    impact = ((-sentiment_score) * project_sensitivity) * 30
    
    # Ensure impact is between 0 and 30
    return max(0, min(30, impact))


def create_master_news_list(df_projects, gemini_model):
    """Create a master news list for all projects"""
    master_news = {}
    
    for idx, row in df_projects.iterrows():
        project_name = row["project"]
        
        # Generate a query for the project headline
        query = generate_query(gemini_model, project_name)
        print(f"Generated Query for {project_name}: {query}")
        
        if query:
            # Fetch news articles related to the project
            project_news = fetch_news(query)
            
            # Analyze sentiment of fetched news articles
            for article in project_news:
                article["sentiment"] = analyze_news_sentiment(article.get("description", ""))
                article["sentiment_label"] = "Positive" if article["sentiment"] > 0.05 else "Negative" if article["sentiment"] < -0.05 else "Neutral"
            
            # Store in master news dictionary
            master_news[project_name] = project_news
    
    return master_news
