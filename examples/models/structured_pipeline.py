import os
import json
import fitz
import requests
import numpy as np
from typing import List, Dict
from pathlib import Path
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType, 
    MilvusClient
)
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import SecretStr

WEBUI_API_KEY = os.getenv("WEBUI_API_KEY",'')
BASE_URL = os.getenv("BASE_URL",'')

if not WEBUI_API_KEY or not BASE_URL:
    raise ValueError('Environment Variables Missing!')

# Configuration
PDF_DIR = "pdfs/"
CHUNK_SIZE = 1000
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "pdf_chunks"

def load_pdfs_from_directory(directory: str) -> List[Path]:
    """Load all PDF files from the specified directory."""
    pdf_dir = Path(directory)
    return list(pdf_dir.glob("*.pdf"))

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_textpage().extractText()
    print("Text extraction done: {pdf_path}")
    return text

def normalize_text_with_llm(text: str) -> Dict:
    """Use LLM to normalize text into structured JSON."""
    prompt = f"""
    Please analyze the following text and convert it into JSON format with the following fields:
    - title: Extract or infer the title
    - date_published: Extract or infer the publication date
    - content: The main content
    - keywords: Extract 5-10 key topics or themes
    - summary: A brief summary of the content

    Text: {text[:2000]}...  # Truncated for prompt length
    """

        # Initialize LLM
    # llm = ChatOpenAI(
    #     base_url='http://89.117.37.210:8080/api/chat/completions',
    #     model='llama3.2:3b',
    #     api_key=SecretStr(WEBUI_API_KEY),
    #     stream_usage=False
        
    # )

        # Initialize LLM for querying
    llm = ChatOllama(
        base_url='http://localhost:11434',
        model='llama3.2:3b',
        # api_key=SecretStr(WEBUI_API_KEY),
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        print(f"normalizing text with llm : {text}")
        print(f"normalized: {response.content}")
        return json.loads(response.content if isinstance(response.content, str) else json.dumps(response.content))
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return valid JSON
        return {
            "title": "Unknown",
            "date_published": "Unknown",
            "content": text,
            "keywords": [],
            "summary": ""
        }

def chunk_text(text: Dict, chunk_size: int) -> List[Dict]:
    """Split the content into fixed-size chunks while preserving metadata."""
    content = text["content"]
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    
    chunked_docs = []
    for i, chunk in enumerate(chunks):
        print(f"chunking in progress.... {i} : {chunk}")
        chunked_docs.append({
            "chunk_id": i,
            "title": text["title"],
            "date_published": text["date_published"],
            "content": chunk,
            "keywords": text["keywords"],
            "summary": text["summary"]
        })
    return chunked_docs

def get_embeddings(text: str) -> List[float]:
    """Get embeddings using the local embeddings endpoint."""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]

def setup_milvus_collection(dim: int):
    """Set up Milvus collection with the required schema."""
    client = MilvusClient( uri="http://localhost:19530", )
    
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="my_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    
    schema = CollectionSchema(fields=fields, enable_dynamic_field=True, description="PDF chunks collection")
    
    index_params = client.prepare_index_params()

    #scalar 
    index_params.add_index(
    field_name="id",
    index_type="STL_SORT"
)

    # Create index for vector field
    index_params.add_index(
        field_name="my_vector", 
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 1024}
    )

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    client.create_index(collection_name=COLLECTION_NAME, sync=False, index_params=index_params)
    
    return client

def process_and_store_pdfs():
    """Main function to process PDFs and store in Milvus."""
    
    # Load and process PDFs
    pdf_paths = load_pdfs_from_directory(PDF_DIR)
    all_chunks = []
    
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        normalized_text = normalize_text_with_llm(text)
        chunks = chunk_text(normalized_text, CHUNK_SIZE)
        all_chunks.extend(chunks)
    
    # Get embeddings for first chunk to determine dimension
    first_embedding = get_embeddings(all_chunks[0]["content"])
    collection = setup_milvus_collection(dim=len(first_embedding)) #len(first_embedding) #768
    print('Milvus Connection established')
    
    # Process and insert chunks
    for chunk in all_chunks:
        embedding = get_embeddings(chunk["content"])
        print(f"inserting chunks in milvus: {chunk}")
        collection.insert(
            collection_name=COLLECTION_NAME,
            data = [
            {
            "chunk_id": chunk["chunk_id"],
            "title": chunk["title"],
            "content": chunk["content"],
            "my_vector": embedding
        }])
    
    collection.flush(collection_name=COLLECTION_NAME)
    print(f"Processed and stored {len(all_chunks)} chunks")
    return collection

def query_rag(query: str, collection: MilvusClient, llm: ChatOllama, top_k: int = 3):
    """Query the RAG system."""
    # Get query embedding
    query_embedding = get_embeddings(query)

    print(f"QUERY RAG SYSTEM") 
       
    # Search Milvus
    collection.load_collection(collection_name=COLLECTION_NAME)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        anns_field="my_vector",
        search_params=search_params,
        limit=top_k,
        output_fields=["content", "title"]
    )
    # print("results:", results)

    # Construct context from retrieved chunks
    context = "\n\n".join([hit['entity']['content'] for hit in results[0]])
    
    # Query LLM with context
    prompt = f"""
    Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.
    
    Context: {context}
    
    Question: {query}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# Example usage
if __name__ == "__main__":
    # Process PDFs and set up RAG
    # collection = process_and_store_pdfs()

    collection = MilvusClient( uri="http://localhost:19530", )
    
    # Initialize LLM for querying
    llm = ChatOllama(
        base_url='http://localhost:11434',
        model='llama3.2:3b',
        # api_key=SecretStr(WEBUI_API_KEY),
    )
    
    # Test query
    query = "In the marpol-practical guid, what Special Areas for sea are mentioned?"
    answer = query_rag(query, collection, llm)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")