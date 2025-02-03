import os

from pipeline.pdf_processor import pick_pdfs, extract_text
from pipeline.text_processor import preprocess_text, normalize_to_json
from pipeline.embeddings_generator import generate_embeddings
from pipeline.vector_store import save_embeddings

WEBUI_API_KEY = os.getenv("WEBUI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

def main():
    pdf_folder = 'data/pdfs'
    processed_folder = 'data/processed'
    embeddings_folder = 'data/embeddings'

    # Step 1: Pick PDFs from the specified folder
    pdf_files = pick_pdfs(pdf_folder)

    for pdf_file in pdf_files:
        # Step 2: Extract text from the PDF
        text = extract_text(pdf_file)

        # Step 3: Preprocess and normalize the text
        json_data = normalize_to_json(text)

        # Step 4: Generate embeddings
        embeddings = generate_embeddings(json_data)

        # Step 5: Save embeddings to the vector store
        save_embeddings(embeddings, embeddings_folder)

    # Test run: Query the LLM model with a sample prompt
    sample_query = "What is the main topic of the processed PDFs?"
    response = query_llm(sample_query)
    print("LLM Response:", response)


def query_llm(prompt):
    import requests
    response = requests.post(
        f"{BASE_URL}/api/chat/completions",
        headers={"Authorization": f"Bearer {WEBUI_API_KEY}"},
        json={"prompt": prompt}
    )
    return response.json()

if __name__ == "__main__":
    main()