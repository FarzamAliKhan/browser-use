def test_embeddings_generator():
    import requests
    import json

    # Sample input data
    sample_json = {
        "title": "Sample PDF",
        "date_published": "2023-01-01",
        "content": "This is a sample content extracted from a PDF.",
        "keywords": ["sample", "pdf", "test"],
        "summary": "This is a summary of the sample PDF."
    }

    # Endpoint for generating embeddings
    endpoint = "http://localhost:11434/api/embeddings"
    request_body = {
        "model": "nomic-embed-text",
        "prompt": sample_json["content"]
    }

    # Make the request to generate embeddings
    response = requests.post(endpoint, json=request_body)
    
    # Check if the response is successful
    assert response.status_code == 200, "Failed to generate embeddings"
    
    # Parse the response
    embeddings = response.json()
    
    # Validate the structure of the embeddings
    assert "embeddings" in embeddings, "Embeddings not found in response"
    assert isinstance(embeddings["embeddings"], list), "Embeddings should be a list"
    assert len(embeddings["embeddings"]) > 0, "Embeddings list should not be empty"