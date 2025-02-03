def generate_embeddings(json_data):
    import requests

    url = "http://localhost:11434/api/embeddings"
    headers = {"Content-Type": "application/json"}
    
    embeddings = []
    
    for item in json_data:
        prompt = json_data  # Assuming 'content' is the field to generate embeddings from
        body = {
            "model": "nomic-embed-text",
            "prompt": prompt
        }
        
        response = requests.post(url, json=body, headers=headers)
        
        if response.status_code == 200:
            embeddings.append(response.json())
        else:
            print(f"Error generating embeddings  {response.text}")
    
    return embeddings

def save_embeddings_to_milvus(embeddings):
    from pymilvus import Collection

    collection = Collection("your_collection_name")  # Replace with your collection name

    for embedding in embeddings:
        # Assuming embedding contains the necessary fields to save
        collection.insert(embedding)  # Adjust according to your embedding structure

# Example usage
# json_data = [...]  # Load your JSON data here
# embeddings = generate_embeddings(json_data)
# save_embeddings_to_milvus(embeddings)