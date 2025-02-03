def save_embeddings(embeddings, collection_name):
    from pymilvus import Collection, connections, utility

    # Connect to Milvus
    connections.connect(host='localhost', port='19530')

    # Check if the collection exists
    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist in Milvus.")

    # Create a collection instance
    collection = Collection(collection_name)

    # Insert embeddings into the collection
    ids = collection.insert(embeddings)

    # Flush the collection to make sure data is written
    collection.flush()

    return ids

def main():
    # Example usage
    embeddings = [...]  # Replace with actual embeddings data
    collection_name = "your_collection_name"
    ids = save_embeddings(embeddings, collection_name)
    print(f"Inserted embeddings with IDs: {ids}")

if __name__ == "__main__":
    main()