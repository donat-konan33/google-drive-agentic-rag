from langchain.vectorstores import Chroma
from pathlib import Path

class EmbeddingsStorer:
    def __init__(self, all_chunks, doc_embeddings):
        self.all_chunks = all_chunks
        self.doc_embeddings = doc_embeddings

    def store_embeddings(self, all_chunks, doc_embeddings, collection_name: str="Drive", collection_description: str="Google Drive documents"):
        """Persistent database for embeddings storage"""
        # Building Knowledge Database with ChromaDB
        client = Chroma.PersistentClient(path="./chroma_db")

        # Create collection for business documents (or get existing)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": collection_description}
        )
        print(f"Created collection: {collection.name}") # <---
        print(f"Collection ID: {collection.id}")

        # store Document with metadata
        # Prepare metadata and add documents to collection
        metadatas = [{"document": Path(chunk["source"]).name} for chunk in all_chunks]
        documents = [chunk["content"] for chunk in all_chunks]

        collection.add(
            documents=documents,
            embeddings=doc_embeddings.tolist(), # Convert numpy array to list
            metadatas=metadatas, # Metadata for each document
            ids=[f"doc_{i}" for i in range(len())], # Unique identifiers for each document
        )

        print(f"Collection count: {collection.count()}")
