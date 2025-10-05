import chromadb
from pathlib import Path
import re

class EmbeddingsStorer:
    def __init__(self):
        self.database = "./chroma_db"
        self.client = chromadb.PersistentClient(path=self.database)

    def _extract_drive_id(self, url: str) -> str:
        match = re.search(r"/d/([^/]+)/", url)
        return match.group(1) if match else url

    def store_embeddings(self, chunks_content, all_chunks, embeddings, collection_name: str="Drive_Data_Science_Docs", collection_description: str="Google Drive Data science Docs"):
        """Persistent database for embeddings storage"""
        # Building Knowledge Database with ChromaDB


        # Create collection for business documents (or get existing)
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": collection_description}
        )
        print(f"Created collection: {collection.name}") # <---
        print(f"Collection ID: {collection.id}")

        # store Document with metadata
        # Prepare metadata and add documents to collection

        metadatas = [
                        {
                            "document_id": self._extract_drive_id(chunk["source"]),
                            "document_title": chunk.get("title")
                        }
                     for chunk in all_chunks
                ]

        BATCH_SIZE = 512
        for i in range(0, len(chunks_content), BATCH_SIZE):
            batch_docs = chunks_content[i:i+BATCH_SIZE]
            batch_embeddings = embeddings[i:i+BATCH_SIZE].tolist() # Convert numpy array to list
            batch_metadatas = metadatas[i:i+BATCH_SIZE]
            batch_ids = [f"doc_{j}" for j in range(i, i+len(batch_docs))]

            collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas, # Metadata for each document
                ids=batch_ids # Unique identifiers for each document
            )

        print(f"Collection count: {collection.count()}")
