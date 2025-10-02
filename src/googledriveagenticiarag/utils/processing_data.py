from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.vectorstores import Chroma # type: ignore
from markitdown import MarkItDown # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore # for local use of embeddings models
from typing import Tuple
import json

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings # type: ignore # whetever used providers embeddings models
import asyncio
from connect_drive import ConnectDrive
from collections import Counter

model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
try :
    connect_drive = ConnectDrive()
except Exception as e:
    print(f"Error during ConnectDrive initialization: {e}")
    connect_drive = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,   # https://www.pinecone.io/learn/chunking-strategies/ # Ils montrent que trop petit → perte de contexte, trop grand → embeddings bruités, et que 10–20% d’overlap est une bonne pratique pour garder la cohérence.
    separators=["\n\n", "\n", ". ", " ", ""] # Split hierarchy
)

class ProcessData:
    def __init__(self):
        if connect_drive is None:
            raise ValueError("ConnectDrive instance is not available.")
        self.docs = connect_drive.load_data()  # list of Document objects
        self.data_processed = "src/googledriveagenticiarag/cache/data_processed.json"

    def check_data_not_still_loaded(docs, data_processed):
        """Check data not already loaded before and return only new sources"""
        # TODO: return only data never loaded before
        sources = [doc.metadata.get("source") for doc in docs]

        try:
            with open(data_processed, "r") as f:
                file = json.load(f)
                ancient_content = file.get("sources", [])
        except (FileNotFoundError, json.JSONDecodeError):
            ancient_content = []

        # New sources not already processed
        new_sources = list(set(sources) - set(ancient_content))

        # update data processed file
        if new_sources:
            all_sources = list(set(ancient_content + new_sources))
            with open(data_processed, "w") as f:
                json.dump({"sources": all_sources}, f, indent=2)

        return new_sources


    def ensure_text(self, doc):
        md = MarkItDown()
        file_path = doc.metadata.get("source", "unknown")
        if file_path.endswith((".png", ".jpg", ".jpeg")):
            # OCR via MarkItDown
            result = md.convert(file_path)
            return result.text_content.strip() or ""
        else:
            # if text-based document
            return doc.page_content.strip() or ""


    async def process_document(self, doc, text_splitter):
        """Process a single document: chunking and embedding"""
        try:
            # Step 1: Chunking, helps to manage context length limitations of LLMs
            chunks = text_splitter.split_text(self.ensure_text(doc))

            # Step 2: Embedding and Storing in ChromaDB
            if chunks:
                return [{"content": chunk, "source": doc.metadata.get("source", "unknown")} for chunk in chunks]
            else:
                print("No chunks created from the document.")
                return None
        except Exception as e:
            print(f"Error processing document: {e}")
            return None

    async def process_documents(self, docs, text_splitter):
        tasks = []
        for doc in docs:
            tasks.append(self.process_document(doc, text_splitter))
        return await asyncio.gather(*tasks)

    def count_documents_chunks(all_chunks):
        """Count chunks or each document"""
        source_counts = Counter(chunk["source"] for chunk in all_chunks) # different sources with their frequencies
        chunk_lengths = [len(chunk["content"]) for chunk in all_chunks]

        print(f"Total chunks created: {len(all_chunks)}")
        print(f"Chunk length: {min(chunk_lengths)}-{max(chunk_lengths)} characters")

        for source, count in source_counts.items():
            print(f"-------------------------\nSource document: {Path(source).name} - Chunks: {count}\n-------------------")

    # create embeddings with documents chunks
    def create_chunks_embeddings(all_chunks):
        """create documents chunks embeddings"""
        documents = [chunk["content"] for chunk in all_chunks]
        model
        try:
            embeddings = model.encode(documents, show_progress_bar=True, batch_size=8)
            print(f"Embedding generation results:")
            print(f"  - Embeddings shape: {embeddings.shape}")
            print(f"  - Vector dimensions: {embeddings.shape[1]}")
            return embeddings
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return None

## OpenAI Embeddings (API)
# from langchain.embeddings import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # learn more

# Creating Searchable Embeddings
# RAG systems need to understand text meaning, not just match keywords. SentenceTransformers converts your text into numerical vectors that capture semantic relationships, allowing the system to find truly relevant information even when exact words don’t match.

# Generate each chunk Embedding

######## tester Vertex AI LLM and Embeddings
# from langchain_community.embeddings import VertexAIEmbeddings
# embeddings = VertexAIEmbeddings()
# async def embed_chunk(chunk, model_name=):
#
#2️⃣ Vertex AI (Text Embeddings)
#
#✅ Vertex AI propose des modèles LLM et embeddings via Text Embeddings API.
#
#❌ Mais chaque requête est payante, même pour le Free Tier.
#
# Le crédit initial te permet de tester gratuitement, mais dès qu’il est épuisé, tu dois payer.
