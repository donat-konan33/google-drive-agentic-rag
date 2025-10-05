"""# check this way for more details https://codecut.ai/open-source-rag-pipeline-intelligent-qa-system/?utm_source=Klaviyo&utm_medium=email&utm_campaign=friday_campaign&_kx=6K-BmsU57-ERSOXiuodLkaFoXfMRecr7RohNBMowvLMa1Yj7nIfQ5vRXIAaXadVA.SgpyU4#document-ingestion-with-markitdown"""

from src.googledriveagenticiarag.embedder import Embedder
import json
from src.googledriveagenticiarag.utils.processing_data import ProcessData
from src.googledriveagenticiarag.connect_drive import ConnectDrive
from src.googledriveagenticiarag.vector_store import EmbeddingsStorer
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore


# get data set up
# TODO
# get connection to ggogle drive and get data

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,   # https://www.pinecone.io/learn/chunking-strategies/ # too short → context loss, too high → embeddings with noise,  overlap around 10–20%  is better practics  to keep  consistency
    separators=["\n\n", "\n", ". ", " ", ""] # Split hierarchy
)

def get_drive_data():
    """get data from google drive"""
    try :
        connect_drive = ConnectDrive()
        if connect_drive is None:
            raise ValueError("ConnectDrive instance is not available.")
        return connect_drive.load_data()
    except Exception as e:
        print(f"Error during ConnectDrive initialization: {e}")

def store_source_processed(process_data):
    """"""
        # update data processed file
    data_processed = "src/googledriveagenticiarag/cache/data_processed.json"
    new_sources = process_data.check_data_never_loaded()
    with open(data_processed, "r") as f:
        file = json.load(f)
        ancient_sources = file.get("sources", [])

    if new_sources:
        all_sources = list(set(ancient_sources + new_sources))
        with open(data_processed, "w") as f:
            json.dump({"sources": all_sources}, f, indent=2)

def run_pipeline(text_splitter=text_splitter):
    """Launch rag pipeline"""

    # Get data
    data = get_drive_data()
    if not data:
        print("No data loader")
        return
    # instantiation of data processing tools
    process_data = ProcessData()
    documents = process_data.get_document(data)
    docs_chunks = asyncio.run(process_data.process_documents(documents, text_splitter)) # chunk docs (list of lists of chunks dicts)

    # Get all chunk content and their embeddings
    # Instantiate embedder
    embedder = Embedder()
    chunks_content, all_chunks, embeddings = embedder.local_embeddings(docs_chunks)

    # Store data in a Vectors Database like Chromadb
    # instantiate storer
    vector_storer = EmbeddingsStorer()
    vector_storer.store_embeddings(chunks_content, all_chunks, embeddings)

    store_source_processed(process_data)

    print("Pipeline completed successfully.")
    # Enhanced Answer Generation with Open-Source LLMs

if __name__ == "__main__":
    run_pipeline()
