from pathlib import Path
from langchain.vectorstores import Chroma # type: ignore
#from markitdown import MarkItDown # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore # for local use of embeddings models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings # type: ignore # whetever used providers embeddings models
from collections import Counter, OrderedDict
from typing import Iterable, Optional, List, Tuple
from langchain.schema import Document
import asyncio
import json

# get model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

class ProcessData:
    def __init__(self):
        #self.docs = connect_drive.load_data()  # list of Document objects : optimization Class method calling by getting data outside
        self.data_processed = "src/googledriveagenticiarag/cache/data_processed.json"


    @staticmethod
    def merge_docs_by_source(docs: Iterable[Document], filter_sources: Optional[Iterable[str]] = None) -> OrderedDict:
         """
         group and concat different pages by source.

         Args:
             docs: iterable of langchain.schema.Document
             filter_sources: iterable (optional)
         Returns:
             OrderedDict where each key is a source and value a dict {"title", "content"}.
         """
         grouped = OrderedDict()

         for doc in docs:
             src = (doc.metadata or {}).get("source") or "unknown"
             if filter_sources is not None and src not in filter_sources:
                 continue

             title = (doc.metadata or {}).get("title") or "untitled"
             text = (doc.page_content or "") + "\n\n"   # séparation entre pages

             if src not in grouped:
                 grouped[src] = {"title": title, "content": text}
             else:
                 grouped[src]["content"] += text

         return grouped


    @staticmethod
    def check_data_never_loaded(data_processed, sources: Counter) -> List[str]:
        """Check data not already loaded before and return only new sources"""
        # TODO: return only data never loaded before
        sources = list(sources.keys())#[doc.metadata.get("source") for doc in docs]

        try:
            with open(data_processed, "r") as f:
                file = json.load(f)
                ancient_sources = file.get("sources", [])
        except (FileNotFoundError, json.JSONDecodeError):
            ancient_sources = []

        # return New sources not already processed
        return list(set(sources) - set(ancient_sources))


    # strurcture docs coming from drive
    def get_document(self, docs: List[Document]):
        """"""
        all_source = Counter(doc.metadata.get("source", "unknown") for doc in docs)
        new_source = self.check_data_never_loaded(self.data_processed, all_source)
        documents = self.merge_docs_by_source(docs, new_source)

        return documents


    async def process_document(self, documents: OrderedDict, source:str, text_splitter):
        """Process a single document: chunking and embedding"""
        try:
            # Step 1: Chunking, helps to manage context length limitations of LLMs
            chunks = text_splitter.split_text(documents.get(source)["content"])

            # Step 2: Embedding and Storing in ChromaDB
            if chunks:
                return [{"source": source, "title": documents.get(source)["title"], "content": chunk }
                        for chunk in chunks]
            else:
                print("No chunks created from the document.")
                return None
        except Exception as e:
            print(f"Error processing document: {e}")
            return None


    async def process_documents(self, documents: OrderedDict, text_splitter):
        """parallelize chunking tasks"""
        tasks = [self.process_document(documents, source, text_splitter) for source in list(documents)]
        return await asyncio.gather(*tasks)


    # Check out our chunkings stats
    def count_documents_chunks(self, all_chunks):
        """Count chunks or each document"""
        source_counts = Counter(chunk["source"] for chunk in all_chunks) # different sources with their frequencies
        chunk_lengths = [len(chunk["content"]) for chunk in all_chunks]

        print(f"Total chunks created: {len(all_chunks)}")
        print(f"Chunk length: {min(chunk_lengths)}-{max(chunk_lengths)} characters")

        for source, count in source_counts.items():
            print(f"-------------------------\nSource document: {source} - Chunks: {count}\n-------------------")


    def _clean_text(self, text: str) -> str:
        """Normalize content"""
        import unicodedata
        if not isinstance(text, str):
            return ""
        # delete non-printing characters or invalid
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
        # Normalize
        text = unicodedata.normalize("NFKC", text)
        return text.strip()

    # create embeddings with documents chunks
    def create_chunks_embeddings(self, all_chunks) -> Tuple[List]:
        """create documents chunks embeddings"""
        all_chunks_doc = [self._clean_text(chunk["content"]) for chunk in all_chunks if chunk.get("content") and chunk["content"].strip()]

        try:
            embeddings = model.encode(all_chunks_doc, show_progress_bar=True, batch_size=8)
            print(f"Embedding generation results:")
            print(f"  - Embeddings shape: {embeddings.shape}")
            print(f"  - Vector dimensions: {embeddings.shape[1]}")
            return all_chunks_doc, all_chunks, embeddings
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return None
