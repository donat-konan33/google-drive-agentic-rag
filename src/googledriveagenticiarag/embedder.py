from src.googledriveagenticiarag.utils.processing_data import ProcessData


import asyncio
# process data and Intelligent Chunking with LangChain
class Embedder:
    def __init__(self):
        """"""
        self.process_data = ProcessData()

    def get_all_chunks(self, docs_chunks):
        """get all chunk from data"""
        try :
            if docs_chunks is not None:
                all_chunks = [chunk for chunk_list in docs_chunks for chunk in chunk_list]
                self.process_data.count_documents_chunks(all_chunks)
                return all_chunks
        except Exception as e:
            print(f"Error during document processing: {e}")
            return None

    # Creating Searchable Embeddings with SentenceTransformers
    # Load Q&A-optimized embedding model (downloads automatically on first use)

    def local_embeddings(self, docs_chunks):
        """get documents embeddings"""
        try :
            all_chunks = self.get_all_chunks(docs_chunks)
            if all_chunks is not None:
                return self.process_data.create_chunks_embeddings(all_chunks=all_chunks)
        except Exception as e:
            print(f"Error during embeddings creation: {e}")
            return None

    # provider embedding
    def huggingface_embeddings(self, docs_chunks):
        """get embeddings drom huggingface api"""
        pass


    def openai_embeddings(self, docs_chunks):
        """get embeddings from OpenAI-based models"""
        pass


    def google_embeddings(self, docs_chunks):
        """Get embeddings from gemini models"""
        pass
