from googledriveagenticiarag.utils.processing_data import ProcessData, text_splitter


import asyncio
# process data and Intelligent Chunking with LangChain
class Embedder:
    def __init__(self, data):
        self.docs_chunks = asyncio.run(ProcessData.process_documents(data, text_splitter))

    def get_all_chunks(self, data):
        """get all chunk from data"""
        try :
            docs_chunks = self.docs_chunk
            if docs_chunks is not None:
                all_chunks = [chunk for chunk_list in docs_chunks for chunk in chunk_list]
                ProcessData.count_documents_chunks(all_chunks)
                return all_chunks
        except Exception as e:
            print(f"Error during document processing: {e}")
            return None

    # Creating Searchable Embeddings with SentenceTransformers
    # Load Q&A-optimized embedding model (downloads automatically on first use)
    @classmethod
    def local_embeddings(self, data):
        """get documents embeddings"""
        try :
            all_chunks = self.get_all_chunks(data)
            if all_chunks is not None:
                return ProcessData.create_chunks_embeddings(all_chunks=all_chunks)
        except Exception as e:
            print(f"Error during embeddings creation: {e}")
            return None
    @classmethod
    def huggingface_embeddings(cls, data):
        """get embeddings drom huggingface api"""
        pass

    @classmethod
    def openai_embeddings(cls, data):
        """get embeddings from OpenAI-based models"""
        pass

    @classmethod
    def google_embeddings(cls, data):
        """Get embeddings from gemini models"""
        pass
