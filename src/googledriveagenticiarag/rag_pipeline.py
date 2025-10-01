"""# check this way for more details https://codecut.ai/open-source-rag-pipeline-intelligent-qa-system/?utm_source=Klaviyo&utm_medium=email&utm_campaign=friday_campaign&_kx=6K-BmsU57-ERSOXiuodLkaFoXfMRecr7RohNBMowvLMa1Yj7nIfQ5vRXIAaXadVA.SgpyU4#document-ingestion-with-markitdown"""
from connect_drive import ConnectDrive
from googledriveagenticiarag.vector_store import store_embeddings
from googledriveagenticiarag.embedder import Embedder

# get data set up
connect_drive = ConnectDrive()
data = connect_drive.load_data() # get objects from Drive

def check_data_not_still_loaded(data):
    """check data not still loaded before"""
    # TODO: return only data never loaded before
    pass


# Building Knowledge Database with ChromaDB
# Enhanced Answer Generation with Open-Source LLMs
