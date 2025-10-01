"""Connect to data from Google Drive and load data contents"""
from langchain_google_community.drive import GoogleDriveLoader  # https://python.langchain.com/api_reference/google_community/drive/langchain_google_community.drive.GoogleDriveLoader.html#langchain_google_community.drive.GoogleDriveLoader
from langchain import Document
import os

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


credentials_path = os.getenv("GCP_CREDENTIAL")
token_path = os.getenv("TOKEN_PATH")

# connection to GCP and then users drive according to scopes defined in GoogleCloudConsole
# Document Ingestion with MarkItDown
class ConnectDrive:
    def __init__(self):
        self.credentials_path = credentials_path
        self.file_types = [".md", ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", "csv"]
        self.SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

    def load_data(self):
        """Load data from Google Drive"""
        drive_loader = GoogleDriveLoader(
            credentials_path=self.credentials_path,
            token_path=token_path,
            load_auth=True,
            scopes=self.SCOPES,
            folder_id=None,  # None defaults to 'My Drive'
            recursive=True,
            file_types=self.file_types,
        )
        try:
            print("Loading data from Google Drive...")
            docs = drive_loader.load()
            print("Data loaded successfully.")
            return docs # return list of Document objects
        except Exception as e:
            print(f"Error loading data from Google Drive: {e}")
            return None

#    def get_credentials(): # manual flow to get token
#        creds = None
#        if os.path.exists('token.pickle'):
#            with open('token.pickle', 'rb') as token:
#                creds = pickle.load(token)
#        if not creds or not creds.valid:
#            if creds and creds.expired and creds.refresh_token:
#                creds.refresh(Request())
#            else:
#                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
#                creds = flow.run_local_server(port=0)
#            with open('token.pickle', 'wb') as token:
#                pickle.dump(creds, token)
#        return creds
