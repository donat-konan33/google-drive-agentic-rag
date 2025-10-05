"""Connect to data from Google Drive and load data contents"""
from langchain_google_community.drive import GoogleDriveLoader  # https://python.langchain.com/api_reference/google_community/drive/langchain_google_community.drive.GoogleDriveLoader.html#langchain_google_community.drive.GoogleDriveLoader
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
from pathlib import Path


service_account_key = os.getenv("GOOGLE_ACCOUNT_FILE")

# connection to GCP and then users drive according to scopes defined in GoogleCloudConsole
# Document Ingestion with MarkItDown
class ConnectDrive:
    def __init__(self):
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.file_types = ["presentation", "pdf"]
        self.SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
        self.token_path = 'token.json'


    def get_auth_token(self):
        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.SCOPES)
        # force server port to be 8506 as defined in Google Cloud Console, GoogleLoader exports everytime dynamic ports WITH PORT=0 and this behavior is not supported by config set into GCP console
        creds = flow.run_local_server(port=8506, prompt="consent")
        with open(self.token_path, "w") as token:
            token.write(creds.to_json())
        print("✅ Token saved !")

    def load_data(self):
        """Load data from Google Drive"""
        if not os.path.exists(self.token_path):
            self.get_auth_token()

        drive_loader = GoogleDriveLoader(
            #service_account_key=service_account_key, auth 1 for managing google drive of the GCP account
            credentials_path=self.credentials_path, # auth 2
            token_path='token.json',                # auth 2
            load_auth=True,
            scopes=self.SCOPES,
            folder_id='root',  # root defaults to 'My Drive'
            recursive=True,
            file_types=self.file_types,
            num_results=15
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
