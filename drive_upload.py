from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OAUTH_CLIENT_FILE = os.path.join(BASE_DIR, "oauth_credentials.json")
TOKEN_FILE = os.path.join(BASE_DIR, "token.json")
FOLDER_ID = "1g5RCR7pVpNAUdJAwxgPzmhUYRQxkRmNS"
SCOPES = ["https://www.googleapis.com/auth/drive"]

def get_drive_credentials():
    credentials = None
    if os.path.exists(TOKEN_FILE):
        credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    elif not credentials or not credentials.valid:
        if not os.path.exists(OAUTH_CLIENT_FILE):
            raise RuntimeError(
                f"Missing {OAUTH_CLIENT_FILE}. Download an OAuth 2.0 Desktop app "
                "client from Google Cloud Console and save it with this name."
            )
        flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CLIENT_FILE, SCOPES)
        credentials = flow.run_local_server(port=0)

    with open(TOKEN_FILE, "w") as token:
        token.write(credentials.to_json())
    return credentials

def upload_to_drive(file_path):
    creds = get_drive_credentials()
    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        "name": os.path.basename(file_path),
        "parents": [FOLDER_ID]
    }
    media = MediaFileUpload(file_path, resumable=True)

    result = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()

    file_id = result["id"]

    # Make this file downloadable by anyone with the link
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
        supportsAllDrives=True,
    ).execute()

    # Re-fetch now that the permission is actually set
    file = service.files().get(
        fileId=file_id,
        fields="webContentLink",
        supportsAllDrives=True,
    ).execute()

    return file["webContentLink"]