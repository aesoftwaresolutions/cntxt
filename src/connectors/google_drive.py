import json
from pathlib import Path
from datetime import datetime
from .base import Connector, DocumentMeta

try:
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials as SACredentials
    from google.oauth2.credentials import Credentials as OAuth2Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io
except ImportError:
    SACredentials = None
    OAuth2Credentials = None
    InstalledAppFlow = None
    build = None
    MediaIoBaseDownload = None


class GoogleDriveConnector(Connector):
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    CHECKPOINT_FILE = '.gdrive_checkpoint.json'

    MIME_TYPES = {
        'text/plain',
        'text/markdown',
        'application/json',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.google-apps.document',
        'application/vnd.google-apps.spreadsheet',
    }

    def __init__(self, credentials_path: str):
        if build is None:
            raise ImportError("google-api-python-client is required")

        self.credentials_path = Path(credentials_path)
        self.service = self._build_service()
        self.checkpoint_path = Path(self.CHECKPOINT_FILE)

    def _build_service(self):
        creds = None
        creds_file = self.credentials_path

        if creds_file.suffix == '.json':
            try:
                creds = SACredentials.from_service_account_file(
                    str(creds_file), scopes=self.SCOPES
                )
            except (ValueError, KeyError):
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_file), scopes=self.SCOPES
                )
                creds = flow.run_local_server(port=0)
                self._save_oauth_creds(creds)

        if creds is None:
            raise ValueError(f"Could not load credentials from {creds_file}")

        return build('drive', 'v3', credentials=creds)

    def list_documents(self, folder_id: str | None = None) -> list[DocumentMeta]:
        mime_query = ' or '.join(f"mimeType='{mt}'" for mt in self.MIME_TYPES)
        query = f"({mime_query}) and trashed=false"

        if folder_id:
            query += f" and '{folder_id}' in parents"

        documents = []
        page_token = None

        while True:
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, modifiedTime, size), nextPageToken',
                pageSize=100,
                pageToken=page_token
            ).execute()

            for item in results.get('files', []):
                doc_meta = DocumentMeta(
                    id=item['id'],
                    name=item['name'],
                    path=f"gdrive://{item['id']}",
                    source='google_drive',
                    modified_at=datetime.fromisoformat(
                        item['modifiedTime'].replace('Z', '+00:00')
                    ),
                    mime_type=item['mimeType'],
                    size_bytes=int(item.get('size', 0))
                )
                documents.append(doc_meta)

            page_token = results.get('nextPageToken')
            if not page_token:
                break

        return documents

    def read_document(self, doc: DocumentMeta) -> str:
        file_id = doc.id
        mime_type = doc.mime_type

        if mime_type == 'application/vnd.google-apps.document':
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType='text/csv'
            )
        else:
            request = self.service.files().get_media(fileId=file_id)

        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)

        while True:
            _, done = downloader.next_chunk()
            if done:
                break

        content = file_content.getvalue()

        if mime_type in {'text/plain', 'text/markdown', 'application/json'}:
            return content.decode('utf-8', errors='replace')
        elif mime_type == 'text/csv':
            return content.decode('utf-8', errors='replace')
        elif mime_type == 'application/pdf':
            try:
                from pdfminer.high_level import extract_text_from_stream
                return extract_text_from_stream(file_content)
            except ImportError:
                return "[PDF extraction requires pdfminer.six]"
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            try:
                from docx import Document
                docx_doc = Document(file_content)
                return '\n'.join(para.text for para in docx_doc.paragraphs)
            except ImportError:
                return "[DOCX extraction requires python-docx]"
        else:
            return content.decode('utf-8', errors='replace')

    def get_changes_since(self, checkpoint: datetime) -> list[DocumentMeta]:
        page_token = self._load_checkpoint()

        if page_token is None:
            checkpoint_rfc = checkpoint.isoformat() + 'Z'
            query = f"modifiedTime > '{checkpoint_rfc}' and trashed=false"
        else:
            query = None

        documents = []

        try:
            if query:
                results = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields='files(id, name, mimeType, modifiedTime, size), nextPageToken',
                    pageSize=100
                ).execute()

                for item in results.get('files', []):
                    mime_type = item['mimeType']
                    if any(mt in mime_type for mt in self.MIME_TYPES):
                        doc_meta = DocumentMeta(
                            id=item['id'],
                            name=item['name'],
                            path=f"gdrive://{item['id']}",
                            source='google_drive',
                            modified_at=datetime.fromisoformat(
                                item['modifiedTime'].replace('Z', '+00:00')
                            ),
                            mime_type=mime_type,
                            size_bytes=int(item.get('size', 0))
                        )
                        documents.append(doc_meta)

                page_token = results.get('nextPageToken')
            else:
                changes_query = f"startPageToken={page_token}"
                results = self.service.changes().list(
                    pageToken=page_token,
                    spaces='drive',
                    fields='changes(file(id, name, mimeType, modifiedTime, size)), nextPageToken, newStartPageToken',
                    pageSize=100
                ).execute()

                for change in results.get('changes', []):
                    file_info = change.get('file')
                    if file_info:
                        mime_type = file_info['mimeType']
                        if any(mt in mime_type for mt in self.MIME_TYPES):
                            doc_meta = DocumentMeta(
                                id=file_info['id'],
                                name=file_info['name'],
                                path=f"gdrive://{file_info['id']}",
                                source='google_drive',
                                modified_at=datetime.fromisoformat(
                                    file_info['modifiedTime'].replace('Z', '+00:00')
                                ),
                                mime_type=mime_type,
                                size_bytes=int(file_info.get('size', 0))
                            )
                            documents.append(doc_meta)

                page_token = results.get('newStartPageToken')

            if page_token:
                self._save_checkpoint(page_token)

        except Exception as e:
            raise RuntimeError(f"Error querying Drive changes: {e}")

        return documents

    def supports_folders(self) -> bool:
        return True

    def _load_checkpoint(self) -> str | None:
        if self.checkpoint_path.exists():
            data = json.loads(self.checkpoint_path.read_text())
            return data.get('pageToken')
        return None

    def _save_checkpoint(self, page_token: str) -> None:
        data = {'pageToken': page_token}
        self.checkpoint_path.write_text(json.dumps(data))

    def _save_oauth_creds(self, creds) -> None:
        creds_file = Path('gdrive_oauth_creds.json')
        creds_data = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
        }
        creds_file.write_text(json.dumps(creds_data))
