import os
import pickle
import base64
from typing import List, Dict, Any
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

from ..config.settings import settings
from ..models.email import EmailData
from src.utils.helpers import parse_email_body

class GmailService:
    """Service for interacting with Gmail API."""
    
    def __init__(self, credentials_path: str = 'credentials.json'):
        self.credentials_path = credentials_path
        self.service = self._authenticate()
    
    def _authenticate(self) -> Any:
        """Authenticate with Gmail API using OAuth2."""
        creds = None
        token_path = 'token.pickle'
        
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, 
                    settings.GMAIL_SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('gmail', 'v1', credentials=creds)
    
    def fetch_emails(self, max_results: int = settings.MAX_EMAILS_TO_FETCH) -> List[EmailData]:
        """Fetch emails from Gmail."""
        results = self.service.users().messages().list(
            userId='me', 
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        emails = []
        
        for message in messages:
            email_data = self.service.users().messages().get(
                userId='me', 
                id=message['id']
            ).execute()
            
            email = EmailData.from_api_response(email_data)
            email.body = self._parse_email_body(email_data['payload'])
            emails.append(email)
        
        return emails
    
    def _parse_email_body(self, payload: Dict[str, Any]) -> str:
        """Extract email body from payload."""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif part['mimeType'] == 'text/html':
                    if 'data' in part['body']:
                        html_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        return BeautifulSoup(html_content, 'html.parser').get_text()
                elif 'parts' in part:
                    body = self._parse_email_body(part)
                    if body:
                        return body
        elif 'body' in payload and 'data' in payload['body']:
            if payload['mimeType'] == 'text/plain':
                return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
            elif payload['mimeType'] == 'text/html':
                html_content = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
                return BeautifulSoup(html_content, 'html.parser').get_text()
        
        return "No readable content found" 