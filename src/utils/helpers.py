from typing import Dict, Any
from bs4 import BeautifulSoup
import base64

def parse_email_body(payload: Dict[str, Any]) -> str:
    """Parse email body from payload."""
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
                body = parse_email_body(part)
                if body:
                    return body
    elif 'body' in payload and 'data' in payload['body']:
        if payload['mimeType'] == 'text/plain':
            return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        elif payload['mimeType'] == 'text/html':
            html_content = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
            return BeautifulSoup(html_content, 'html.parser').get_text()
    
    return "No readable content found" 