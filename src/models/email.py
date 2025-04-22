from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

@dataclass
class EmailData:
    """Data class for email information."""
    id: str
    thread_id: str
    subject: str
    sender: str
    date: str
    body: str
    raw_data: Dict[str, Any]

    @classmethod
    def from_api_response(cls, email_data: Dict[str, Any]) -> 'EmailData':
        """Create EmailData instance from Gmail API response."""
        headers = email_data['payload']['headers']
        
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
        date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
        
        try:
            parsed_date = datetime.strptime(date, '%a, %d %b %Y %H:%M:%S %z')
            date_str = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except:
            date_str = date
            
        return cls(
            id=email_data['id'],
            thread_id=email_data['threadId'],
            subject=subject,
            sender=sender,
            date=date_str,
            body='',  # Body will be set separately
            raw_data=email_data
        ) 