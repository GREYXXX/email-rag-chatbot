from .base import BaseService
from ..core.server import MCPMessage
from ...services.gmail_service import GmailService as OriginalGmailService

class EmailService(BaseService):
    def __init__(self):
        self.gmail_service = OriginalGmailService()
        self.sessions: Dict[str, Dict] = {}
        
    async def process(self, message: MCPMessage) -> MCPMessage:
        action_handlers = {
            "initialize": self.initialize_session,
            "fetch_emails": self.fetch_emails,
            "refresh_emails": self.refresh_emails
        }
        
        handler = action_handlers.get(message.action)
        if not handler:
            raise ValueError(f"Unknown action: {message.action}")
            
        return await handler(message)
        
    async def initialize_session(self, message: MCPMessage) -> MCPMessage:
        credentials_path = message.params.get("credentials_path")
        session_id = message.session_id
        
        self.sessions[session_id] = {
            "gmail_service": OriginalGmailService(credentials_path)
        }
        
        return MCPMessage.create(
            service_type="email",
            action="initialized",
            params={"status": "success"},
            session_id=session_id
        ) 