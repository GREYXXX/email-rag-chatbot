from typing import Dict, Type, Optional
from .message import MCPMessage
from .registry import MessageRouter, ServiceRegistry
from .session import SessionManager

class MCPServer:
    def __init__(self):
        self.router = MessageRouter()
        self.session_manager = SessionManager()
        self.service_registry = ServiceRegistry()
        
    async def handle_message(self, message: MCPMessage):
        try:
            service = self.service_registry.get_service(message.service_type)
            return await service.process(message)
        except Exception as e:
            return MCPMessage.create(
                service_type="error",
                action="error_response",
                params={"error": str(e)},
                session_id=message.session_id
            ) 