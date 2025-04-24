import asyncio
import websockets
import uuid
import json
from typing import Callable, Dict, Optional
from ..core.message import MCPMessage

class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session_id = str(uuid.uuid4())
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.callbacks: Dict[str, Callable] = {}
        
    async def connect(self):
        self.websocket = await websockets.connect(self.server_url)
        asyncio.create_task(self._listen())
        
    async def _listen(self):
        while True:
            try:
                message = await self.websocket.recv()
                await self._handle_message(message)
            except websockets.ConnectionClosed:
                break
                
    async def send_message(self, service_type: str, action: str, params: dict) -> MCPMessage:
        """Send a message to the server."""
        message = MCPMessage.create(
            service_type=service_type,
            action=action,
            params=params,
            session_id=self.session_id
        )
        await self.websocket.send(message.to_json())  # Using to_json() method
        return message
        
    async def _handle_message(self, message_data: str):
        """Handle incoming messages."""
        try:
            message = MCPMessage.from_json(message_data)
            if message.service_type in self.callbacks:
                await self.callbacks[message.service_type](message)
        except Exception as e:
            print(f"Error handling message: {e}")
    
    def on_message(self, service_type: str, callback: Callable):
        """Register a callback for a specific message type."""
        self.callbacks[service_type] = callback 