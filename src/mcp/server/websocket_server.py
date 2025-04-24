import asyncio
import websockets
import json
from ..core.server import MCPServer
from ..core.message import MCPMessage
from typing import Dict, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.mcp_server = MCPServer()
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle individual client connections."""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    # Parse the message
                    if isinstance(message, str):
                        msg_data = json.loads(message)
                    else:
                        msg_data = json.loads(message.decode())
                    
                    mcp_message = MCPMessage(**msg_data)
                    
                    # Process the message
                    response = await self.mcp_server.handle_message(mcp_message)
                    
                    # Send response back to client
                    await websocket.send(response.to_json())
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    error_response = MCPMessage.create(
                        service_type="error",
                        action="error_response",
                        params={"error": str(e)},
                        session_id=msg_data.get("session_id", "unknown")
                    )
                    await websocket.send(error_response.to_json())
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        finally:
            self.clients.remove(websocket)

    async def start(self):
        """Start the WebSocket server."""
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            await asyncio.Future()

def run_server():
    """Run the WebSocket server."""
    server = WebSocketServer()
    asyncio.run(server.start())

if __name__ == "__main__":
    run_server() 