from typing import Dict, Any, Callable
from .message import MCPMessage

class MessageRouter:
    def __init__(self):
        self.routes: Dict[str, Dict[str, Callable]] = {}
        
    def register_route(self, service_type: str, action: str, handler: Callable):
        """Register a new route handler."""
        if service_type not in self.routes:
            self.routes[service_type] = {}
        self.routes[service_type][action] = handler
        
    async def route(self, message: MCPMessage) -> MCPMessage:
        """Route a message to its handler."""
        if message.service_type not in self.routes:
            raise ValueError(f"No routes for service {message.service_type}")
        
        route_map = self.routes[message.service_type]
        if message.action not in route_map:
            raise ValueError(f"No handler for action {message.action}")
            
        return await route_map[message.action](message)

class ServiceRegistry:
    def __init__(self):
        self._services: Dict[str, Any] = {}
        
    def register(self, service_type: str, service: Any):
        """Register a new service."""
        self._services[service_type] = service
        
    def get_service(self, service_type: str) -> Any:
        """Get a registered service."""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type} not registered")
        return self._services[service_type] 