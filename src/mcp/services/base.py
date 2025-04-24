from abc import ABC, abstractmethod
from ..core.server import MCPMessage

class BaseService(ABC):
    @abstractmethod
    async def process(self, message: MCPMessage) -> MCPMessage:
        pass 