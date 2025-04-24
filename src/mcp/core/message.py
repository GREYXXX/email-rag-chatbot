from dataclasses import dataclass, asdict
import time
import uuid
import json
from typing import Dict, Any

@dataclass
class MCPMessage:
    service_type: str
    action: str
    params: Dict[str, Any]
    session_id: str
    timestamp: float = time.time()
    message_id: str = str(uuid.uuid4())
    
    @classmethod
    def create(cls, service_type: str, action: str, params: dict, session_id: str):
        return cls(
            service_type=service_type,
            action=action,
            params=params,
            session_id=session_id
        )

    def to_json(self) -> str:
        """Convert message to JSON string."""
        data = {
            "service_type": self.service_type,
            "action": self.action,
            "params": self.params,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def get_response_data(self) -> Dict[str, Any]:
        """Get the response data from params."""
        return self.params 