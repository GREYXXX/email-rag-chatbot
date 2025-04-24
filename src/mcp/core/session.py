from typing import Dict, Any
import time

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_active = time.time()
        self.data: Dict[str, Any] = {}

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_active = time.time()

    def set_data(self, key: str, value: Any):
        """Set session data."""
        self.data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get session data."""
        return self.data.get(key, default)

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        
    def create_session(self, session_id: str) -> Session:
        """Create a new session."""
        session = Session(session_id)
        self.sessions[session_id] = session
        return session
        
    def get_session(self, session_id: str) -> Session:
        """Get an existing session or create a new one."""
        if session_id not in self.sessions:
            return self.create_session(session_id)
        return self.sessions[session_id]
        
    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
    def cleanup_old_sessions(self, max_age: float = 3600):
        """Clean up sessions older than max_age seconds."""
        current_time = time.time()
        to_delete = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.last_active > max_age
        ]
        for session_id in to_delete:
            self.delete_session(session_id) 