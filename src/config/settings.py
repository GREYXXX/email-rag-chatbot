from typing import List, Dict

class Settings:
    """Application configuration settings."""
    
    # Gmail API settings
    GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    MAX_EMAILS_TO_FETCH = 1000
    
    # Vector DB settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    VECTOR_DB_PATH = "./email_vectordb"
    
    # Model settings
    DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Available models configuration
    AVAILABLE_MODELS: List[Dict] = [
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "1.1B",
            "description": "Small but efficient chat model"
        },
        {
            "name": "microsoft/phi-1_5",
            "size": "1.3B",
            "description": "Microsoft's Phi-1.5 small model"
        },
        {
            "name": "deepseek-ai/deepseek-coder-1.3b-base",
            "size": "1.3B",
            "description": "Code-focused small model"
        },
        {
            "name": "databricks/dolly-v2-3b",
            "size": "3B",
            "description": "Dolly instruction model (slightly larger)"
        },
        {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "size": "7B",
            "description": "Llama 2 chat model (requires approval)"
        }
    ]

settings = Settings() 