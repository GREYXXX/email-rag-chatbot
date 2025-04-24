import streamlit as st
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from ..mcp.client.sdk import MCPClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitUI:
    """Streamlit UI for the Email Chatbot."""
    
    def __init__(self):
        self.client = MCPClient("ws://localhost:8765")
        self.is_initialized = False
        self.executor = ThreadPoolExecutor()
        self.is_server_connected = False
        
    def initialize(self):
        """Initialize the UI synchronously."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_initialize())
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            st.error("Failed to connect to server. Please ensure the server is running.")
            if st.button("Retry Connection"):
                self.initialize()
    
    async def _async_initialize(self):
        """Async initialization with retry logic."""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                await self.client.connect()
                self.is_server_connected = True
                return
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise ConnectionError("Failed to connect to server after multiple attempts")
    
    def create_ui(self):
        """Create the Streamlit UI."""
        st.set_page_config(page_title="Email Chatbot with LLM", layout="wide")
        st.title("Email LLM Chatbot")
        
        # Server status indicator
        with st.sidebar:
            if not self.is_server_connected:
                st.warning("⚠️ Server disconnected")
                if st.button("Connect to Server"):
                    self.initialize()
            else:
                st.success("✅ Server connected")
        
        if not self.is_initialized:
            st.info("Attempting to connect to server...")
            self.initialize()
            
        if self.is_server_connected:
            self._create_sidebar()
            self._create_chat_interface()
        else:
            st.error("""
                Server connection failed. Please:
                1. Ensure the server is running (python server.py)
                2. Click 'Connect to Server' to retry
                """)
    
    def _create_sidebar(self):
        """Create the sidebar with configuration options."""
        with st.sidebar:
            st.header("Configuration")
            
            # Gmail connection
            st.subheader("Gmail Connection")
            credentials_file = st.file_uploader(
                "Upload Gmail API credentials (JSON)", 
                type=["json"]
            )
            
            if credentials_file and not self.is_initialized:
                with open("temp_credentials.json", "wb") as f:
                    f.write(credentials_file.getbuffer())
                
                with st.spinner("Connecting to Gmail and processing emails..."):
                    try:
                        self._handle_credentials("temp_credentials.json")
                        st.success("Successfully connected to Gmail!")
                    except Exception as e:
                        st.error(f"Error connecting to Gmail: {str(e)}")
            
            if not self.is_initialized:
                st.info("Gmail is not connected. You can still ask general questions.")
            
            # Model selection
            self._create_model_selector()
    
    def _create_model_selector(self):
        """Create the model selection interface."""
        st.subheader("LLM Model Selection")
        
        # Get available models
        response = self._run_async(self.client.send_message(
            "llm",
            "list_models",
            {}
        ))
        
        # Extract models from response
        models_data = response.get_response_data()
        model_names = models_data.get("models", [])
        
        if not model_names:
            st.warning("No models available")
            return
            
        selected_model = st.selectbox(
            "Select Model",
            [model["name"] for model in model_names]
        )
        
        # Show API key input for cloud models
        if selected_model in ["gpt-4", "claude-3"]:
            api_key = st.text_input("API Key", type="password")
        else:
            api_key = None
        
        if st.button("Change Model"):
            with st.spinner(f"Loading model {selected_model}..."):
                try:
                    self._handle_model_switch(selected_model, api_key)
                    st.success(f"Model changed to {selected_model}")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    def _handle_model_switch(self, model_name: str, api_key: Optional[str] = None):
        """Handle model switching synchronously."""
        params = {"model_name": model_name}
        if api_key:
            params["api_key"] = api_key
        
        response = self._run_async(self.client.send_message(
            "llm",
            "switch_model",
            params
        ))
        return response.get_response_data()
    
    def _handle_credentials(self, credentials_path: str):
        """Handle credentials synchronously."""
        self._run_async(self.client.send_message(
            "email",
            "initialize",
            {"credentials_path": credentials_path}
        ))
    
    def _run_async(self, coroutine):
        """Run an async function synchronously."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coroutine)
    
    def _create_chat_interface(self):
        """Create the chat interface."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if prompt := st.chat_input("Ask a question about your emails or anything else..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response = self._handle_chat_message(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    def _handle_chat_message(self, prompt: str) -> str:
        """Handle chat messages synchronously."""
        response = self._run_async(self.client.send_message(
            "llm",
            "generate_response",
            {"prompt": prompt}
        ))
        return response.get_response_data().get("response", "Error generating response") 