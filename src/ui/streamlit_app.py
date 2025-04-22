import streamlit as st
from typing import Optional
from ..services.gmail_service import GmailService
from ..services.vector_store_service import VectorStoreService
from ..services.llm_service import LLMService

class StreamlitUI:
    """Streamlit UI for the Email Chatbot."""
    
    def __init__(self):
        self.gmail_service: Optional[GmailService] = None
        self.vector_store_service = VectorStoreService()
        self.llm_service = LLMService()
        self.is_initialized = False
        
    def initialize_services(self, credentials_path: str) -> None:
        """Initialize all services."""
        self.gmail_service = GmailService(credentials_path)
        
        if not self.vector_store_service.load_vector_db():
            emails = self.gmail_service.fetch_emails()
            self.vector_store_service.process_emails(emails)
        
        self.is_initialized = True
    
    def create_ui(self):
        """Create the Streamlit UI."""
        st.set_page_config(page_title="Email Chatbot with LLM", layout="wide")
        st.title("Email LLM Chatbot")
        
        self._create_sidebar()
        self._create_chat_interface()
    
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
                        self.initialize_services("temp_credentials.json")
                        st.success("Successfully connected to Gmail!")
                    except Exception as e:
                        st.error(f"Error connecting to Gmail: {str(e)}")
            
            if not self.is_initialized:
                st.info("Gmail is not connected. You can still ask general questions.")
            
            # Model selection
            self._create_model_selector()
            
            # Refresh emails button
            if st.button("Refresh Emails") and self.is_initialized:
                with st.spinner("Fetching new emails..."):
                    emails = self.gmail_service.fetch_emails()
                    self.vector_store_service.process_emails(emails)
                    st.success(f"Successfully refreshed and processed {len(emails)} emails.")
    
    def _create_model_selector(self):
        """Create the model selection interface."""
        st.subheader("LLM Model Selection")
        available_models = self.llm_service.get_available_models()
        model_names = [model["name"] for model in available_models]
        model_descriptions = [
            f"{model['name']} ({model['size']}): {model['description']}"
            for model in available_models
        ]
        
        selected_model_description = st.selectbox(
            "Select LLM Model",
            model_descriptions,
            index=0
        )
        selected_model = model_names[model_descriptions.index(selected_model_description)]
        
        if st.button("Change Model"):
            with st.spinner(f"Loading model {selected_model}..."):
                try:
                    self.llm_service.load_model(selected_model)
                    st.success(f"Model changed to {selected_model}")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
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
                if self.is_initialized:
                    relevant_docs = self.vector_store_service.search(prompt)
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    response = self.llm_service.generate_response(prompt, context)
                else:
                    response = self.llm_service.generate_response(prompt)
                    response += "\n\n[Note: I've answered based on general knowledge. To get answers specific to your emails, please connect your Gmail account first.]"
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response}) 