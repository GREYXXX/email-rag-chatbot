# Email LLM Chatbot with RAG
# A complete implementation for a chatbot that can use Gmail as a knowledge source
# and answer questions using various LLMs (with 1.4B models as default)

import os
import base64
import re
import json
import pickle
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Vector DB and embeddings

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLM modules
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)

# Constants
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Lighter default for testing
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_EMAILS_TO_FETCH = 1000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_DB_PATH = "./email_vectordb"

class GmailConnector:
    """Handles connection and data retrieval from Gmail."""
    
    def __init__(self, credentials_path: str = 'credentials.json'):
        self.credentials_path = credentials_path
        self.service = self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Gmail API."""
        creds = None
        token_path = 'token.pickle'
        
        # Load existing token if available
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # Refresh token if expired or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('gmail', 'v1', credentials=creds)
    
    def fetch_emails(self, max_results: int = MAX_EMAILS_TO_FETCH) -> List[Dict[str, Any]]:
        """Fetch emails from Gmail."""
        results = self.service.users().messages().list(userId='me', maxResults=max_results).execute()
        messages = results.get('messages', [])
        
        emails = []
        for message in messages:
            email_data = self.service.users().messages().get(userId='me', id=message['id']).execute()
            email_info = self._parse_email(email_data)
            emails.append(email_info)
        
        return emails
    
    def _parse_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse email data into a structured format."""
        headers = email_data['payload']['headers']
        
        # Extract header information
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
        date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
        
        # Process date if available
        try:
            parsed_date = datetime.strptime(date, '%a, %d %b %Y %H:%M:%S %z')
            date_str = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except:
            date_str = date
        
        # Get email body
        body = self._get_email_body(email_data['payload'])
        
        return {
            'id': email_data['id'],
            'thread_id': email_data['threadId'],
            'subject': subject,
            'sender': sender,
            'date': date_str,
            'body': body,
            'raw_data': email_data  # Keep raw data for potential additional processing
        }
    
    def _get_email_body(self, payload: Dict[str, Any]) -> str:
        """Extract the email body text from the payload."""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif part['mimeType'] == 'text/html':
                    if 'data' in part['body']:
                        html_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        soup = BeautifulSoup(html_content, 'html.parser')
                        return soup.get_text()
                elif 'parts' in part:
                    body = self._get_email_body(part)
                    if body:
                        return body
        elif 'body' in payload and 'data' in payload['body']:
            if payload['mimeType'] == 'text/plain':
                return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
            elif payload['mimeType'] == 'text/html':
                html_content = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text()
        
        return "No readable content found"


class EmailProcessor:
    """Processes emails and creates a vector database for retrieval."""
    
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_db = None
    
    def process_emails(self, emails: List[Dict[str, Any]]) -> None:
        """Process emails and store them in a vector database."""
        documents = []
        
        for email in emails:
            # Create a formatted text from the email
            email_text = f"""
            Subject: {email['subject']}
            From: {email['sender']}
            Date: {email['date']}
            
            {email['body']}
            """
            
            # Create document chunks
            chunks = self.text_splitter.split_text(email_text)
            
            # Create metadata for each chunk
            metadata = {
                'email_id': email['id'],
                'subject': email['subject'],
                'sender': email['sender'],
                'date': email['date']
            }
            
            # Add documents with metadata
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=metadata))
        
        # Create or update vector database
        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        self.vector_db.persist()
    
    def load_vector_db(self) -> bool:
        """Load existing vector database if available."""
        if os.path.exists(VECTOR_DB_PATH):
            self.vector_db = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )
            return True
        return False
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search the vector database for relevant documents."""
        if not self.vector_db:
            raise ValueError("Vector database not initialized. Process emails first.")
        
        return self.vector_db.similarity_search(query, k=k)


class LLMManager:
    """Manages different LLM models for RAG."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model(model_name)
    
    def load_model(self, model_name: str) -> None:
        """Load a specific LLM model with appropriate device selection."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Determine the right device
        if torch.cuda.is_available():
            device_map = "auto"
            # Configure quantization for CUDA
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            # Force CPU usage for Mac to avoid MPS issues
            device_map = "cpu"
            quantization_config = None
        
        # Load model with appropriate configuration
        try:
            if quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    trust_remote_code=True
                )
        except Exception as e:
            print(f"Error loading model with device map {device_map}: {e}")
            print("Falling back to CPU only...")
            # Fallback to CPU-only loading
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                trust_remote_code=True
            )
    
    def get_available_models(self) -> List[Dict]:
        """Return a list of recommended small LLMs for email processing."""
        return [
            {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "size": "1.1B", "description": "Small but efficient chat model"},
            {"name": "microsoft/phi-1_5", "size": "1.3B", "description": "Microsoft's Phi-1.5 small model"},
            {"name": "deepseek-ai/deepseek-coder-1.3b-base", "size": "1.3B", "description": "Code-focused small model"},
            {"name": "databricks/dolly-v2-3b", "size": "3B", "description": "Dolly instruction model (slightly larger)"},
            {"name": "meta-llama/Llama-2-7b-chat-hf", "size": "7B", "description": "Llama 2 chat model (requires approval)"}
        ]
    
    def generate_response(self, prompt: str, context: str = "", max_length: int = 1024) -> str:
        """Generate response using the loaded LLM."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized.")
        
        # Create a prompt with context
        system_message = "You are a helpful assistant that answers questions about emails."
        if context:
            full_prompt = f"Context from emails:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"{prompt}\n\nAnswer:"
        
        print(f"Full prompt: {full_prompt}")
        # Determine device - avoid MPS and use CPU if needed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Generate response
        try:
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode and clean up response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Extract only the answer part
            response = response.split("Answer:")[-1].strip()
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response. Please try again with a different question or model. Error details: {str(e)}"


class EmailChatbot:
    """Main chatbot class that integrates all components."""
    
    def __init__(self):
        self.gmail_connector = None
        self.email_processor = EmailProcessor()
        self.llm_manager = LLMManager()
        self.emails = []
        self.is_initialized = False
    
    def initialize(self, credentials_path: str = 'credentials.json') -> None:
        """Initialize the chatbot with Gmail credentials."""
        self.gmail_connector = GmailConnector(credentials_path)
        
        # Try to load existing vector DB
        if not self.email_processor.load_vector_db():
            # If no existing DB, fetch and process emails
            self.emails = self.gmail_connector.fetch_emails()
            self.email_processor.process_emails(self.emails)
        
        self.is_initialized = True
    
    def change_llm(self, model_name: str) -> None:
        """Change the LLM model used by the chatbot."""
        self.llm_manager.load_model(model_name)
    
    def answer_question(self, question: str, k: int = 5) -> str:
        """Answer a question using RAG on email content or general knowledge if not initialized."""
        # If not initialized, provide general answers but remind about Gmail connection
        if not self.is_initialized:
            # Generate response for general question without email context
            general_response = self.llm_manager.generate_response(question, context="")
            
            # Add reminder about Gmail connection
            reminder = "\n\n[Note: I've answered based on general knowledge. To get answers specific to your emails, please connect your Gmail account first.]"
            
            return general_response + reminder
        
        # If initialized, continue with RAG on email content
        # Search for relevant email content
        relevant_docs = self.email_processor.search(question, k=k)
        
        # Extract context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate response using LLM with context
        response = self.llm_manager.generate_response(question, context)
        
        return response

    # def answer_question_1(self, question: str, k: int = 5) -> str:
    #     """Answer a question using RAG on email content."""
    #     if not self.is_initialized:
    #         return "Chatbot is not initialized. Please connect Gmail first."
        
    #     # Search for relevant email content
    #     relevant_docs = self.email_processor.search(question, k=k)
        
    #     # Extract context from relevant documents
    #     context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
    #     # Generate response using LLM with context
    #     response = self.llm_manager.generate_response(question, context)
        
    #     return response
    
    def refresh_emails(self) -> None:
        """Refresh emails from Gmail and update the vector database."""
        if not self.is_initialized:
            return "Chatbot is not initialized. Please connect Gmail first."
        
        self.emails = self.gmail_connector.fetch_emails()
        self.email_processor.process_emails(self.emails)
        
        return f"Successfully refreshed and processed {len(self.emails)} emails."


def create_ui():
    st.set_page_config(page_title="Email Chatbot with LLM", layout="wide")
    
    st.title("Email LLM Chatbot")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = EmailChatbot()
        st.session_state.messages = []
        st.session_state.initialized = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Gmail connection
        st.subheader("Gmail Connection")
        credentials_file = st.file_uploader("Upload Gmail API credentials (JSON)", type=["json"])
        
        if credentials_file and not st.session_state.initialized:
            # Save the uploaded file temporarily
            with open("temp_credentials.json", "wb") as f:
                f.write(credentials_file.getbuffer())
            
            # Initialize chatbot
            with st.spinner("Connecting to Gmail and processing emails..."):
                try:
                    st.session_state.chatbot.initialize("temp_credentials.json")
                    st.session_state.initialized = True
                    st.success("Successfully connected to Gmail!")
                except Exception as e:
                    st.error(f"Error connecting to Gmail: {str(e)}")
                    st.session_state.initialized = False
        
        # Add a note if Gmail is not connected
        if not st.session_state.initialized:
            st.info("Gmail is not connected. You can still ask general questions, but email-specific questions will require Gmail connection.")
        
        # LLM Model selection
        st.subheader("LLM Model Selection")
        available_models = st.session_state.chatbot.llm_manager.get_available_models()
        model_names = [model["name"] for model in available_models]
        model_descriptions = [f"{model['name']} ({model['size']}): {model['description']}" for model in available_models]
        
        selected_model_description = st.selectbox(
            "Select LLM Model",
            model_descriptions,
            index=0
        )
        selected_model = model_names[model_descriptions.index(selected_model_description)]
        
        if st.button("Change Model"):
            with st.spinner(f"Loading model {selected_model}..."):
                try:
                    st.session_state.chatbot.change_llm(selected_model)
                    st.success(f"Model changed to {selected_model}")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        
        # Refresh emails button
        if st.button("Refresh Emails") and st.session_state.initialized:
            with st.spinner("Fetching new emails..."):
                result = st.session_state.chatbot.refresh_emails()
                st.success(result)
    
    # Display chat messages
    for message in st.session_state.messages:
        if hasattr(st, "chat_message"):
            with st.chat_message(message["role"]):
                st.write(message["content"])
        else:
            st.text(f"{message['role'].upper()}: {message['content']}")
    
    # Chat input
    prompt = st.text_input("Ask a question about your emails or anything else...")
    submit_button = st.button("Send")
    
    if submit_button and prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        if hasattr(st, "chat_message"):
            with st.chat_message("user"):
                st.write(prompt)
        else:
            st.text(f"USER: {prompt}")
        
        # Generate and display response
        if hasattr(st, "chat_message"):
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = st.session_state.chatbot.answer_question(prompt)
                    st.write(response)
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Generating response..."):
                response = st.session_state.chatbot.answer_question(prompt)
                st.write(f"ASSISTANT: {response}")
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    create_ui()
    