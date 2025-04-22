from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from src.config.settings import settings
from src.models.email import EmailData

class VectorStoreService:
    """Service for managing vector storage and retrieval of email content."""
    
    def __init__(self, embedding_model_name: str = settings.EMBEDDING_MODEL):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_db = None
    
    def process_emails(self, emails: List[EmailData]) -> None:
        """Process emails and store them in vector database."""
        documents = []
        
        for email in emails:
            email_text = f"""
            Subject: {email.subject}
            From: {email.sender}
            Date: {email.date}
            
            {email.body}
            """
            
            chunks = self.text_splitter.split_text(email_text)
            
            metadata = {
                'email_id': email.id,
                'subject': email.subject,
                'sender': email.sender,
                'date': email.date
            }
            
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=metadata))
        
        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=settings.VECTOR_DB_PATH
        )
        self.vector_db.persist()
    
    def load_vector_db(self) -> bool:
        """Load existing vector database."""
        if os.path.exists(settings.VECTOR_DB_PATH):
            self.vector_db = Chroma(
                persist_directory=settings.VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )
            return True
        return False
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search the vector database for relevant documents."""
        if not self.vector_db:
            raise ValueError("Vector database not initialized. Process emails first.")
        
        return self.vector_db.similarity_search(query, k=k) 