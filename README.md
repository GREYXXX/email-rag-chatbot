# Email LLM Chatbot

A powerful email chatbot that uses Large Language Models (LLMs) to interact with your Gmail inbox. Built with RAG (Retrieval-Augmented Generation) architecture, this application allows you to have natural conversations about your emails while maintaining privacy and security.

## 🌟 Features

- 🔒 Secure Gmail integration with read-only access
- 🤖 Multiple LLM support with easy model switching
- 🔍 Semantic search across your emails
- 💾 Local vector database for quick retrieval
- 🚀 Streamlit-based interactive UI
- 🎯 RAG architecture for accurate responses
- 📊 Memory-efficient processing with chunking
- 🔄 Real-time email refresh capability

## 🛠 Technology Stack

- Python 3.7+
- Streamlit
- LangChain
- HuggingFace Transformers
- Google Gmail API
- ChromaDB
- Sentence Transformers
- Beautiful Soup 4

## 📋 Prerequisites

- Python 3.7 or higher
- A Google Cloud Platform account
- Gmail account
- Git

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-llm-chatbot.git
cd email-llm-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔑 Google API Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Gmail API"
   - Click "Enable"

4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Choose "Desktop application" as the application type
   - Name your OAuth client
   - Click "Create"

5. Download credentials:
   - Find your newly created OAuth 2.0 client ID
   - Click the download icon (⬇️)
   - Save the file as `credentials.json` in your project root directory

6. Configure OAuth consent screen:
   - Go to "APIs & Services" > "OAuth consent screen"
   - Choose "External" user type
   - Fill in the required information:
     - App name
     - User support email
     - Developer contact information
   - Add the following scope:
     - `https://www.googleapis.com/auth/gmail.readonly`
   - Add your test users (email addresses)

## 🏃‍♂️ Running the Application

1. Ensure your virtual environment is activated
2. Run the application:
```bash
streamlit run main.py
```

3. First-time setup:
   - The app will open in your browser
   - Click "Upload Gmail API credentials"
   - Select your `credentials.json` file
   - Follow the Google authentication flow
   - Grant the requested permissions

## 💡 Usage

1. **Initial Setup**:
   - Upload your Google credentials when prompted
   - Authenticate with your Google account
   - Wait for initial email processing

2. **Chat Interface**:
   - Type your questions about emails in the chat input
   - The bot will respond using context from your emails
   - Use natural language for queries

3. **Model Selection**:
   - Use the sidebar to switch between different LLMs
   - Available models range from 1.1B to 7B parameters
   - Choose based on your performance needs

4. **Email Refresh**:
   - Click "Refresh Emails" in the sidebar to update
   - This will fetch and process new emails