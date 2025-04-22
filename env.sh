#!/bin/bash

# Setup script for Email LLM Chatbot
echo "Setting up Email LLM Chatbot..."

# Create virtual environment
python -m venv email_chatbot_env
source email_chatbot_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch 
pip install transformers
pip install accelerate
pip install bitsandbytes
pip install sentence-transformers
pip install google-api-python-client
pip install google-auth-oauthlib
pip install beautifulsoup4
pip install streamlit
pip install langchain
pip install chromadb
pip install pandas
pip install numpy

echo "Dependencies installed!"
echo ""
echo "Setup Instructions:"
echo "1. Create a Google Cloud project and enable Gmail API"
echo "2. Download OAuth credentials (as 'credentials.json')"
echo "3. Run the application with: streamlit run email_chatbot.py"
echo ""
echo "Setup complete!"