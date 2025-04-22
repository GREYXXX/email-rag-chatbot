import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import after path setup
import streamlit as st
from src.ui.streamlit_app import StreamlitUI

def main():
    """Main entry point for the application."""
    ui = StreamlitUI()
    ui.create_ui()

if __name__ == "__main__":
    main() 