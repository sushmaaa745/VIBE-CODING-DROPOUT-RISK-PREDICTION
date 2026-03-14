"""Streamlit entrypoint for deployment.

This file is kept at the project root so Streamlit Cloud / Streamlit Sharing can detect and run it easily.
"""

from app.app import main

if __name__ == "__main__":
    main()
