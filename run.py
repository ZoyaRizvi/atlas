"""run.py

Entry point. Run this file to launch Atlas.

SETUP:
    pip install -r requirements.txt
    # Create a .env file or set your key
    # XAI_API_KEY='xai-...'
    python3 run.py
"""

from __future__ import annotations

from pathlib import Path

from chat_window import main


def load_dotenv():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    dotenv_path = Path(__file__).parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)


if __name__ == "__main__":
    load_dotenv()
    main()
