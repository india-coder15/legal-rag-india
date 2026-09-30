"""
Entry point: starts the Legal Research Assistant UI.

Usage:
    python run.py              # local + public shareable link
    python run.py --no-share   # local only (http://localhost:7860)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backend.llm.mistral_client import is_live
from app.gradio_app import launch

if __name__ == "__main__":
    share  = "--no-share" not in sys.argv
    status = "ONLINE" if is_live() else "OFFLINE (mock mode)"
    print(f"LLM endpoint: {status}")
    print("Starting Legal Research Assistant...")
    launch(share=share)
