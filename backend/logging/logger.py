"""
Central logger for the Legal RAG system.
All backend activity prints to terminal with timestamps.
"""
import logging
import sys

_FMT  = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format=_FMT,
    datefmt=_DATE,
)

# Suppress noisy third-party loggers
for _noisy in ("httpx", "httpcore", "openai", "gradio", "uvicorn", "asyncio",
               "urllib3", "filelock", "transformers", "LiteLLM", "litellm"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
