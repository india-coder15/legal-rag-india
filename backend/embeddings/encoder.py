"""
Embedding encoder — loads BAAI/bge-small-en-v1.5 once as a singleton.
Provides encode_texts() for documents and encode_query() for queries.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import EMBEDDING_MODEL, EMBEDDING_DIM

_model = None  # singleton


def get_model():
    """Lazy-load the embedding model. ~130MB, loaded once per session."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode_texts(texts: list, batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of document texts into normalized float32 vectors.
    Returns ndarray of shape (N, EMBEDDING_DIM).
    Uses 'Represent this sentence: ' instruction prefix for bge model.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = get_model()
    prefixed = [f"Represent this sentence: {t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single search query into a normalized float32 vector.
    Returns ndarray of shape (1, EMBEDDING_DIM).
    Uses 'Represent this sentence for searching: ' prefix for bge model.
    """
    model = get_model()
    prefixed = f"Represent this sentence for searching: {query}"
    vec = model.encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.astype(np.float32)
