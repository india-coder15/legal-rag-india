"""
FAISS index management for laws and sections.

Two indices:
  - laws.index     : one vector per law (8000 at scale)
  - sections.index : one vector per section (100K+ at scale)

Both use IndexFlatIP (exact inner-product / cosine search since vectors are normalized).
Saved to database/faiss/ alongside JSON ID mapping files.

Parallel batch search: splits the full index into batches, searches each in a
ThreadPoolExecutor, merges results — supports growing database without full rebuild.
"""
import sys
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import FAISS_DIR, EMBEDDING_DIM

# Singleton caches
_law_index    = None
_law_ids      = None
_sec_index    = None
_sec_ids      = None


# ── Build & Save ───────────────────────────────────────────────────────────────

def build_law_index(law_uids: list, embeddings: np.ndarray):
    """Build and save FAISS law index from (N, 384) embeddings array."""
    import faiss
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_DIR / "laws.index"))
    with open(FAISS_DIR / "laws_ids.json", "w") as f:
        json.dump(law_uids, f)
    print(f"[faiss] Law index saved — {len(law_uids)} vectors")


def build_section_index(section_uids: list, embeddings: np.ndarray):
    """Build and save FAISS section index from (N, 384) embeddings array."""
    import faiss
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(section_uids)

    if n > 10_000:
        # IVF for large sets: approximate search, much faster at 100K+
        nlist  = min(256, n // 10)
        quant  = faiss.IndexFlatIP(EMBEDDING_DIM)
        index  = faiss.IndexIVFFlat(quant, EMBEDDING_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = 32   # probe 32 clusters per query (accuracy vs speed)
    else:
        # Exact search for smaller sets
        index = faiss.IndexFlatIP(EMBEDDING_DIM)

    index.add(embeddings)
    faiss.write_index(index, str(FAISS_DIR / "sections.index"))
    with open(FAISS_DIR / "sections_ids.json", "w") as f:
        json.dump(section_uids, f)
    print(f"[faiss] Section index saved — {n} vectors")


# ── Load (singleton) ───────────────────────────────────────────────────────────

def load_law_index():
    """Load law index + IDs. Cached after first call."""
    global _law_index, _law_ids
    if _law_index is not None:
        return _law_index, _law_ids
    import faiss
    idx_path = FAISS_DIR / "laws.index"
    ids_path = FAISS_DIR / "laws_ids.json"
    if not idx_path.exists() or not ids_path.exists():
        return None, []
    _law_index = faiss.read_index(str(idx_path))
    with open(ids_path) as f:
        _law_ids = json.load(f)
    return _law_index, _law_ids


def load_section_index():
    """Load section index + IDs. Cached after first call."""
    global _sec_index, _sec_ids
    if _sec_index is not None:
        return _sec_index, _sec_ids
    import faiss
    idx_path = FAISS_DIR / "sections.index"
    ids_path = FAISS_DIR / "sections_ids.json"
    if not idx_path.exists() or not ids_path.exists():
        return None, []
    _sec_index = faiss.read_index(str(idx_path))
    with open(ids_path) as f:
        _sec_ids = json.load(f)
    return _sec_index, _sec_ids


# ── Parallel Batch Search ──────────────────────────────────────────────────────

def _search_batch(index, ids: list, query_vec: np.ndarray,
                  batch_start: int, batch_end: int, top_k: int) -> list:
    """
    Search a slice of the index [batch_start:batch_end].
    Returns list of (uid, score) pairs.
    """
    import faiss
    batch_ids  = ids[batch_start:batch_end]
    batch_size = len(batch_ids)
    if batch_size == 0:
        return []

    # Reconstruct vectors for this batch slice and search in-memory sub-index
    sub_vecs = np.zeros((batch_size, EMBEDDING_DIM), dtype=np.float32)
    for i in range(batch_size):
        sub_vecs[i] = index.reconstruct(batch_start + i)

    sub_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    sub_index.add(sub_vecs)

    k       = min(top_k, batch_size)
    scores, positions = sub_index.search(query_vec, k)
    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos >= 0 and score > 0:
            results.append((batch_ids[pos], float(score)))
    return results


def search_laws(query_vec: np.ndarray, top_k: int = 20,
                batch_size: int = 50, max_workers: int = 4) -> list:
    """
    Search all law vectors in parallel batches.
    Returns list of (law_uid, similarity_score) sorted by score desc.

    All batches run concurrently via ThreadPoolExecutor.
    Results from all batches are merged and top_k returned.
    """
    index, ids = load_law_index()
    if index is None or not ids:
        return []

    total   = len(ids)
    batches = [(i, min(i + batch_size, total))
               for i in range(0, total, batch_size)]

    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_search_batch, index, ids, query_vec, start, end, top_k): (start, end)
            for start, end in batches
        }
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception:
                pass

    # Merge: keep best score per uid, sort, return top_k
    best = {}
    for uid, score in all_results:
        if uid not in best or score > best[uid]:
            best[uid] = score
    sorted_results = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def search_sections(query_vec: np.ndarray, top_k: int = 30,
                    law_uids: list = None,
                    batch_size: int = 200, max_workers: int = 4) -> list:
    """
    Search all section vectors in parallel batches.
    Optionally filter results to specific law_uids (post-filter after search).
    Returns list of (section_uid, similarity_score) sorted by score desc.
    """
    index, ids = load_section_index()
    if index is None or not ids:
        return []

    # For IVF index we can search directly (no reconstruction needed)
    try:
        k = min(top_k * 5, len(ids))  # oversample to allow filtering
        scores, positions = index.search(query_vec, k)
        results = [
            (ids[pos], float(score))
            for score, pos in zip(scores[0], positions[0])
            if pos >= 0 and score > 0
        ]
    except Exception:
        # Fallback to batch search if direct search fails
        total   = len(ids)
        batches = [(i, min(i + batch_size, total))
                   for i in range(0, total, batch_size)]
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_search_batch, index, ids, query_vec, s, e, top_k): (s, e)
                for s, e in batches
            }
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception:
                    pass

    # Post-filter by law_uid prefix if requested
    if law_uids:
        # Section UIDs follow pattern SEC_NNN_XXX where NNN is law number
        law_nums = set()
        for lu in law_uids:
            num = lu.replace("LAW_", "").zfill(3)
            law_nums.add(num)
        results = [
            (uid, score) for uid, score in results
            if any(uid.startswith(f"SEC_{n}") for n in law_nums)
        ]

    # Merge best score per uid, sort, return top_k
    best = {}
    for uid, score in results:
        if uid not in best or score > best[uid]:
            best[uid] = score
    sorted_results = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def reload_indices():
    """Force reload of cached indices (call after rebuild)."""
    global _law_index, _law_ids, _sec_index, _sec_ids
    _law_index = _law_ids = _sec_index = _sec_ids = None
