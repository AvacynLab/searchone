try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.core.config import FAISS_INDEX_FILE, DATA_DIR
import logging
from app.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index_path = FAISS_INDEX_FILE
        self.mapping_path = DATA_DIR / "id_mapping.json"
        if _HAS_FAISS:
            self._use_faiss = True
        else:
            self._use_faiss = False
        self._load_or_init()

    def _load_or_init(self):
        if self._use_faiss:
            if self.index_path.exists():
                idx = faiss.read_index(str(self.index_path))
                if idx.d != self.dim:
                    # dimension mismatch: reset index and mapping
                    logger.warning("FAISS index dimension %s != expected %s, resetting index", idx.d, self.dim)
                    self.index = faiss.IndexFlatIP(self.dim)
                    self.id_mapping = {}
                    self.next_id = 0
                else:
                    self.index = idx
                    if self.mapping_path.exists():
                        with open(self.mapping_path, "r", encoding="utf-8") as f:
                            self.id_mapping = json.load(f)
                    else:
                        self.id_mapping = {}
                    self.next_id = int(max(self.id_mapping.keys())) + 1 if self.id_mapping else 0
            else:
                self.index = faiss.IndexFlatIP(self.dim)
                self.id_mapping = {}
                self.next_id = 0
        else:
            # simple in-memory fallback index
            self.index = None
            self.id_mapping = {}
            self._embs = []  # list of numpy arrays
            self.next_id = 0

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        # embeddings shape: (n, dim)
        n, d = embeddings.shape
        assert d == self.dim
        embeddings = self._normalize(embeddings)
        ids = []
        for m in metadatas:
            sid = str(self.next_id)
            self.id_mapping[sid] = m
            ids.append(self.next_id)
            self.next_id += 1
        if self._use_faiss:
            self.index.add(embeddings.astype('float32'))
            self._save()
        else:
            for i in range(n):
                self._embs.append(np.array(embeddings[i], dtype=float))
        logger.info("Added %d embeddings to index (faiss=%s)", n, self._use_faiss)
        return ids

    def search(self, query_emb: np.ndarray, top_k: int = 5):
        # query_emb shape: (1, dim) or (n, dim)
        results = []
        if self._use_faiss and self.index.ntotal == 0:
            return results
        if self._use_faiss:
            query_emb = self._normalize(query_emb)
            scores, idxs = self.index.search(query_emb.astype('float32'), top_k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                sid = str(idx)
                meta = self.id_mapping.get(sid)
                results.append({"score": float(score), "metadata": meta})
        else:
            # naive cosine-like similarity using dot product
            q = np.array(query_emb).reshape(-1)
            sims = []
            for i, e in enumerate(self._embs):
                # cosine-like score
                denom = (np.linalg.norm(q) * np.linalg.norm(e))
                score = float(np.dot(q, e) / denom) if denom != 0 else 0.0
                sims.append((score, i))
            sims.sort(key=lambda x: x[0], reverse=True)
            for score, idx in sims[:top_k]:
                sid = str(idx)
                meta = self.id_mapping.get(sid)
                results.append({"score": float(score), "metadata": meta})
        logger.debug("Search returned %d results", len(results))
        return results

    def _save(self):
        if not self._use_faiss:
            return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.id_mapping, f, ensure_ascii=False, indent=2)

    def stats(self):
        """Return basic health stats about the vector store."""
        if self._use_faiss:
            return {
                "use_faiss": True,
                "dim": self.dim,
                "ntotal": int(self.index.ntotal),
                "mapping_size": len(self.id_mapping),
                "index_path": str(self.index_path),
                "mapping_path": str(self.mapping_path),
                "dim_mismatch": self.index.d != self.dim if hasattr(self.index, "d") else False,
            }
        else:
            return {
                "use_faiss": False,
                "dim": self.dim,
                "emb_count": len(self._embs),
                "mapping_size": len(self.id_mapping),
            }

    def reset(self):
        """Reset the FAISS index and mapping files, then re-init in-memory structures."""
        try:
            if self.index_path.exists():
                self.index_path.unlink()
            if self.mapping_path.exists():
                self.mapping_path.unlink()
        except Exception:
            logger.warning("Could not delete index/mapping files during reset", exc_info=True)
        self._load_or_init()

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings if they are non-zero to avoid scaling issues."""
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms
