# indexer.py
import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_DIR = Path(DATA_DIR) / "faiss_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

_INDEX_FILE = INDEX_DIR / "index.faiss"
_IDS_FILE = INDEX_DIR / "ids.pkl"
_METAS_FILE = INDEX_DIR / "metas.pkl"
_DIM_FILE = INDEX_DIR / "dim.txt"


class FAISSIndexer:
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        # Normalize to unit length for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def build_index(self, ids, docs, metas):
        """
        Rebuild the FAISS index from scratch from given documents.
        This approach is simple and robust (no incremental complexity).
        """
        if not ids or not docs or not metas:
            raise ValueError("ids/docs/metas must be non-empty lists")

        if not (len(ids) == len(docs) == len(metas)):
            raise ValueError("Length mismatch")

        logger.info("Computing embeddings for %d documents", len(docs))
        emb = self.model.encode(docs, convert_to_numpy=True, normalize_embeddings=False)
        emb = emb.astype("float32")
        emb = self._normalize(emb)

        # create IndexFlatIP for cosine (after normalization)
        index = faiss.IndexFlatIP(self.dim)
        index.add(emb)

        # persist
        faiss.write_index(index, str(_INDEX_FILE))
        with open(_IDS_FILE, "wb") as f:
            pickle.dump(list(ids), f)
        with open(_METAS_FILE, "wb") as f:
            pickle.dump(list(metas), f)
        with open(_DIM_FILE, "w") as f:
            f.write(str(self.dim))

        logger.info("FAISS index saved (%s)", _INDEX_FILE)

    def index_exists(self):
        return _INDEX_FILE.exists() and _IDS_FILE.exists() and _METAS_FILE.exists()

    def load_index(self):
        if not self.index_exists():
            raise RuntimeError("Index not found on disk")
        index = faiss.read_index(str(_INDEX_FILE))
        with open(_IDS_FILE, "rb") as f:
            ids = pickle.load(f)
        with open(_METAS_FILE, "rb") as f:
            metas = pickle.load(f)
        with open(_DIM_FILE, "r") as f:
            dim = int(f.read().strip())
        return index, ids, metas, dim
