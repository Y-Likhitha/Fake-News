# query_engine.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from indexer import FAISSIndexer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")


class QueryEngine:
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.indexer = FAISSIndexer(model_name)
        if self.indexer.index_exists():
            self.index, self.ids, self.metas, self.dim = self.indexer.load_index()
        else:
            self.index = None
            self.ids = []
            self.metas = []

    @staticmethod
    def _normalize(emb):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    @staticmethod
    def distance_to_similarity(ip_score):
        # ip_score = cosine since vectors are normalized; range [-1,1]
        # Map to [0,1], but we'll keep as-is for thresholding expecting 0..1
        return float((ip_score + 1.0) / 2.0)

    def query_text(self, text, top_k=5, score_threshold=0.7):
        if self.index is None:
            return {"decision": "no_match", "matches": []}

        emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=False)
        emb = emb.astype("float32")
        emb = self._normalize(emb)

        # faiss returns (D, I) where D are inner products
        D, I = self.index.search(emb, top_k)
        D = D[0]
        I = I[0]

        matches = []
        for score, idx in zip(D, I):
            if idx < 0 or idx >= len(self.ids):
                continue
            sim = float(score)  # cosine inner product in [-1,1]
            similarity = self.distance_to_similarity(sim)
            meta = self.metas[idx]
            matches.append({
                "id": self.ids[idx],
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "verdict": meta.get("verdict", ""),
                "url": meta.get("url", ""),
                "score": similarity
            })

        # filter by threshold (user uses 0..1 threshold)
        filtered = [m for m in matches if m["score"] >= score_threshold]
        if filtered:
            filtered.sort(key=lambda x: x["score"], reverse=True)
            return {"decision": "matched_fact", "matches": filtered}
        else:
            # return raw matches for debugging but decision no_match
            return {"decision": "no_match", "matches": matches}
