# query_engine.py
import numpy as np
from sentence_transformers import SentenceTransformer
from indexer import FAISSIndexer
from google_api import fetch_google_factchecks

class QueryEngine:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.indexer = FAISSIndexer(model_name)
        if self.indexer.index_exists():
            self.index, self.ids, self.metas, self.dim = self.indexer.load_index()
        else:
            self.index = None

    @staticmethod
    def _normalize(emb):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    @staticmethod
    def distance_to_similarity(ip):
        return float((ip + 1.0) / 2.0)

    def _faiss_query(self, text, top_k):
        if self.index is None:
            return []

        emb = self.model.encode([text], convert_to_numpy=True).astype("float32")
        emb = self._normalize(emb)

        D, I = self.index.search(emb, top_k)
        D, I = D[0], I[0]

        matches = []
        for score, idx in zip(D, I):
            if idx < 0:
                continue
            meta = self.metas[idx]
            sim = self.distance_to_similarity(score)
            matches.append({
                "title": meta["title"],
                "source": meta["source"],
                "verdict": meta.get("verdict", ""),
                "url": meta["url"],
                "score": sim
            })
        return matches

    def query_text(self, text, top_k=5, score_threshold=0.7):
        
        # -------------------------
        # 1️⃣  Semantic matches (FAISS)
        # -------------------------
        semantic_matches = self._faiss_query(text, top_k)

        # -------------------------
        # 2️⃣  Google Fact Check API (Exact fact-checks)
        # -------------------------
        google_matches_raw = fetch_google_factchecks(query=text, page_size=5)

        google_matches = []
        for g in google_matches_raw:
            google_matches.append({
                "title": g["title"],
                "source": g["source"],
                "verdict": g["verdict"],
                "url": g["url"],
                "score": 1.0  # Google API = exact match
            })

        # Merge both
        all_matches = semantic_matches + google_matches

        # -------------------------
        # 3️⃣  Threshold filtering
        # -------------------------
        filtered = [m for m in all_matches if m["score"] >= score_threshold]

        if filtered:
            filtered.sort(key=lambda x: x["score"], reverse=True)
            return {"decision": "matched_fact", "matches": filtered}

        return {"decision": "no_match", "matches": all_matches}
