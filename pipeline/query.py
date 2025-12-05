import os
from .indexer import ChromaIndexer

class QueryService:
    def __init__(self, model_name=None, persist_dir=None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        self.indexer = ChromaIndexer(persist_dir=self.persist_dir, model_name=model_name)

    @staticmethod
    def distance_to_similarity(dist):
        try:
            d = float(dist)
            return 1.0 / (1.0 + d)
        except Exception:
            return 0.0

    def query_text(self, text, top_k=5, score_threshold=0.7):
        res = self.indexer.query(text, top_k=top_k)
        if not res:
            return {"decision": "no_match", "matches": []}

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        matches = []
        for doc, meta, dist in zip(docs, metas, dists):
            sim = self.distance_to_similarity(dist)
            matches.append({
                "title": meta.get("title"),
                "source": meta.get("source"),
                "verdict": meta.get("verdict"),
                "url": meta.get("url"),
                "score": sim
            })

        filtered = [m for m in matches if m["score"] >= score_threshold]
        if filtered:
            filtered.sort(key=lambda x: x["score"], reverse=True)
            return {"decision": "matched_fact", "matches": filtered}
        else:
            return {"decision": "no_match", "matches": matches}
