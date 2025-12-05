import os
from .indexer import ChromaIndexer


class QueryService:
    def __init__(self, model_name=None, persist_dir=None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        self.indexer = ChromaIndexer(persist_dir=self.persist_dir)

    def query_text(self, text, top_k=5, score_threshold=0.7):
        result = self.indexer.query(text, top_k)

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        out = []

        for meta, dist in zip(metas, dists):
            similarity = 1 / (1 + float(dist))  # convert Euclidean distance → similarity
            out.append({
                "title": meta.get("title"),
                "source": meta.get("source"),
                "verdict": meta.get("verdict"),
                "url": meta.get("url"),
                "score": similarity
            })

        if out and out[0]["score"] >= score_threshold:
            decision = "matched_fact"
        else:
            decision = "no_match"

        return {"decision": decision, "matches": out}
