import os
from .indexer import ChromaIndexer


class QueryService:
    def __init__(self, model_name=None, persist_dir=None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

        self.indexer = ChromaIndexer(
            persist_dir=self.persist_dir,
            model_name=self.model_name
        )

    def query_text(self, text, top_k=5, score_threshold=0.7):

        # Query Chroma
        result = self.indexer.query(text, top_k=top_k)

        # Chroma always returns dict with lists, never False
        if result is None or "documents" not in result:
            return {"decision": "no_match", "matches": []}

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        matches = []

        for doc, meta, dist in zip(docs, metas, dists):
            matches.append({
                "title": meta.get("title"),
                "source": meta.get("source"),
                "url": meta.get("url"),
                "score": float(dist)
            })

        # Final decision
        if matches and matches[0]["score"] >= score_threshold:
            decision = "matched_fact"
        else:
            decision = "no_match"

        return {"decision": decision, "matches": matches}
