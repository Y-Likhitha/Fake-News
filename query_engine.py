import os
import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-mpnet-base-v2"

class QueryEngine:
    def __init__(self, persist_dir="./data/chroma_db"):
        self.persist_dir = persist_dir

        # Load mpnet model for query embeddings
        self.model = SentenceTransformer(EMBED_MODEL)

        # Connect to existing ChromaDB (DO NOT RESET)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection("factchecks")

    @staticmethod
    def distance_to_similarity(d):
        try:
            return 1.0/(1.0+float(d))
        except:
            return 0.0

    def query_text(self, text, top_k=5, score_threshold=0.7):
        # Compute embeddings SAME WAY AS INDEXER
        query_emb = self.model.encode([text], convert_to_numpy=True).tolist()

        res = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )

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

        return {"decision": "no_match", "matches": matches}
