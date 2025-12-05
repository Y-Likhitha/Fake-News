
import os
from .indexer import ChromaIndexer

class QueryService:
    def __init__(self, model_name=None, persist_dir=None):
        self.indexer = ChromaIndexer(persist_dir or "./data/chroma_db")

    def distance_to_similarity(self, d):
        try: return 1/(1+float(d))
        except: return 0.0

    def query_text(self, text, top_k=5, score_threshold=0.7):
        r = self.indexer.query(text, top_k)
        docs = r.get("documents",[[]])[0]
        metas = r.get("metadatas",[[]])[0]
        dists = r.get("distances",[[]])[0]

        matches=[]
        for doc,m,dist in zip(docs,metas,dists):
            sim=self.distance_to_similarity(dist)
            if sim>=score_threshold:
                matches.append({
                    "title": m.get("title"),
                    "source": m.get("source"),
                    "verdict": m.get("verdict"),
                    "url": m.get("url"),
                    "score": sim
                })
        if matches:
            matches.sort(key=lambda x:x["score"], reverse=True)
            return {"decision":"matched_fact","matches":matches}
        return {"decision":"no_match","matches":[]}
