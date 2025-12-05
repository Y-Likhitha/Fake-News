import os
from .indexer import ChromaIndexer
from sentence_transformers import SentenceTransformer
import numpy as np

class QueryService:
    def __init__(self, model_name=None, persist_dir=None):
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL','all-MiniLM-L6-v2')
        self.persist_dir = persist_dir or os.getenv('CHROMA_PERSIST_DIR','./data/chroma_db')
        self.indexer = ChromaIndexer(persist_dir=self.persist_dir, model_name=self.model_name)

    def query_text(self, text, top_k=5, score_threshold=0.7):
        # Chroma returns distances; for simplicity we return them as-is
        res = self.indexer.query(text, top_k=top_k)
        # res structure: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
        out = []
        if not res:
            return {'decision':'no_match','matches':[]}
        docs = res.get('documents',[[]])[0]
        metas = res.get('metadatas',[[]])[0]
        dists = res.get('distances',[[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma distances are implementation-specific; we pass them through
            out.append({'title': meta.get('title'), 'source': meta.get('source'), 'url': meta.get('url'), 'score': float(dist)})
        # Decide match if top score >= threshold (note: higher score may be better or worse depending on Chroma impl)
        decision = 'no_match'
        if out and out[0]['score'] >= score_threshold:
            decision = 'matched_fact'
        return {'decision':decision, 'matches': out}
