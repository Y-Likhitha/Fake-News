import os
import logging
from sentence_transformers import SentenceTransformer
import chromadb

logging.basicConfig(level=logging.INFO)

class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db", model_name="all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Persistent client for chroma 0.5+
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            self.collection = self.client.get_collection("factchecks")
        except Exception:
            self.collection = self.client.create_collection("factchecks")

    def embed_texts(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def add(self, ids, documents, metadatas):
        # Chroma requires non-empty lists and string documents
        if not ids or not documents or not metadatas:
            raise ValueError("Non-empty lists required for ids, documents, metadatas")
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logging.info("Added %d documents to Chroma", len(ids))

    def query(self, text, top_k=5):
        # returns dict with documents, metadatas, distances
        return self.collection.query(query_texts=[text], n_results=top_k)
