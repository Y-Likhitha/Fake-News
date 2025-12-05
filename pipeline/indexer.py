
import os
from sentence_transformers import SentenceTransformer
import chromadb

class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.collection = self.client.get_collection("factchecks")
        except:
            self.collection = self.client.create_collection("factchecks")

    def add(self, ids, docs, metas):
        self.collection.add(ids=ids, documents=docs, metadatas=metas)

    def query(self, text, top_k=5):
        return self.collection.query(query_texts=[text], n_results=top_k)
