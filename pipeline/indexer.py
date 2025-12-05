import os
import logging
from sentence_transformers import SentenceTransformer
import chromadb

logging.basicConfig(level=logging.INFO)


class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db", model_name="all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(path=self.persist_dir)

        try:
            self.collection = self.client.get_collection("factchecks")
        except:
            self.collection = self.client.create_collection("factchecks")

    def add(self, ids, texts, metadatas):
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        logging.info(f"Added {len(ids)} docs to Chroma.")

    def query(self, text, top_k=5):
        return self.collection.query(
            query_texts=[text],
            n_results=top_k
        )
