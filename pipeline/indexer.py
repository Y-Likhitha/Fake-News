import os
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)


class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db", model_name="all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.model_name = model_name

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Correct Chroma persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Load or create collection
        try:
            self.collection = self.client.get_collection("factchecks")
        except Exception:
            self.collection = self.client.create_collection("factchecks")

    def embed(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def add(self, ids, texts, metadatas):
        """
        Add items to Chroma.
        IMPORTANT: No .persist() needed in ChromaDB 0.5+
        """
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        logging.info(f"Added {len(ids)} items to ChromaDB collection.")

    def query(self, text, top_k=5):
        """
        Query collection with semantic search.
        """
        return self.collection.query(
            query_texts=[text],
            n_results=top_k
        )
