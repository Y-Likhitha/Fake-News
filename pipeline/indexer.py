import os

# Disable ONNX conversion for Streamlit Cloud
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CHROMA_DISABLE_EMBEDDINGS_H5", "true")
os.environ.setdefault("CHROMA_CONVERT_EMBEDDINGS_TO_ONNX", "false")

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import InvalidArgumentError


class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db", model_name=None):
        self.persist_dir = persist_dir
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

        # Load embedding model
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        try:
            self.collection = self.client.get_collection("factchecks")

            # Check dimensionality mismatch
            info = self.collection.peek()
            if info and "embeddings" in info and len(info["embeddings"][0]) != self.embedding_dim:
                print("⚠️ Dimension mismatch detected — resetting collection")
                self.client.delete_collection("factchecks")
                self.collection = self.client.create_collection("factchecks")

        except Exception:
            self.collection = self.client.create_collection("factchecks")

    def add(self, ids, docs, metas):
        embeddings = self.model.encode(docs, convert_to_numpy=True).tolist()

        # Ensure metadata has no None
        def clean(m):
            return {k: ("" if v is None else v) for k, v in m.items()}
        metas = [clean(m) for m in metas]

        try:
            self.collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metas
            )
        except InvalidArgumentError as e:
            # AUTO FIX: Delete old collection and recreate with correct dimension
            print("🔥 Rebuilding collection due to embedding dimension mismatch")
            self.client.delete_collection("factchecks")
            self.collection = self.client.create_collection("factchecks")

            self.collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metas
            )

    def query(self, text, top_k=5):
        return self.collection.query(query_texts=[text], n_results=top_k)
