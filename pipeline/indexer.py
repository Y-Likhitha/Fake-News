import os
# Disable ONNX conversion to avoid corrupted ONNX in cloud
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CHROMA_DISABLE_EMBEDDINGS_H5", "true")
os.environ.setdefault("CHROMA_CONVERT_EMBEDDINGS_TO_ONNX", "false")

from sentence_transformers import SentenceTransformer
import chromadb

class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db", model_name=None):
        self.persist_dir = persist_dir
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
        # load embedding model
        self.model = SentenceTransformer(self.model_name)
        # persistent client (chroma 0.5+)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            self.collection = self.client.get_collection("factchecks")
        except Exception:
            self.collection = self.client.create_collection("factchecks")

    def add(self, ids, docs, metas):
        # compute embeddings locally and send them to chroma to avoid ONNX conversion
        embeddings = self.model.encode(docs, convert_to_numpy=True).tolist()
        # ensure metas contain no None
        def clean_meta(m):
            return {k: ("" if v is None else v) for k, v in m.items()}
        clean_metas = [clean_meta(m) for m in metas]
        self.collection.add(ids=ids, documents=docs, metadatas=clean_metas, embeddings=embeddings)

    def query(self, text, top_k=5):
        return self.collection.query(query_texts=[text], n_results=top_k)
