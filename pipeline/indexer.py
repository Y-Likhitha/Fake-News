import os, json, logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)

class ChromaIndexer:
    def __init__(self, persist_dir='./data/chroma_db', model_name='all-MiniLM-L6-v2'):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # initialize chroma client
        self.client = chromadb.Client(Settings(persist_directory=self.persist_dir))
        self.collection = None
        try:
            self.collection = self.client.get_collection('factchecks')
        except Exception:
            self.collection = self.client.create_collection('factchecks')

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def add(self, ids, texts, metadatas):
        # Chroma accepts raw texts and metadatas; it will embed if configured
        # But we will add embeddings directly via .add with documents
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        self.client.persist()
        logging.info('Added %d docs to Chroma', len(ids))

    def query(self, text, top_k=5):
        results = self.collection.query(query_texts=[text], n_results=top_k)
        return results
