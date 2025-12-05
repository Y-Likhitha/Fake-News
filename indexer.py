import os, shutil
# disable ONNX conversion & parallelism
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('CHROMA_DISABLE_EMBEDDINGS_H5','true')
os.environ.setdefault('CHROMA_CONVERT_EMBEDDINGS_TO_ONNX','false')

from sentence_transformers import SentenceTransformer
import chromadb

# hard-lock model to mpnet for consistency
EMBED_MODEL = 'all-mpnet-base-v2'

# reset local chroma db to avoid dimension mismatch
def reset_chroma(persist_dir='./data/chroma_db'):
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
        except Exception:
            pass

class SimpleIndexer:
    def __init__(self, persist_dir='./data/chroma_db'):
        reset_chroma(persist_dir)
        self.persist_dir = persist_dir
        self.model = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            self.collection = self.client.get_collection('factchecks')
        except Exception:
            self.collection = self.client.create_collection('factchecks')

    def add(self, ids, docs, metas):
        # sanitize metas
        clean_metas = []
        for m in metas:
            clean = {k: ('' if v is None else v) for k, v in m.items()}
            clean_metas.append(clean)
        # compute embeddings locally and pass to chroma (avoid ONNX)
        embeddings = self.model.encode(docs, convert_to_numpy=True).tolist()
        self.collection.add(ids=ids, documents=docs, metadatas=clean_metas, embeddings=embeddings)

    def query(self, text, top_k=5):
        return self.collection.query(query_texts=[text], n_results=top_k)
