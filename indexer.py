# indexer.py (FINAL SAFE VERSION)
import os
import shutil
import math
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import InvalidArgumentError, InternalError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable problematic conversions
os.environ["CHROMA_CONVERT_EMBEDDINGS_TO_ONNX"] = "false"
os.environ["CHROMA_DISABLE_EMBEDDINGS_H5"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBED_MODEL = "all-mpnet-base-v2"  # 768 dim


class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db", fallback_in_memory=True):
        self.persist_dir = persist_dir
        self.fallback_in_memory = fallback_in_memory
        self.model = SentenceTransformer(EMBED_MODEL)

        logger.info(f"Initializing ChromaIndexer @ {persist_dir}")

        # Try persistent storage
        self.client, self.collection, self.is_persistent = self._init_client()

    # ---------------------------------------------------------------
    # SAFE creation of persistent → fallback to in-memory
    # ---------------------------------------------------------------
    def _init_client(self):
        try:
            logger.info("Trying PersistentClient...")
            client = chromadb.PersistentClient(path=self.persist_dir)
            try:
                collection = client.get_collection("factchecks")
            except:
                collection = client.create_collection("factchecks")

            logger.info("Using PERSISTENT Chroma client.")
            return client, collection, True

        except Exception as e:
            logger.warning(f"Persistent client failed: {e}")

            if not self.fallback_in_memory:
                raise RuntimeError("Persistent storage unusable and fallback disabled.")

            logger.info("Falling back to IN-MEMORY Chroma client...")
            client = chromadb.Client()  # in-memory
            try:
                collection = client.get_collection("factchecks")
            except:
                collection = client.create_collection("factchecks")

            logger.info("Using IN-MEMORY Chroma client.")
            return client, collection, False

    # ---------------------------------------------------------------
    # Sanitize helpers
    # ---------------------------------------------------------------
    def _sanitize_meta(self, m):
        clean = {}
        for k, v in (m or {}).items():
            if v is None:
                clean[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def _ensure_str(self, x):
        return "" if x is None else str(x)

    # ---------------------------------------------------------------
    # Add with auto-repair
    # ---------------------------------------------------------------
    def add(self, ids, docs, metas, batch_size=256):

        if not ids or not docs or not metas:
            raise ValueError("ids/docs/metas cannot be empty")

        if not (len(ids) == len(docs) == len(metas)):
            raise ValueError("Length mismatch between ids/docs/metas")

        logger.info(f"Indexing {len(ids)} items...")

        ids_clean = [self._ensure_str(x) for x in ids]
        docs_clean = [self._ensure_str(x) for x in docs]
        metas_clean = [self._sanitize_meta(x) for x in metas]

        # Filter empty docs
        final_triplets = []
        for i, d, m in zip(ids_clean, docs_clean, metas_clean):
            if d.strip():
                final_triplets.append((i, d.strip(), m))
            else:
                logger.info(f"Skipping empty doc {i}")

        if not final_triplets:
            logger.warning("No documents to index after filtering!")
            return

        ids_f, docs_f, metas_f = zip(*final_triplets)

        # Embeddings
        try:
            emb = self.model.encode(
                list(docs_f),
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()
        except Exception as e:
            logger.exception("Embedding failed:")
            raise

        # Batch insert helper
        def _do_batches(collection):
            total = len(ids_f)
            batches = math.ceil(total / batch_size)
            for b in range(batches):
                s = b * batch_size
                e = min((b + 1) * batch_size, total)
                logger.info(f"Adding batch {b+1}/{batches}")

                collection.add(
                    ids=list(ids_f[s:e]),
                    documents=list(docs_f[s:e]),
                    metadatas=list(metas_f[s:e]),
                    embeddings=list(emb[s:e]),
                )

        # First attempt
        try:
            _do_batches(self.collection)
            logger.info("Indexing completed.")
            return

        except Exception as err:
            logger.warning(f"Add failed: {err}")

            # Attempt persistent repair (ONLY if persistent)
            if self.is_persistent:
                logger.info("Trying persistent DB repair...")

                try:
                    shutil.rmtree(self.persist_dir)
                except:
                    pass
                os.makedirs(self.persist_dir, exist_ok=True)

                # Re-init client
                self.client, self.collection, self.is_persistent = self._init_client()

                # retry
                _do_batches(self.collection)
                logger.info("Repair & retry successful.")
                return

            # If persistent already failed → fallback already active → nothing else to fix
            logger.error("Error happened on IN-MEMORY client — cannot repair further.")
            raise
