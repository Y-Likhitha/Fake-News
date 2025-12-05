# indexer.py
import os
import shutil
import math
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import InvalidArgumentError, InternalError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable ONNX conversion (avoid protobuf issues)
os.environ.setdefault("CHROMA_CONVERT_EMBEDDINGS_TO_ONNX", "false")
os.environ.setdefault("CHROMA_DISABLE_EMBEDDINGS_H5", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")  # 768-dim default


class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db", fallback_in_memory=True):
        """
        Tries to create a PersistentClient at persist_dir. If that fails,
        and fallback_in_memory=True, uses an in-memory chromadb.Client() instead.
        """
        self.persist_dir = persist_dir
        self.model = SentenceTransformer(EMBED_MODEL)
        self.fallback_in_memory = fallback_in_memory

        self.client = None
        self.collection = None
        self._init_client_and_collection()

    def _init_client_and_collection(self):
        # Try persist first
        try:
            logger.info(f"Attempting PersistentClient at {self.persist_dir}")
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            # If client created, try to get/create collection
            try:
                self.collection = self.client.get_collection("factchecks")
            except Exception:
                logger.info("Persistent collection missing or corrupted; trying to create it.")
                self.collection = self.client.create_collection("factchecks")
            logger.info("Using persistent Chroma client.")
            return
        except Exception as e:
            logger.warning(f"Persistent Chroma client FAILED: {e}")

        # Fallback to in-memory client if allowed
        if self.fallback_in_memory:
            try:
                logger.info("Falling back to in-memory Chroma client.")
                # chromadb.Client() returns in-memory client
                self.client = chromadb.Client()
                try:
                    self.collection = self.client.get_collection("factchecks")
                except Exception:
                    self.collection = self.client.create_collection("factchecks")
                logger.info("Using in-memory Chroma client (non-persistent).")
                return
            except Exception as e2:
                logger.exception("In-memory Chroma client also failed.")
                raise RuntimeError("Both persistent and in-memory Chroma client creation failed.") from e2

        raise RuntimeError("Persistent Chroma client creation failed and fallback_in_memory is disabled.")

    def _sanitize_meta(self, m: dict):
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

    def add(self, ids, docs, metas, batch_size=256):
        if not ids or not docs or not metas:
            raise ValueError("ids, docs, metas must be non-empty lists")
        if not (len(ids) == len(docs) == len(metas)):
            raise ValueError("Length mismatch: ids, docs, metas")

        logger.info(f"Indexing {len(ids)} documents (persist_dir={self.persist_dir}).")
        ids_clean = [self._ensure_str(i) for i in ids]
        docs_clean = [self._ensure_str(d) for d in docs]
        metas_clean = [self._sanitize_meta(m) for m in metas]

        filtered = []
        for i, d, m in zip(ids_clean, docs_clean, metas_clean):
            if isinstance(d, str) and d.strip():
                filtered.append((i, d.strip(), m))
            else:
                logger.info(f"Skipping empty/invalid document for id={i}")

        if not filtered:
            logger.warning("No valid documents to index after filtering.")
            return

        ids_final, docs_final, metas_final = zip(*filtered)

        # compute embeddings locally (avoid ONNX)
        try:
            embeddings = self.model.encode(list(docs_final), convert_to_numpy=True, normalize_embeddings=True).tolist()
        except Exception as e:
            logger.exception("Embedding generation failed")
            raise

        # batch add helper
        def _add_batches(client_collection):
            n = len(ids_final)
            batches = math.ceil(n / batch_size)
            for b in range(batches):
                s = b * batch_size
                e = min((b + 1) * batch_size, n)
                logger.info(f"Adding batch {b+1}/{batches} (items {s}..{e})")
                client_collection.add(
                    ids=list(ids_final[s:e]),
                    documents=list(docs_final[s:e]),
                    metadatas=list(metas_final[s:e]),
                    embeddings=list(embeddings[s:e])
                )

        # Attempt to add; if persistent fails mid-way, try recover (or fallback to in-memory)
        try:
            _add_batches(self.collection)
            logger.info("Indexing finished successfully.")
            return
        except Exception as exc:
            logger.warning(f"Error while adding to current collection: {exc}")

            # if we were using persistent, try to recreate it and retry once
            if isinstance(self.client, chromadb.PersistentClient):
                try:
                    logger.info("Attempting to delete & recreate persistent collection and retry.")
                    try:
                        self.client.delete_collection("factchecks")
                    except Exception:
                        logger.info("delete_collection may have failed or collection not present.")
                    self.collection = self.client.create_collection("factchecks")
                    _add_batches(self.collection)
                    logger.info("Reindex succeeded after recreating persistent collection.")
                    return
                except Exception as e2:
                    logger.warning(f"Persistent recreate failed: {e2}")

            # final fallback: try in-memory if allowed
            if self.fallback_in_memory:
                try:
                    logger.info("Falling back to a new in-memory client and indexing there.")
                    mem_client = chromadb.Client()
                    try:
                        mem_collection = mem_client.get_collection("factchecks")
                    except Exception:
                        mem_collection = mem_client.create_collection("factchecks")
                    _add_batches(mem_collection)
                    # switch our references so queries use in-memory collection from now on
                    self.client = mem_client
                    self.collection = mem_collection
                    logger.info("Indexing succeeded on in-memory fallback.")
                    return
                except Exception as e3:
                    logger.exception("In-memory fallback also failed.")
                    raise

            # ultimately re-raise if we couldn't recover
            raise
