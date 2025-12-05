import os
import shutil
import math
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import InvalidArgumentError, InternalError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable ONNX conversion (important!)
os.environ.setdefault("CHROMA_CONVERT_EMBEDDINGS_TO_ONNX", "false")
os.environ.setdefault("CHROMA_DISABLE_EMBEDDINGS_H5", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EMBED_MODEL = "all-mpnet-base-v2"  # 768-dim


class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db"):
        """
        The indexer is used ONLY during pipeline ingestion.
        QueryEngine loads embeddings separately (no DB reset).
        """
        self.persist_dir = persist_dir
        self.model = SentenceTransformer(EMBED_MODEL)

        # Persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Try to load or create the collection
        try:
            self.collection = self.client.get_collection("factchecks")
        except Exception:
            logger.warning("Collection missing — creating new one.")
            self.collection = self.client.create_collection("factchecks")

    # ---------------------------
    # Helper: sanitize metadata
    # ---------------------------
    def _sanitize_meta(self, m: dict):
        clean = {}
        for k, v in (m or {}).items():
            if v is None:
                clean[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                # fallback for nested objects
                clean[k] = str(v)
        return clean

    # ---------------------------
    # Helper: ensure string
    # ---------------------------
    def _ensure_str(self, x):
        return "" if x is None else str(x)

    # ---------------------------
    # SAFE ADD METHOD (Batched)
    # ---------------------------
    def add(self, ids, docs, metas, batch_size=256):
        """
        Adds documents + embeddings safely:
        - sanitizes everything
        - skips empty docs
        - generates embeddings locally (no ONNX)
        - adds in batches
        - automatic recovery if Chroma errors
        """

        # --- Basic validation ---
        if not ids or not docs or not metas:
            raise ValueError("ids, docs, metas must be non-empty lists")

        if not (len(ids) == len(docs) == len(metas)):
            raise ValueError(
                f"Length mismatch: ids={len(ids)}, docs={len(docs)}, metas={len(metas)}"
            )

        # --- Sanitize ids, docs, metadata ---
        ids_clean = [self._ensure_str(i) for i in ids]
        docs_clean = [self._ensure_str(d) for d in docs]
        metas_clean = [self._sanitize_meta(m) for m in metas]

        # --- Filter out empty docs ---
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

        # --- Generate embeddings (768-dim) ---
        try:
            embeddings = self.model.encode(
                list(docs_final),
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()
        except Exception as e:
            logger.exception("Embedding generation failed")
            raise

        if len(embeddings) != len(docs_final):
            raise ValueError("Embedding count mismatch")

        # ---------------------------
        # Batch insert helper
        # ---------------------------
        def _add_batches():
            n = len(ids_final)
            batches = math.ceil(n / batch_size)

            for b in range(batches):
                s = b * batch_size
                e = min((b + 1) * batch_size, n)
                logger.info(f"Adding batch {b+1}/{batches} (items {s} to {e})")

                self.collection.add(
                    ids=list(ids_final[s:e]),
                    documents=list(docs_final[s:e]),
                    metadatas=list(metas_final[s:e]),
                    embeddings=list(embeddings[s:e])
                )

        # ---------------------------
        # Attempt add — with recovery
        # ---------------------------
        try:
            _add_batches()
            logger.info(f"Successfully added {len(ids_final)} documents.")

        except InternalError as e:
            logger.warning(f"Chroma InternalError: {e}")
            logger.info("Attempting auto-recovery by rebuilding collection...")

            try:
                # delete & recreate
                try:
                    self.client.delete_collection("factchecks")
                except Exception:
                    logger.warning("Collection deletion failed or not needed")

                self.collection = self.client.create_collection("factchecks")

                # retry batches
                _add_batches()
                logger.info("Recovery succeeded — data reindexed.")

            except Exception as ee:
                logger.exception("Recovery failed — cannot reindex.")
                raise ee

        except InvalidArgumentError as iv:
            logger.exception("InvalidArgumentError during add")
            raise

        except Exception as ex:
            logger.exception("Unexpected error during add")
            raise


# END FILE
