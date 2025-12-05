import os
import shutil
import logging
import math
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import InvalidArgumentError, InternalError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable problematic ONNX conversions
os.environ["CHROMA_CONVERT_EMBEDDINGS_TO_ONNX"] = "false"
os.environ["CHROMA_DISABLE_EMBEDDINGS_H5"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBED_MODEL = "all-mpnet-base-v2"  # 768-dim model


class ChromaIndexer:
    def __init__(self, persist_dir="./data/chroma_db"):
        self.persist_dir = persist_dir
        self.model = SentenceTransformer(EMBED_MODEL)

        logger.info(f"Initializing ChromaIndexer with persist_dir={persist_dir}")

        # Try connecting to the database — if corruption occurs, auto-repair
        self.client = self._safe_create_client()

        # Try loading collection, otherwise auto-repair
        self.collection = self._safe_get_or_create_collection("factchecks")

    # ------------------------------------------------------------------
    # AUTO-REPAIR: Create PersistentClient safely
    # ------------------------------------------------------------------
    def _safe_create_client(self):
        try:
            return chromadb.PersistentClient(path=self.persist_dir)
        except Exception as e:
            logger.error(f"Chroma persistent storage corrupted! {e}")
            logger.warning("Auto-repair: wiping database folder...")

            try:
                shutil.rmtree(self.persist_dir)
            except Exception:
                pass

            os.makedirs(self.persist_dir, exist_ok=True)
            return chromadb.PersistentClient(path=self.persist_dir)

    # ------------------------------------------------------------------
    # AUTO-REPAIR: Load or recreate collection safely
    # ------------------------------------------------------------------
    def _safe_get_or_create_collection(self, name):
        try:
            return self.client.get_collection(name)
        except Exception:
            logger.warning("Collection is corrupted or missing. Recreating...")
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
            return self.client.create_collection(name)

    # ------------------------------------------------------------------
    # Metadata sanitization
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # SAFE ADD: with batching, corruption repair, and retry
    # ------------------------------------------------------------------
    def add(self, ids, docs, metas, batch_size=256):

        if not ids or not docs or not metas:
            raise ValueError("ids, docs, metas cannot be empty")

        if not (len(ids) == len(docs) == len(metas)):
            raise ValueError("Length mismatch between ids/docs/metas")

        logger.info(f"Indexing {len(ids)} documents...")

        # sanitize everything
        ids_clean = [self._ensure_str(i) for i in ids]
        docs_clean = [self._ensure_str(d) for d in docs]
        metas_clean = [self._sanitize_meta(m) for m in metas]

        # filter out empty docs
        filtered = []
        for i, d, m in zip(ids_clean, docs_clean, metas_clean):
            if d.strip():
                filtered.append((i, d.strip(), m))
            else:
                logger.info(f"Skipping empty doc id={i}")

        if not filtered:
            logger.warning("No usable documents after filtering.")
            return

        ids_f, docs_f, metas_f = zip(*filtered)

        # compute embeddings
        embeddings = self.model.encode(
            list(docs_f),
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        # add in batches
        def _do_batches():
            total = len(ids_f)
            batches = math.ceil(total / batch_size)

            for b in range(batches):
                s = b * batch_size
                e = min((b + 1) * batch_size, total)
                logger.info(f"Adding batch {b+1}/{batches} ({s}–{e})")
                self.collection.add(
                    ids=list(ids_f[s:e]),
                    documents=list(docs_f[s:e]),
                    metadatas=list(metas_f[s:e]),
                    embeddings=list(embeddings[s:e]),
                )

        # ---- SAFE EXECUTION WITH CORRUPTION RECOVERY ----
        try:
            _do_batches()
            logger.info("Indexing completed successfully.")

        except InternalError as err:
            logger.error(f"Chroma InternalError: {err}")
            logger.warning("Auto-repair: wiping DB and rebuilding...")

            # wipe DB completely
            try:
                shutil.rmtree(self.persist_dir)
            except Exception:
                pass

            os.makedirs(self.persist_dir, exist_ok=True)

            # recreate client + collection
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.create_collection("factchecks")

            # retry indexing
            _do_batches()
            logger.info("Auto-repair succeeded. Clean database rebuilt.")

        except InvalidArgumentError as ia:
            logger.error(f"InvalidArgumentError: {ia}")
            raise

        except Exception as e:
            logger.exception("Unexpected indexer error:")
            raise

