import os
import json
from dotenv import load_dotenv
from pathlib import Path
from .scraper import fetch_google_factchecks, scrape_altnews, scrape_factly
from .indexer import ChromaIndexer

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
GOOGLE_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")

RAW_PATH = Path(DATA_DIR) / "raw.jsonl"
RAW_PATH.parent.mkdir(exist_ok=True)


def load_raw():
    if not RAW_PATH.exists():
        return []
    return [json.loads(l) for l in RAW_PATH.read_text().splitlines()]


def save_raw(items):
    RAW_PATH.write_text("\n".join(json.dumps(i) for i in items))


def normalize(rec):
    return {
        "id": rec.get("url") or rec.get("claim"),
        "title": rec.get("title") or rec.get("claim"),
        "text": rec.get("text") or rec.get("claim"),
        "source": rec.get("source"),
        "url": rec.get("url"),
        "verdict": rec.get("verdict"),
    }


def append_unique(existing, new):
    ids = {r.get("id") for r in existing}
    added = []
    for n in new:
        nid = n.get("url") or n.get("claim")
        if nid not in ids:
            ids.add(nid)
            existing.append(n)
            added.append(n)
    return existing, added


def run_pipeline():
    existing = load_raw()
    new_items = []

    g = fetch_google_factchecks(GOOGLE_KEY)
    existing, added = append_unique(existing, g)
    new_items.extend(added)

    a = scrape_altnews()
    existing, added = append_unique(existing, a)
    new_items.extend(added)

    f = scrape_factly()
    existing, added = append_unique(existing, f)
    new_items.extend(added)

    save_raw(existing)

    # Embed into Chroma
    indexer = ChromaIndexer(persist_dir=CHROMA_DIR)
    normalized = [normalize(r) for r in existing]

    ids = [n["id"] for n in normalized]
    texts = [n["text"] for n in normalized]
    metas = [{"title": n["title"], "source": n["source"], "url": n["url"], "verdict": n["verdict"]} for n in normalized]

    indexer.add(ids, texts, metas)

    return len(new_items)
