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
RAW_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_raw():
    if not RAW_PATH.exists():
        return []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_raw(items):
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def normalize(record):
    text = record.get("text") or record.get("claim") or ""
    title = record.get("title") or record.get("claim") or "Untitled"

    return {
        "id": record.get("url") or record.get("claim") or str(abs(hash(text))),
        "title": title,
        "text": text.strip(),
        "source": record.get("source"),
        "url": record.get("url"),
        "verdict": record.get("verdict")
    }



def append_unique(existing, new_items):
    seen = {item.get("id") for item in existing}
    added = []

    for item in new_items:
        uid = item.get("url") or item.get("claim")
        if uid not in seen:
            item["id"] = uid
            seen.add(uid)
            existing.append(item)
            added.append(item)

    return existing, added


def run_pipeline():
    # Load current dataset
    existing = load_raw()
    new_records = []

    # Google Fact Check API
    google_items = fetch_google_factchecks(GOOGLE_KEY)
    existing, added = append_unique(existing, google_items)
    new_records.extend(added)

    # Altnews
    altnews_items = scrape_altnews()
    existing, added = append_unique(existing, altnews_items)
    new_records.extend(added)

    # Factly
    factly_items = scrape_factly()
    existing, added = append_unique(existing, factly_items)
    new_records.extend(added)

    # Save updated raw dataset
    save_raw(existing)

    # Build Chroma index
    indexer = ChromaIndexer(persist_dir=CHROMA_DIR)

    normalized = [normalize(item) for item in existing]
    ids = [item["id"] for item in normalized]
    texts = [item["text"] for item in normalized]
    metadatas = [
        {
            "title": item["title"],
            "source": item["source"],
            "url": item["url"],
            "verdict": item["verdict"]
        }
        for item in normalized
    ]

    indexer.add(ids, texts, metadatas)

    return len(new_records)
