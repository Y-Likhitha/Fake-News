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


# ---------------------------
# Load raw dataset
# ---------------------------
def load_raw():
    if not RAW_PATH.exists():
        return []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------
# Save raw dataset
# ---------------------------
def save_raw(items):
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# ---------------------------
# Normalize scraped records
# ---------------------------
def normalize(record):
    text = record.get("text") or record.get("claim") or ""
    title = record.get("title") or record.get("claim") or "Untitled"

    uid = record.get("url") or record.get("claim") or str(abs(hash(text)))

    return {
        "id": uid,
        "title": title.strip(),
        "text": text.strip(),
        "source": record.get("source"),
        "url": record.get("url"),
        "verdict": record.get("verdict")
    }


# ---------------------------
# Merge without duplicates
# ---------------------------
def append_unique(existing, new_items):
    seen = {item.get("id") for item in existing}
    added = []

    for item in new_items:
        uid = item.get("url") or item.get("claim")
        if not uid:
            continue
        if uid not in seen:
            seen.add(uid)
            existing.append(item)
            added.append(item)

    return existing, added


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def run_pipeline():
    existing = load_raw()
    new_records = []

    # Fetch data
    google_items = fetch_google_factchecks(GOOGLE_KEY)
    existing, added = append_unique(existing, google_items)
    new_records.extend(added)

    alt_items = scrape_altnews()
    existing, added = append_unique(existing, alt_items)
    new_records.extend(added)

    factly_items = scrape_factly()
    existing, added = append_unique(existing, factly_items)
    new_records.extend(added)

    # Save updated dataset
    save_raw(existing)

    # Normalize for indexing
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

    # CLEANING: remove any empty / None texts
    clean_ids = []
    clean_texts = []
    clean_metas = []

    for i, t, m in zip(ids, texts, metadatas):
        if t and isinstance(t, str) and t.strip():
            clean_ids.append(i)
            clean_texts.append(t.strip())
            clean_metas.append(m)

    # Index into Chroma
    indexer = ChromaIndexer(persist_dir=CHROMA_DIR)

    # Avoid inserting empty lists into Chroma
    if clean_ids and clean_texts and clean_metas:
        indexer.add(clean_ids, clean_texts, clean_metas)
    else:
        print("⚠️ No valid documents to add — scraping may have returned no usable text.")


    return len(new_records)
