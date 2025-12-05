# pipeline.py
from google_api import fetch_google_factchecks

import os
import json
from pathlib import Path
from scraper import fetch_all
from indexer import FAISSIndexer

DATA_DIR = os.getenv("DATA_DIR", "./data")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
RAW_PATH = Path(DATA_DIR) / "raw.jsonl"
CHROMA_DIR = Path(DATA_DIR) / "faiss_index"

def load_raw():
    if not RAW_PATH.exists():
        return []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_raw(items):
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def normalize(record):
    text = (record.get("text") or "").strip()
    title = (record.get("title") or "").strip()
    uid = record.get("url") or (title[:120] if title else None)
    if not uid:
        uid = str(abs(hash(text)))
    return {
        "id": uid,
        "title": title or text[:120] or "Untitled",
        "text": text,
        "source": record.get("source") or "",
        "url": record.get("url") or "",
        "verdict": record.get("verdict") or ""
    }

def run_pipeline(limit_per_feed=40):
    existing = load_raw()
    new_items = []

    # 1. RSS FEEDS
    rss_items = fetch_all(limit_per_feed=limit_per_feed)

    # 2. GOOGLE FACT CHECK API (latest fact-checks)
    google_items = fetch_google_factchecks(page_size=20)

    combined = rss_items + google_items

    # Deduplicate
    seen = {it.get("id") for it in existing}
    for item in combined:
        uid = item.get("url") or item.get("title")
        if not uid:
            continue
        if uid not in seen:
            seen.add(uid)
            existing.append(item)
            new_items.append(item)

    save_raw(existing)

    # Normalize
    normalized = [normalize(r) for r in existing if (r.get("text") or "").strip()]

    ids = [n["id"] for n in normalized]
    docs = [n["text"] for n in normalized]
    metas = [
        {"title": n["title"], "source": n["source"], "url": n["url"], "verdict": n["verdict"]}
        for n in normalized
    ]

    # Rebuild FAISS
    FAISSIndexer().build_index(ids, docs, metas)

    return len(new_items)

