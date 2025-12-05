import os, json
from pathlib import Path
from dotenv import load_dotenv
from .rss_scraper import fetch_feeds
from .indexer import ChromaIndexer

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
RAW_PATH = Path(DATA_DIR) / "raw.jsonl"

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

def append_unique(existing, new_items):
    seen = {it.get("id") for it in existing}
    added = []
    for n in new_items:
        uid = n.get("url") or n.get("title") or None
        if not uid:
            continue
        if uid not in seen:
            n["id"] = uid
            existing.append(n)
            added.append(n)
            seen.add(uid)
    return existing, added

def run_pipeline(limit_per_feed=50):
    existing = load_raw()
    new_items = []

    feeds = fetch_feeds(limit_per_feed=limit_per_feed)
    existing, added = append_unique(existing, feeds)
    new_items.extend(added)

    save_raw(existing)

    normalized = [normalize(r) for r in existing]

    ids = [n["id"] for n in normalized]
    texts = [n["text"] for n in normalized]
    metas = [
        {"title": n["title"], "source": n["source"], "url": n["url"], "verdict": n["verdict"]}
        for n in normalized
    ]

    # filter only those with non-empty text
    clean_ids, clean_texts, clean_metas = [], [], []
    for i, t, m in zip(ids, texts, metas):
        if isinstance(t, str) and t.strip():
            clean_ids.append(i)
            clean_texts.append(t.strip())
            # ensure no None in metadata values
            clean_metas.append({k: ("" if v is None else v) for k, v in m.items()})


    indexer = ChromaIndexer(persist_dir=CHROMA_DIR)
    if clean_ids:
        indexer.add(clean_ids, clean_texts, clean_metas)
    else:
        print("No valid documents to index (RSS fetch may have failed or returned empty).")

    return len(new_items)
