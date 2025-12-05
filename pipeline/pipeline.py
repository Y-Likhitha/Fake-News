
import os, json
from pathlib import Path
from dotenv import load_dotenv
from .rss_scraper import fetch_feeds
from .indexer import ChromaIndexer

load_dotenv()

DATA_DIR = "./data"
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR","./data/chroma_db")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
RAW_PATH = Path(DATA_DIR)/"raw.jsonl"

def load_raw():
    if not RAW_PATH.exists(): return []
    return [json.loads(x) for x in RAW_PATH.read_text(encoding="utf-8").splitlines()]

def save_raw(items):
    with open(RAW_PATH,"w",encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it,ensure_ascii=False)+"\n")

def normalize(r):
    text = (r.get("text") or "").strip()
    title = (r.get("title") or "").strip()
    uid = r.get("url") or title[:50] or str(abs(hash(text)))
    return {
        "id": uid,
        "title": title or "Untitled",
        "text": text,
        "source": r.get("source",""),
        "url": r.get("url",""),
        "verdict": r.get("verdict") or ""
    }

def run_pipeline(limit_per_feed=40):
    existing = load_raw()
    feeds = fetch_feeds(limit_per_feed=limit_per_feed)

    # dedupe
    seen=set(x.get("id") for x in existing)
    new=[]
    for f in feeds:
        n = normalize(f)
        if n["id"] not in seen:
            seen.add(n["id"])
            existing.append(n)
            new.append(n)

    save_raw(existing)

    ids=[x["id"] for x in existing if x["text"].strip()]
    docs=[x["text"] for x in existing if x["text"].strip()]
    metas=[{
        "title": x["title"],
        "source": x["source"],
        "url": x["url"],
        "verdict": x["verdict"]
    } for x in existing if x["text"].strip()]

    if ids:
        ix=ChromaIndexer(CHROMA_DIR)
        ix.add(ids,docs,metas)

    return len(new)
