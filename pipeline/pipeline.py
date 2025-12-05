import os, json, logging
from .scraper import fetch_google_factchecks, scrape_altnews_recent, scrape_factly_recent
from .indexer import ChromaIndexer
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
logging.basicConfig(level=logging.INFO)
DATA_DIR = os.getenv('DATA_DIR','./data')
CHROMA_DIR = os.getenv('CHROMA_PERSIST_DIR','./data/chroma_db')
GOOGLE_KEY = os.getenv('GOOGLE_FACTCHECK_API_KEY','')
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

RAW_JSON = os.path.join(DATA_DIR,'factchecks_raw.jsonl')

def _load_raw():
    if not os.path.exists(RAW_JSON):
        return []
    out = []
    with open(RAW_JSON,'r', encoding='utf-8') as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except:
                pass
    return out

def _save_raw(items):
    with open(RAW_JSON,'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

def normalize_record(rec):
    txt = rec.get('text') or rec.get('claim') or ''
    title = rec.get('title') or rec.get('claim') or ''
    _id = rec.get('url') or rec.get('claim') or f"{rec.get('source')}_{abs(hash(txt)) % (10**10)}"
    return {'id': _id, 'source': rec.get('source'), 'url': rec.get('url'), 'title': title, 'text': txt, 'published_at': rec.get('published_at') or rec.get('claimDate')}

def append_unique(existing, new, key='id'):
    seen = {e.get(key) for e in existing}
    added = []
    for n in new:
        k = n.get(key) or n.get('url') or n.get('claim')
        if k not in seen:
            existing.append(n)
            seen.add(k)
            added.append(n)
    return existing, added

def run_pipeline(update_google=True, update_altnews=True, update_factly=True, save_csv=True, build_index=True):
    existing = _load_raw()
    all_new = []
    if update_google:
        g = fetch_google_factchecks(GOOGLE_KEY)
        existing, added = append_unique(existing, g, key='claim')
        all_new.extend(added)
    if update_altnews:
        a = scrape_altnews_recent(limit=40)
        existing, added = append_unique(existing, a, key='url')
        all_new.extend(added)
    if update_factly:
        f = scrape_factly_recent(limit=40)
        existing, added = append_unique(existing, f, key='url')
        all_new.extend(added)
    # save raw
    _save_raw(existing)
    # normalize and index into Chroma
    indexer = ChromaIndexer(persist_dir=CHROMA_DIR)
    normalized = [normalize_record(r) for r in existing]
    texts = [n['text'] or n['title'] for n in normalized]
    ids = [n['id'] for n in normalized]
    metadatas = [{'title': n['title'], 'source': n['source'], 'url': n['url'], 'published_at': n.get('published_at')} for n in normalized]
    indexer.add(ids, texts, metadatas)
    return len(all_new)
