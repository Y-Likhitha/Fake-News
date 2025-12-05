import os, json
from pathlib import Path
from scraper import fetch_all
from indexer import ChromaIndexer


DATA_DIR = os.getenv('DATA_DIR','./data')
CHROMA_DIR = os.getenv('CHROMA_PERSIST_DIR','./data/chroma_db')
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
RAW_PATH = Path(DATA_DIR)/'raw.jsonl'

def load_raw():
    if not RAW_PATH.exists():
        return []
    with open(RAW_PATH,'r',encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_raw(items):
    with open(RAW_PATH,'w',encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it,ensure_ascii=False)+'\n')

def normalize(r):
    text = (r.get('text') or '').strip()
    title = (r.get('title') or '').strip()
    uid = r.get('url') or (title[:120] if title else None)
    if not uid:
        uid = str(abs(hash(text)))
    return {'id': uid, 'title': title or text[:120] or 'Untitled', 'text': text, 'source': r.get('source') or '', 'url': r.get('url') or '', 'verdict': r.get('verdict') or ''}

def run_pipeline(limit_per_feed=40):
    existing = load_raw()
    feeds = fetch_all(limit_per_feed=limit_per_feed)
    seen = {x.get('id') for x in existing}
    new = []
    for f in feeds:
        n = normalize(f)
        if n['id'] not in seen:
            seen.add(n['id'])
            existing.append(n)
            new.append(n)
    save_raw(existing)
    normalized = [normalize(x) for x in existing]
    ids = [n['id'] for n in normalized if n['text'].strip()]
    docs = [n['text'] for n in normalized if n['text'].strip()]
    metas = [{'title': n['title'], 'source': n['source'], 'url': n['url'], 'verdict': n['verdict']} for n in normalized if n['text'].strip()]
    if ids:
        idx = ChromaIndexer(persist_dir=CHROMA_DIR)
        idx.add(ids, docs, metas)
    return len(new)
