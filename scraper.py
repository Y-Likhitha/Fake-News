# scraper.py
import feedparser
from datetime import datetime
import os

DEFAULT_FEEDS = [
    "https://factly.in/feed/",
    "https://www.altnews.in/feed/",
    "https://www.boomlive.in/feed/",
    "https://www.politifact.com/rss/whats-hot/"
]

def parse_feed(url, limit=50):
    try:
        d = feedparser.parse(url)
        items = []
        for e in d.entries[:limit]:
            title = e.get("title","").strip()
            summary = e.get("summary","") or e.get("description","") or ""
            link = e.get("link")
            published = e.get("published") or e.get("updated")
            items.append({
                "source": url,
                "title": title,
                "text": summary,
                "url": link,
                "verdict": None,
                "published_at": published,
                "retrieved_at": datetime.utcnow().isoformat()
            })
        return items
    except Exception:
        return []

def fetch_all(feeds=None, limit_per_feed=50):
    feeds = feeds or os.getenv("RSS_FEEDS") or ",".join(DEFAULT_FEEDS)
    if isinstance(feeds, str):
        feeds = [f.strip() for f in feeds.split(",") if f.strip()]
    all_items = []
    for f in feeds:
        all_items.extend(parse_feed(f, limit=limit_per_feed))
    return all_items
