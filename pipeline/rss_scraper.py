
import feedparser
from datetime import datetime
import os

DEFAULT_FEEDS = [
    "https://factly.in/feed/",
    "https://www.altnews.in/feed/",
    "https://www.boomlive.in/feed/",
    "https://www.politifact.com/rss/whats-hot/"
]

def parse_feed(url):
    d = feedparser.parse(url)
    items=[]
    for e in d.entries:
        items.append({
            "source": url,
            "title": e.get("title",""),
            "text": e.get("summary","") or e.get("description",""),
            "url": e.get("link"),
            "verdict": None,
            "published_at": e.get("published") or e.get("updated"),
            "retrieved_at": datetime.utcnow().isoformat()
        })
    return items

def fetch_feeds(feeds=None, limit_per_feed=40):
    feeds = feeds or os.getenv("RSS_FEEDS") or ",".join(DEFAULT_FEEDS)
    if isinstance(feeds,str):
        feeds = [x.strip() for x in feeds.split(",") if x.strip()]
    all_items=[]
    for f in feeds:
        all_items.extend(parse_feed(f)[:limit_per_feed])
    return all_items
