import feedparser
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)

# Default RSS feeds, overridable via .env or pipeline
DEFAULT_FEEDS = [
    "https://factly.in/feed/",
    "https://www.altnews.in/feed/",
    "https://www.boomlive.in/feed/",
    "https://www.politifact.com/rss/whats-hot/"  # politifact example
]

def parse_feed(url):
    try:
        d = feedparser.parse(url)
        items = []
        for entry in d.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            link = entry.get("link")
            published = entry.get("published") or entry.get("updated")
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
        logging.exception("Failed to parse RSS feed: %s", url)
        return []

def fetch_feeds(feeds=None, limit_per_feed=50):
    feeds = feeds or os.getenv("RSS_FEEDS") or ",".join(DEFAULT_FEEDS)
    if isinstance(feeds, str):
        feeds = [f.strip() for f in feeds.split(",") if f.strip()]
    result = []
    for f in feeds:
        items = parse_feed(f)[:limit_per_feed]
        result.extend(items)
    return result
