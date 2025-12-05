import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)

def fetch_google_factchecks(api_key, page_size=50):
    if not api_key:
        logging.warning('No Google API key provided.')
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"key": api_key, "pageSize": page_size}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        claims = data.get('claims', [])
        out = []
        for c in claims:
            out.append({
                'source': 'google_factcheck',
                'claim': c.get('text'),
                'claimDate': c.get('claimDate'),
                'claimReview': c.get('claimReview', []),
                'retrieved_at': datetime.utcnow().isoformat()
            })
        return out
    except Exception:
        logging.exception('Google factcheck fetch failed')
        return []

def scrape_altnews_recent(limit=30):
    base = "https://www.altnews.in"
    listing = base + "/category/fact-checks/"
    items = []
    try:
        r = requests.get(listing, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        posts = soup.select('article')
        count = 0
        for post in posts:
            if count >= limit:
                break
            a = post.find('a', href=True)
            href = a['href'] if a else None
            title = (post.find(['h2','h3']).get_text(strip=True) if post.find(['h2','h3']) else '')
            if not href:
                continue
            if not href.startswith('http'):
                href = base + href
            rr = requests.get(href, timeout=15)
            sp = BeautifulSoup(rr.text, 'html.parser')
            content = (sp.select_one('.entry-content') or sp.body).get_text(separator='\n', strip=True)
            date_tag = sp.find('time')
            date_str = date_tag.get('datetime') if date_tag else None
            items.append({
                'source': 'altnews',
                'url': href,
                'title': title,
                'text': content,
                'published_at': date_str,
                'retrieved_at': datetime.utcnow().isoformat()
            })
            count += 1
            time.sleep(0.3)
    except Exception:
        logging.exception('AltNews scraper failed')
    return items

def scrape_factly_recent(limit=30):
    base = "https://factly.in"
    listing = base + "/category/fact-check/"
    items = []
    try:
        r = requests.get(listing, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        posts = soup.select('article')
        count = 0
        for post in posts:
            if count >= limit:
                break
            a = post.find('a', href=True)
            href = a['href'] if a else None
            title = (post.find(['h2','h3']).get_text(strip=True) if post.find(['h2','h3']) else '')
            if not href:
                continue
            if not href.startswith('http'):
                href = base + href
            rr = requests.get(href, timeout=15)
            sp = BeautifulSoup(rr.text, 'html.parser')
            content = (sp.select_one('.entry-content') or sp.body).get_text(separator='\n', strip=True)
            date_tag = sp.find('time')
            date_str = date_tag.get('datetime') if date_tag else None
            items.append({
                'source': 'factly',
                'url': href,
                'title': title,
                'text': content,
                'published_at': date_str,
                'retrieved_at': datetime.utcnow().isoformat()
            })
            count += 1
            time.sleep(0.3)
    except Exception:
        logging.exception('Factly scraper failed')
    return items
