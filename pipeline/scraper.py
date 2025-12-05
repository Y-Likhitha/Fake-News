import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------
# GOOGLE FACT CHECK API
# ---------------------------------------------------------
def fetch_google_factchecks(api_key, page_size=50):
    if not api_key:
        logging.warning("Missing GOOGLE_FACTCHECK_API_KEY")
        return []

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"key": api_key, "pageSize": page_size}

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        out = []
        for c in data.get("claims", []):
            review = (c.get("claimReview") or [{}])[0]
            verdict = review.get("textualRating")

            out.append({
                "source": "google_factcheck",
                "claim": c.get("text"),
                "verdict": verdict,
                "url": review.get("url"),
                "publisher": review.get("publisher", {}).get("name"),
                "claimDate": c.get("claimDate"),
                "retrieved_at": datetime.utcnow().isoformat()
            })

        return out

    except Exception:
        logging.exception("Google Fact Check API failed")
        return []


# ---------------------------------------------------------
# ALTNEWS SCRAPER
# ---------------------------------------------------------
def scrape_altnews(limit=30):
    base = "https://www.altnews.in"
    url = base + "/category/fact-checks/"
    results = []

    try:
        pg = requests.get(url, timeout=15)
        pg.raise_for_status()

        soup = BeautifulSoup(pg.text, "html.parser")
        posts = soup.select("article")

        for post in posts[:limit]:
            a = post.find("a", href=True)
            if not a:
                continue

            title = a.get_text(strip=True)
            link = a["href"]

            article = requests.get(link, timeout=15)
            sp = BeautifulSoup(article.text, "html.parser")

            content = sp.select_one(".entry-content")
            text = content.get_text("\n", strip=True) if content else ""

            results.append({
                "source": "altnews",
                "title": title,
                "text": text,
                "url": link,
                "verdict": "fact-check",
                "retrieved_at": datetime.utcnow().isoformat()
            })

        return results

    except Exception:
        logging.exception("AltNews scraping failed")
        return []


# ---------------------------------------------------------
# FACTLY SCRAPER
# ---------------------------------------------------------
def scrape_factly(limit=30):
    base_url = "https://factly.in/category/fact-check/"
    results = []

    try:
        pg = requests.get(base_url, timeout=15)
        pg.raise_for_status()

        soup = BeautifulSoup(pg.text, "html.parser")
        posts = soup.select("article")

        for post in posts[:limit]:
            a = post.find("a", href=True)
            if not a:
                continue

            title = a.get_text(strip=True)
            link = a["href"]

            article = requests.get(link, timeout=15)
            sp = BeautifulSoup(article.text, "html.parser")

            content = sp.select_one(".entry-content")
            text = content.get_text("\n", strip=True) if content else ""

            results.append({
                "source": "factly",
                "title": title,
                "text": text,
                "url": link,
                "verdict": "fact-check",
                "retrieved_at": datetime.utcnow().isoformat()
            })

        return results

    except Exception:
        logging.exception("Factly scraping failed")
        return []
