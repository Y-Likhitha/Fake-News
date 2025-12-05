import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


# -----------------------------
# GOOGLE FACT CHECK API
# -----------------------------
def fetch_google_factchecks(api_key, page_size=50):
    if not api_key:
        logging.warning("No GOOGLE_FACTCHECK_API_KEY provided.")
        return []

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"key": api_key, "pageSize": page_size}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        out = []
        for c in data.get("claims", []):
            review = c.get("claimReview", [{}])[0]
            verdict = review.get("textualRating")

            out.append({
                "source": "google_factcheck",
                "claim": c.get("text"),
                "verdict": verdict,
                "publisher": review.get("publisher", {}).get("name"),
                "url": review.get("url"),
                "claimDate": c.get("claimDate"),
                "retrieved_at": datetime.utcnow().isoformat()
            })

        return out
    except Exception:
        logging.exception("Google Fact Check API failed")
        return []


# ---------------------------------
# SCRAPER FOR ALTNEWS
# ---------------------------------
def scrape_altnews(limit=30):
    base = "https://www.altnews.in"
    url = base + "/category/fact-checks/"
    results = []

    try:
        page = requests.get(url, timeout=15)
        soup = BeautifulSoup(page.text, "html.parser")
        posts = soup.select("article")

        for post in posts[:limit]:
            a = post.find("a", href=True)
            if not a: continue

            title = post.text.strip()
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
                "retrieved_at": datetime.utcnow().isoformat(),
                "verdict": "fact-check"  # label placeholder
            })

        return results

    except Exception:
        logging.exception("AltNews scraping failed")
        return []


# ---------------------------------
# SCRAPER FOR FACTLY
# ---------------------------------
def scrape_factly(limit=30):
    base = "https://factly.in/category/fact-check/"
    results = []

    try:
        pg = requests.get(base, timeout=15)
        soup = BeautifulSoup(pg.text,"html.parser")
