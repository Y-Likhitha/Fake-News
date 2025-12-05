# google_api.py
import os
import requests
from datetime import datetime

API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

ENDPOINT = (
    "https://factchecktools.googleapis.com/v1alpha1/claims:search"
)


def fetch_google_factchecks(query=None, lang="en", page_size=20):
    """
    Fetch fact-checks using Google Fact Check Tools API.
    If query=None, returns latest fact-checks.
    """
    if not API_KEY:
        print("⚠ GOOGLE_FACTCHECK_API_KEY not set. Skipping Google API.")
        return []

    params = {
        "key": API_KEY,
        "pageSize": page_size,
        "languageCode": lang
    }

    # optional: query specific text
    if query:
        params["query"] = query

    try:
        r = requests.get(ENDPOINT, params=params, timeout=10)
        data = r.json()

        items = []
        for c in data.get("claims", []):
            claim = c.get("text", "")
            claim_url = c.get("claimReview", [{}])[0].get("url", "")
            source = c.get("claimReview", [{}])[0].get("publisher", {}).get("name", "")
            title = c.get("claimReview", [{}])[0].get("title", "")
            verdict = c.get("claimReview", [{}])[0].get("textualRating", "")
            date = c.get("claimReview", [{}])[0].get("reviewDate", "")

            items.append({
                "title": title or claim[:120],
                "text": claim,
                "source": source,
                "url": claim_url,
                "verdict": verdict,
                "published_at": date,
                "retrieved_at": datetime.utcnow().isoformat()
            })
        return items

    except Exception as e:
        print("Google API error:", e)
        return []
