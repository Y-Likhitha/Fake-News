# google_api.py
import os
import requests
from datetime import datetime

# Load API key correctly
API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


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

    # Optional: add query text
    if query:
        params["query"] = query

    try:
        r = requests.get(ENDPOINT, params=params, timeout=10)
        data = r.json()

        items = []
        for c in data.get("claims", []):
            claim = c.get("text", "")
            review = c.get("claimReview", [{}])[0]

            items.append({
                "title": review.get("title") or claim[:120],
                "text": claim,
                "source": review.get("publisher", {}).get("name", ""),
                "url": review.get("url", ""),
                "verdict": review.get("textualRating", ""),
                "published_at": review.get("reviewDate", ""),
                "retrieved_at": datetime.utcnow().isoformat(),
            })

        return items

    except Exception as e:
        print("Google API error:", e)
        return []
