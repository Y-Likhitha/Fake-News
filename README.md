# Fake News Verifier (Streamlit + Chroma)
Fast real-time fact-check verification project.

Defaults:
- Embedding model: all-MiniLM-L6-v2
- Vector DB: Chroma (persisted under data/chroma_db)
- UI: Professional Streamlit UI (sidebar settings, results area)

Quick start:
1. Create and activate a Python virtualenv.
2. Copy `.env.example` to `.env` and set GOOGLE_FACTCHECK_API_KEY.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the pipeline once to build the DB:
   ```
   python -m pipeline.pipeline --run-once
   ```
5. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

Notes:
- Respect scraping terms & robots.txt for AltNews / Factly.
- Chroma will be created under `data/chroma_db`.
- For production, replace simple scrapers with robust RSS/API ingestion.
