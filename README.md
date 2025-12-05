Fake News Verifier - Simple Structure (mpnet)
============================================
Files:
- app.py            : Streamlit UI
- scraper.py        : RSS scraper (feedparser)
- indexer.py        : Chroma indexer (embeds with SentenceTransformer)
- query_engine.py   : Query wrapper (similarity conversion + filtering)
- pipeline.py       : Run full pipeline (fetch -> index)
- requirements.txt  : Python deps
- .env.example      : env template

Quick start:
1. Copy .env.example to .env and edit if needed.
2. pip install -r requirements.txt
3. python -c "import pipeline; print(pipeline.run_pipeline())"
4. streamlit run app.py
