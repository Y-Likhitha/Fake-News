import streamlit as st
import os
from pipeline import run_pipeline
from query_engine import QueryEngine

st.set_page_config(page_title='Fake News Verifier', layout='wide')

# ---- Dark theme ----
st.markdown(
    """
    <style>
    body {background:#000; color:#00ff00}
    .stApp {background:#000;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title('🕵️ Fake News Verifier — Dark (mpnet)')

# ---- SIDEBAR ----
with st.sidebar:
    st.header('Controls')

    if st.button('Run pipeline (fetch & index)'):
        st.info('Running pipeline... this may take a minute.')
        added = run_pipeline()
        st.success(f'Pipeline finished. New items added: {added}')

    st.markdown('---')

    k = st.slider('Top-k results', 1, 10, 5)
    threshold = st.slider('Similarity threshold', 30, 95, 70) / 100.0


# ---- MAIN UI ----
st.markdown('Enter a claim or headline to verify:')
query = st.text_area('Claim / Text', height=140)

if st.button('Check news'):
    if not query.strip():
        st.warning('Please enter a claim.')
    else:
        qe = QueryEngine()
        res = qe.query_text(query, top_k=k, score_threshold=threshold)

        st.subheader(f"Decision: {res.get('decision')}")

        if res.get('matches'):
            for m in res['matches']:
                st.markdown(f"**{m.get('title','—')}**  \nSource: {m.get('source')} — Verdict: {m.get('verdict','Unknown')} — Similarity: {m.get('score'):.3f}  \nURL: {m.get('url')}")
        else:
            st.info('No close fact-check found.')
