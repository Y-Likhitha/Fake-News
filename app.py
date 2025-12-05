import streamlit as st
from pipeline.query import QueryService
from pipeline.pipeline import run_pipeline
import os

st.set_page_config(page_title="Fake News Verifier", layout="wide")

st.markdown(
    """<style>
    body {background-color:#000000;}
    .stApp {background:#000000; color:#00ff00;}
    h1,h2,h3,h4,h5,h6,p,div,span {color:#00ff00 !important;}
    .stTextInput>div>div>input {background:#001100; color:#00ff00;}
    .stTextArea textarea {background:#001100; color:#00ff00;}
    .stButton>button {background:#003300; color:#00ff00; border:1px solid #00ff00;}
    </style>""", unsafe_allow_html=True
)

st.title("🕵️ Fake News Verifier — Dark/Hacker Edition")

with st.sidebar:
    st.header("Controls")
    if st.button("Run Pipeline (Fetch + Index)"):
        st.info("Running pipeline…")
        added = run_pipeline()
        st.success(f"Added {added} new fact-check records.")

    model = st.text_input("Embedding model", value=os.getenv("EMBEDDING_MODEL","all-mpnet-base-v2"))
    k = st.slider("Top-k results", 1, 10, 5)
    threshold = st.slider("Score threshold", 40, 95, 70) / 100.0

st.subheader("Enter a claim to verify")
query = st.text_area("Your Claim", height=120)

if st.button("Check"):
    if not query.strip():
        st.warning("Enter a claim first.")
    else:
        qs = QueryService(model_name=model)
        res = qs.query_text(query, top_k=k, score_threshold=threshold)
        st.subheader(f"Decision: **{res.get('decision')}**")

        if res.get("matches"):
            for m in res.get("matches", []):
                st.markdown(
                    f"""<div style='padding:10px;border:1px solid #00ff00;margin:5px;'>
                    <b>{m.get('title','—')}</b><br>
                    Source: {m.get('source','?')}<br>
                    Verdict: {m.get('verdict','Unknown')}<br>
                    Score: {m.get('score'):.3f}<br>
                    URL: <a href='{m.get('url')}' style='color:#00ff00;'>{m.get('url')}</a>
                    </div>""", unsafe_allow_html=True
                )
        else:
            st.info('No high-similarity fact-check found.')