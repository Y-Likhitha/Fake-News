import streamlit as st
from pipeline.query import QueryService
from pipeline.pipeline import run_pipeline
import os

st.set_page_config(page_title='Fake News Verifier', layout='wide')
st.title('Fake News Verifier — Real-time (Streamlit + Chroma)')

with st.sidebar:
            st.header('Settings')
            if st.button('Run pipeline (fetch & index)'):
                st.info('Running pipeline... this may take a minute')
                added = run_pipeline(update_google=True, update_altnews=True, update_factly=True, save_csv=True, build_index=False)
                st.success(f'Pipeline finished. New items added: {added}')
            model = st.text_input('Embedding model', value=os.getenv('EMBEDDING_MODEL','all-MiniLM-L6-v2'))
            k = st.slider('Top-k results', 1, 10, 5)
            threshold = st.slider('Match similarity threshold', 50, 95, 75) / 100.0

        st.markdown('Enter headline or claim to verify. The app will search the Chroma/embeddings index for matching fact-checks.')
        query = st.text_area('Claim / Text', height=120)
        if st.button('Check news'):
            if not query.strip():
                st.warning('Please enter some text to check.')
            else:
                qs = QueryService(model_name=model)
                res = qs.query_text(query, top_k=k, score_threshold=threshold)
                st.write('Decision:', res.get('decision'))
                st.write('Top matches:')
                for m in res.get('matches', []):
                    st.markdown(f"- **{m.get('title') or '—'}**  
Source: {m.get('source')} — Score: {m.get('score'):.3f}  
URL: {m.get('url')}")
