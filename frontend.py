"""AgriScheme AI agriculture-themed Streamlit UI; communicates ONLY with local FastAPI.
Run: python -m streamlit run frontend.py
"""
from __future__ import annotations
import os
from pathlib import Path
from urllib.parse import quote
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent/'.env',override=False)
API=os.getenv('AGRISCHEME_API_URL','http://127.0.0.1:8000').rstrip('/')
st.set_page_config(page_title='AgriScheme AI • Tamil Nadu',page_icon='🌾',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
:root{--pine:#163f31;--ink:#1b3b2c;--cream:#faf8ed;--mint:#e8f3e9;--gold:#e4bb65}
.stApp,[data-testid="stAppViewContainer"]{background:linear-gradient(160deg,#f8f9ed 0%,#eef5e9 68%,#f7f4e5 100%);color:#183729}
[data-testid="stSidebar"]{background:#133b2d!important;color:#f0f6e9}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#f4f4e5!important}
h1,h2,h3{color:#183d2c!important}
.hero{background:linear-gradient(115deg,rgba(17,61,43,.98),rgba(35,89,50,.94));padding:35px;border-radius:23px;color:#fdf9e8;box-shadow:0 12px 35px rgba(19,61,43,.15);border-bottom:4px solid #cfa951}
.hero h1{color:#f9f6e7!important;font-size:2.6rem;letter-spacing:-.035em;margin:0}
.hero p{color:#d9e8d4!important;margin:8px 0 0;font-size:1.03rem}
.eyebrow{font-size:.8rem;color:#d8c783;letter-spacing:.2em;font-weight:700}
.metric{background:#fffef7;border:1px solid #d9e5d3;border-radius:16px;padding:20px;box-shadow:0 5px 18px rgba(30,70,32,.04)}
.metric strong{font-size:1.7rem;color:#17543b;display:block}
.metric span{color:#66816f;font-size:.85rem}
.note{background:#e2efdf;border-left:4px solid #55815d;padding:12px 16px;border-radius:9px;color:#355a42}
[data-testid="stChatInput"]{border:1px solid #85a689!important;background:#fffdf6!important;border-radius:17px!important;box-shadow:0 4px 15px rgba(16,59,37,.07)}
[data-testid="stChatInput"] textarea{color:#153f2c!important;-webkit-text-fill-color:#153f2c!important;background:#fffdf6!important;caret-color:#153f2c!important}
[data-testid="stChatInput"] textarea::placeholder{color:#6d8a72!important;-webkit-text-fill-color:#6d8a72!important;opacity:1!important}
[data-testid="stChatMessage"]{background:#fffef8;border:1px solid #e2e9da;border-radius:15px;padding:1rem;margin:.6rem 0}
.stButton button[kind="primary"]{background:#28734c!important;color:#fffaf0!important;border:0!important}
[data-testid="stSidebar"] .stButton button{border-color:#82a189}
[data-testid="stTextInput"] input{background:#fffef8!important;color:#183729!important;-webkit-text-fill-color:#183729!important}
[data-testid="stSelectbox"] input{color:#183729!important}
</style>''',unsafe_allow_html=True)

def api(method,path,**kw):
    try:
        r=requests.request(method,API+path,timeout=180,**kw)
        if r.status_code>=400:
            try: detail=r.json().get('detail',r.text)
            except Exception: detail=r.text
            raise RuntimeError(f'Backend HTTP {r.status_code}: {detail}')
        return r.json()
    except requests.ConnectionError:
        raise RuntimeError('Cannot connect to FastAPI. Start the backend in a separate terminal: python -m uvicorn backend:app --reload --port 8000')
    except requests.Timeout:
        raise RuntimeError('Backend request timed out. Try fewer or smaller PDF documents.')

def show_sources(sources):
    with st.expander(f'📚 Evidence & diagnostics ({len(sources)} passages)',expanded=False):
        st.caption('RRF scores indicate search ranking, not factual certainty. Citations refer only to your uploaded PDFs.')
        for s in sources:
            st.markdown(f"**[{s['source_id']}] {s['filename']} · Page {s['page']}**")
            st.caption(f"Chunk `{s['chunk_id']}` | RRF: {s['rrf_score']:.5f} | {'; '.join(s['retrieval_hits'])}")
            st.code(s['text'][:1600],language=None,wrap_lines=True)
            st.markdown(f"[Open saved PDF]({API}/documents/{quote(s['doc_id'])}/pdf#page={s['page']})")
            st.divider()

st.markdown('''<div class="hero"><div class="eyebrow">TAMIL NADU · AGRICULTURE INTELLIGENCE</div>
<h1>🌾 AgriScheme AI</h1><p>நம் விவசாயம். நம் வாய்ப்புகள். · Discover farmer schemes with verifiable evidence.</p>
<p style="font-size:.88rem;color:#c2d8bb!important">Private PDF library • Tamil & English • No web search • Persistent FAISS</p></div>''',unsafe_allow_html=True)
try:
    health=api('GET','/health'); connected=True
except Exception as e:
    health={'documents':0,'chunks':0,'index_ready':False,'api_key_configured':False}; connected=False
    st.error(str(e))

with st.sidebar:
    st.markdown('## 🌱 Knowledge Library')
    st.caption('Upload official schemes, policy notes, and government guideline PDFs.')
    if connected: st.success('● FastAPI connected')
    else: st.error('● Backend offline')
    if connected and not health['api_key_configured']: st.warning('OpenAI API key missing in backend .env')
    uploaded=st.file_uploader('Add PDF documents',type=['pdf'],accept_multiple_files=True)
    if st.button('🌿 Index selected PDFs',type='primary',use_container_width=True,disabled=(not connected or not uploaded)):
        try:
            payload=[('files',(f.name,f.getvalue(),'application/pdf')) for f in uploaded]
            with st.spinner('Reading PDFs and indexing only new content…'):
                response=api('POST','/documents/upload',files=payload)
            for item in response['results']:
                if item['status']=='indexed': st.success(f"{item['filename']}: {item['chunks']} chunks indexed")
                elif item['status']=='duplicate': st.info(f"{item['filename']}: already indexed")
                else: st.warning(f"{item['filename']}: {item.get('reason','skipped')}")
            st.rerun()
        except Exception as e: st.error(str(e))
    if connected:
        try: docs=api('GET','/documents')['documents']
        except Exception as e: docs=[]; st.error(str(e))
    else: docs=[]
    with st.expander(f'📂 Saved reports ({len(docs)})',expanded=False):
        if not docs: st.caption('No reports indexed yet')
        for d in docs:
            st.markdown(f"**{d['filename']}**")
            st.caption(f"{d['readable_pages']} readable pages · {d['chunks']} chunks")
            if st.button('Remove document',key='delete-'+d['id'],use_container_width=True):
                try:
                    api('DELETE','/documents/'+quote(d['id'])); st.rerun()
                except Exception as e: st.error(str(e))
            st.divider()
    st.markdown('### ⚙️ Search controls')
    language=st.selectbox('Answer language',['Auto','English','தமிழ்'])
    top_k=st.slider('Evidence passages',3,12,7)
    if st.button('Clear conversation',use_container_width=True):
        st.session_state.messages=[]; st.rerun()
    st.caption('🔒 Uploaded PDFs are the only knowledge source. OpenAI is used for embeddings, translation and response generation.')

c1,c2,c3=st.columns(3)
for col,value,label in zip((c1,c2,c3),(health['documents'],health['chunks'],'FAISS + BM25'),('Saved PDFs','Searchable passages','Hybrid search')):
    with col: st.markdown(f'<div class="metric"><strong>{value}</strong><span>{label}</span></div>',unsafe_allow_html=True)

st.markdown('### Ask about agricultural schemes')
st.markdown('<div class="note">Example: <b>கன்னியாகுமரியில் வாழை சாகுபடிக்கான மானியங்கள் என்ன?</b> &nbsp;·&nbsp; What irrigation assistance is described in my PDFs?</div>',unsafe_allow_html=True)
if 'messages' not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m['role']):
        st.markdown(m['content'])
        if m.get('sources'): show_sources(m['sources'])
question=st.chat_input('Ask about schemes in Tamil or English…',disabled=not connected)
if question:
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'): st.markdown(question)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Searching indexed PDFs and preparing evidence-based answer…'):
                result=api('POST','/chat',json={'question':question,'language':language,'top_k':top_k})
            answer=result['answer']; sources=result['sources']
            st.markdown(answer)
            if sources: show_sources(sources)
        except Exception as e:
            answer=f'Unable to answer: {e}'; sources=[]; st.error(answer)
    st.session_state.messages.append({'role':'assistant','content':answer,'sources':sources})
st.caption('⚠️ This is an educational research assistant. Subsidies, deadlines, and eligibility must be verified with the issuing government department. Offline PDF search does not establish current scheme availability.')
