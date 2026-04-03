"""
SK Legal Assistant — Streamlit Frontend
========================================
Run with: streamlit run frontend/app.py
Requires the FastAPI backend to be running on localhost:8000.
"""

import time
import httpx
import streamlit as st

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SK Právny Asistent",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ SK Právny a Finančný Asistent")
st.caption("Powered by BGE-M3 · Qdrant Hybrid Search · Cross-Encoder · Qwen2.5:7b")

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Nastavenia pipeline")
    top_k_retrieve = st.slider(
        "Počet kandidátov (retrieval)", min_value=10, max_value=100, value=20, step=5
    )
    top_k_rerank = st.slider(
        "Top-K po rerankingu", min_value=1, max_value=10, value=5
    )
    show_sources = st.checkbox("Zobraziť zdrojové pasáže", value=True)
    show_timing = st.checkbox("Zobraziť časovanie pipeline", value=True)

    st.divider()
    st.subheader("Príklady otázok")
    example_questions = [
        "Čo je daň z príjmu fyzickej osoby?",
        "Aké sú podmienky pre daňový bonus na dieťa?",
        "Čo je to obchodná spoločnosť?",
        "Aké sú práva zamestnanca pri ukončení pracovného pomeru?",
    ]
    for eq in example_questions:
        if st.button(eq, use_container_width=True):
            st.session_state["prefill"] = eq

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
try:
    health = httpx.get(f"{API_URL}/health", timeout=3.0).json()
    passage_count = health.get("qdrant_passages", 0)
    if passage_count == 0:
        st.error(
            "⚠️ Qdrant databáza je prázdna. Spusti najprv: `python scripts/ingest.py`"
        )
    else:
        st.success(f"✅ Backend beží · {passage_count:,} pasáží v databáze")
except Exception:
    st.error("❌ Backend API nie je dostupný. Spusti: `uvicorn app.main:app --reload`")

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
prefill = st.session_state.pop("prefill", "")
question = st.text_area(
    "Vaša otázka (po slovensky)",
    value=prefill,
    height=100,
    placeholder="Napr. Aké sú podmienky na oslobodenie od dane z predaja nehnuteľnosti?",
)

submit = st.button("🔍 Hľadať odpoveď", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Pipeline call
# ---------------------------------------------------------------------------
if submit and question.strip():
    with st.spinner("Hľadám relevantné pasáže a generujem odpoveď..."):
        t_start = time.perf_counter()
        try:
            response = httpx.post(
                f"{API_URL}/ask",
                json={
                    "question": question.strip(),
                    "top_k_retrieve": top_k_retrieve,
                    "top_k_rerank": top_k_rerank,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            total_ms = (time.perf_counter() - t_start) * 1000

        except httpx.HTTPStatusError as e:
            st.error(f"API chyba: {e.response.status_code} — {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Neočakávaná chyba: {e}")
            st.stop()

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📋 Odpoveď")
    st.markdown(data["answer"])

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    if show_timing:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Retrieval", f"{data['retrieval_ms']:.0f} ms")
        col2.metric("Reranking", f"{data['rerank_ms']:.0f} ms")
        col3.metric("Generovanie", f"{data['generation_ms']:.0f} ms")
        col4.metric("Celkovo", f"{total_ms:.0f} ms")

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    if show_sources and data.get("sources"):
        st.divider()
        st.subheader("📚 Zdrojové pasáže")
        for i, src in enumerate(data["sources"], 1):
            with st.expander(f"[{i}] {src['title']}  ·  skóre: {src['score']:.4f}"):
                st.markdown(src["context"])

elif submit and not question.strip():
    st.warning("Prosím, zadaj otázku.")
