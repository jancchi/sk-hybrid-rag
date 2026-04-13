"""Minimal Streamlit frontend for streaming answers."""

import json
import time

import httpx
import streamlit as st

API_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 420.0

st.set_page_config(page_title="SK Asistent", page_icon="⚖️", layout="centered")
st.title("⚖️ SK Právny a Finančný Asistent")
st.caption("Streamovaná odpoveď z backendu. Generovanie trvá približne 2 minúty.")

try:
    health = httpx.get(f"{API_URL}/health", timeout=3.0)
    health.raise_for_status()
    passage_count = health.json().get("qdrant_passages", 0)
    st.success(f"Backend je dostupný. Pasáže v databáze: {passage_count:,}")
except Exception:
    st.error(
        "Backend nie je dostupný. Spusti: `uvicorn app.main:app --reload --port 8000`"
    )

question = st.text_area(
    "Otázka",
    height=120,
    placeholder="Napr. Aké sú podmienky pre daňový bonus na dieťa?",
)
submit = st.button("Spýtať sa", type="primary", use_container_width=True)


def iter_sse_events(response: httpx.Response):
    current_event = "message"
    data_lines: list[str] = []

    for raw_line in response.iter_lines():
        if raw_line is None:
            continue

        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield current_event, "\n".join(data_lines)
                data_lines = []
                current_event = "message"
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())

    if data_lines:
        yield current_event, "\n".join(data_lines)


if submit:
    if not question.strip():
        st.warning("Zadaj otázku.")
        st.stop()

    start = time.perf_counter()
    payload: dict = {}
    status = st.status("Pripravujem odpoveď...", expanded=False)
    st.info("Prosím počkaj. Odpoveď sa streamuje a môže trvať približne 2 minúty.")
    st.subheader("Odpoveď")

    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/ask-stream",
            json={
                "question": question.strip(),
                "top_k_retrieve": 20,
                "top_k_rerank": 5,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()

            def token_stream():
                token_stream.done = False
                for event_name, body in iter_sse_events(response):
                    event_data = json.loads(body)

                    if event_name == "meta":
                        payload.update(event_data)
                        status.update(label="Generujem odpoveď...", state="running")
                    elif event_name == "token":
                        text = event_data.get("text", "")
                        if text:
                            yield text
                    elif event_name == "done":
                        payload.update(event_data)
                        token_stream.done = True
                    elif event_name == "error":
                        raise RuntimeError(event_data.get("message", "Chyba streamu"))

            rendered = st.write_stream(token_stream)
            if not getattr(token_stream, "done", False):
                raise RuntimeError("Stream sa ukončil predčasne.")

        total_ms = (time.perf_counter() - start) * 1000
        if not payload.get("answer") and isinstance(rendered, str):
            payload["answer"] = rendered
        status.update(label="Hotovo", state="complete")

    except httpx.HTTPStatusError as exc:
        status.update(label="Chyba API", state="error")
        st.error(f"API chyba: {exc.response.status_code} — {exc.response.text}")
        st.stop()
    except json.JSONDecodeError:
        status.update(label="Neplatná odpoveď", state="error")
        st.error("Backend vrátil neplatné JSON dáta v streame.")
        st.stop()
    except Exception as exc:
        status.update(label="Chyba", state="error")
        st.error(f"Neočakávaná chyba: {exc}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Retrieval", f"{payload.get('retrieval_ms', 0):.0f} ms")
    col2.metric("Reranking", f"{payload.get('rerank_ms', 0):.0f} ms")
    col3.metric("Generovanie", f"{payload.get('generation_ms', 0):.0f} ms")
    col4.metric("Celkovo", f"{total_ms:.0f} ms")

    if payload.get("sources"):
        st.subheader("Zdroje")
        for i, src in enumerate(payload["sources"], 1):
            title = src.get("title", "Bez názvu")
            score = src.get("score", 0)
            context = src.get("context", "")
            with st.expander(f"[{i}] {title} · skóre: {score:.4f}"):
                st.write(context)
