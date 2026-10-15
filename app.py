"""Streamlit UI: access-key gate + chat over the CV."""
import hmac
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

import rag
from config import Settings, load_settings

st.set_page_config(page_title="Chat with My CV", page_icon="💬", layout="centered", initial_sidebar_state="expanded")

def _css(name: str) -> str:
    return (Path(__file__).parent / name).read_text()


st.markdown(f"<style>{_css('style.css')}</style>", unsafe_allow_html=True)
if st.session_state.get("dark"):
    st.markdown(f"<style>{_css('style_dark.css')}</style>", unsafe_allow_html=True)

EXAMPLES = [
    "Give me a 3-sentence summary of this profile.",
    "What is the most recent job and what were the main responsibilities?",
    "Which projects or awards stand out?",
]


@st.cache_resource
def _attempt_tracker() -> dict:
    """Server-wide failed-attempt state, shared by all sessions (so reloading the page
    does not reset it). Resets when the server process restarts."""
    return {"failures": 0, "locked_until": 0.0}


@st.cache_resource
def _daily() -> dict:
    """Server-wide question count for today, shared by all sessions. Resets each day and on restart."""
    return {"day": "", "count": 0}


def daily_used() -> int:
    d = _daily()
    today = datetime.now().strftime("%Y-%m-%d")
    if d["day"] != today:
        d["day"], d["count"] = today, 0
    return d["count"]


def questions_left(s: Settings) -> int:
    """Quota lives in the browser session: a page refresh starts a fresh one."""
    return max(0, s.max_questions - st.session_state.get("asked", 0))


@st.cache_resource(show_spinner=False)
def _client(api_key: str):
    return rag.make_client(api_key)


@st.cache_resource(show_spinner=False)
def _index(api_key: str, cv_path: str, mtime: float, emb_model: str, size: int, overlap: int):
    """Cached per (CV file version, model, chunk settings). Not re-run on every question."""
    s = load_settings()
    return rag.CVIndex.build(_client(api_key), s)


def get_index(s: Settings):
    mtime = s.cv_path.stat().st_mtime if s.cv_path.exists() else 0.0
    return _index(s.gemini_api_key, str(s.cv_path), mtime, s.embedding_model, s.chunk_size, s.chunk_overlap)


def gate(s: Settings) -> bool:
    """Render the access screen. Returns True once the visitor is authorised."""
    if st.session_state.get("authed"):
        return True

    tracker = _attempt_tracker()
    st.markdown('<div class="badge">🔒</div>', unsafe_allow_html=True)
    st.title("Chat with My CV")
    st.markdown('<p class="subtle">This assistant is private. Enter the access key to continue.</p>', unsafe_allow_html=True)

    remaining = tracker["locked_until"] - time.time()
    if remaining > 0:
        st.error(f"Too many failed attempts. Try again in {int(remaining) + 1} seconds.")
        return False

    with st.form("gate"):
        key = st.text_input("Access key", type="password")
        submitted = st.form_submit_button("Unlock chat  →", use_container_width=True)
    if submitted:
        if hmac.compare_digest(key.encode(), s.app_access_key.encode()):
            tracker["failures"] = 0
            st.session_state["authed"] = True
            with st.spinner("Unlocking… preparing the CV (first time only)"):
                try:
                    get_index(s)  # warm the cache so the chat opens instantly
                except rag.CVError:
                    pass  # chat() shows the error message
            st.rerun()
        tracker["failures"] += 1
        if tracker["failures"] >= s.max_failed_attempts:
            tracker["locked_until"] = time.time() + s.lockout_seconds
            tracker["failures"] = 0
            st.error(f"Too many failed attempts. Locked for {s.lockout_seconds} seconds.")
        else:
            st.error("Incorrect access key.")
    st.toggle("Dark mode", key="dark")
    return False


def sidebar(s: Settings) -> bool:
    with st.sidebar:
        st.header("Options")
        show_sources = st.toggle("Show CV excerpts used", value=True)
        st.toggle("Dark mode", key="dark")
        messages = st.session_state.get("messages", [])
        st.download_button(
            "Download chat (.txt)",
            data=rag.chat_to_text(messages),
            file_name=f"cv-chat-{datetime.now():%Y%m%d-%H%M}.txt",
            mime="text/plain",
            disabled=not messages,
            use_container_width=True,
        )
        if st.button("Reset chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
        if s.show_admin_tools and st.button("Test Gemini connection", use_container_width=True):
            with st.spinner("Calling Gemini…"):
                try:
                    st.success(rag.check_connection(_client(s.gemini_api_key), s))
                except rag.CVError as exc:
                    st.error(str(exc))
        if st.button("Log out", use_container_width=True):
            for k in ("authed", "messages"):  # keep the question count so logging out cannot reset the quota
                st.session_state.pop(k, None)
            st.rerun()
        st.caption(f"Questions left: **{questions_left(s)}** of {s.max_questions}")
        st.caption(f"Chat model: `{s.chat_model}`  \nFallback: `{s.fallback_model or 'off'}`  \nEmbeddings: `{s.embedding_model}`")
    return show_sources


def render_sources(hits: list[dict]) -> None:
    if not hits:
        return
    with st.expander(f"CV excerpts used ({len(hits)})"):
        for h in hits:
            st.markdown(f"**{h['section'].title()}** · similarity {h['score']:.2f}")
            st.text(h["text"])


def chat(s: Settings) -> None:
    show_sources = sidebar(s)
    st.markdown('<div class="badge">💬</div>', unsafe_allow_html=True)
    st.title("Chat with My CV")
    st.markdown(
        '<p class="subtle">Ask about skills, experience, education, projects or awards. '
        "Answers are drawn only from the CV itself.</p>",
        unsafe_allow_html=True,
    )

    messages = st.session_state.setdefault("messages", [])
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if show_sources and m["role"] == "assistant":
                render_sources(m.get("sources", []))

    if daily_used() >= s.max_daily_questions:
        st.info("The daily question limit for this demo has been reached. Please come back tomorrow.")
        st.chat_input("Daily limit reached", disabled=True)
        return
    left = questions_left(s)
    if left <= 0:
        st.info(f"You have used all {s.max_questions} questions. Thanks for trying it out!")
        st.chat_input("Question limit reached", disabled=True)
        return

    try:
        with st.spinner("Preparing the CV…"):
            index = get_index(s)
    except rag.CVError as exc:
        st.error(str(exc))
        return

    question = st.chat_input("Ask a question about the CV…")
    if not messages and not question:
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES):
            if col.button(ex, use_container_width=True):
                question = ex

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the CV and writing an answer…"):
                client = _client(s.gemini_api_key)
                hits = index.search(client, s.embedding_model, question, s.top_k, s.min_score)
                reply = rag.answer(client, s.chat_model, question, hits, history, s.fallback_model)
        except rag.CVError as exc:
            st.error(str(exc))
            return
        sources = [{"section": h.chunk.section, "score": h.score, "text": h.chunk.text} for h in hits]
        st.markdown(reply)
        if show_sources:
            render_sources(sources)
    _daily()["count"] += 1
    st.session_state["asked"] = st.session_state.get("asked", 0) + 1  # count only answered questions
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": reply, "sources": sources})
    st.rerun()  # redraw so the sidebar download button includes this turn


def main() -> None:
    s = load_settings()
    missing = s.missing()
    if missing:
        st.title("Chat with My CV")
        st.error(
            f"Missing configuration: {', '.join(missing)}. Copy `.env.example` to `.env` and fill it in "
            "(or set Streamlit secrets), then restart."
        )
        return
    if gate(s):
        chat(s)


main()
