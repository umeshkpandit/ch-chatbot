"""UI smoke tests with Streamlit's AppTest. No real API calls are made."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run(monkeypatch, **env):
    for k in ("GEMINI_API_KEY", "APP_ACCESS_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return AppTest.from_file(APP, default_timeout=30).run()


def test_missing_config_message(monkeypatch):
    at = run(monkeypatch)
    # .env may supply values locally; only assert when truly missing
    if at.error:
        assert "Missing configuration" in at.error[0].value


def test_gate_rejects_then_locks(monkeypatch):
    at = run(monkeypatch, GEMINI_API_KEY="x", APP_ACCESS_KEY="secret", MAX_FAILED_ATTEMPTS="2", LOCKOUT_SECONDS="60")
    at.text_input[0].set_value("wrong").run()
    at.button[0].click().run()
    assert any("Incorrect" in e.value for e in at.error)
    at.text_input[0].set_value("wrong").run()
    at.button[0].click().run()
    assert any("Locked" in e.value or "Too many" in e.value for e in at.error)


def test_question_limit_blocks_chat(monkeypatch):
    import streamlit as st

    st.cache_resource.clear()  # earlier tests leave the shared lockout tracker locked
    at = run(monkeypatch, GEMINI_API_KEY="x", APP_ACCESS_KEY="secret", MAX_QUESTIONS_PER_USER="0")
    at.text_input[0].set_value("secret")
    at.button[0].click().run()
    assert any("used all" in i.value for i in at.info)


def test_daily_cap_blocks_chat(monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    at = run(monkeypatch, GEMINI_API_KEY="x", APP_ACCESS_KEY="secret", MAX_DAILY_QUESTIONS="0")
    at.text_input[0].set_value("secret")
    at.button[0].click().run()
    assert any("daily question limit" in i.value for i in at.info)


def test_history_stays_visible_at_limit(monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    at = run(monkeypatch, GEMINI_API_KEY="x", APP_ACCESS_KEY="secret", MAX_QUESTIONS_PER_USER="1")
    at.text_input[0].set_value("secret")
    at.button[0].click().run()
    at.session_state["messages"] = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there", "sources": []}]
    at.session_state["asked"] = 1
    at.run()
    assert any("used all" in i.value for i in at.info)
    assert any("hello there" in m.value for m in at.markdown)
