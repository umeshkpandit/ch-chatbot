"""Offline tests: no network or API key needed."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag  # noqa: E402

CV = Path(__file__).resolve().parent.parent / "cv.pdf"
SAMPLE = "Jane Doe\nEngineer\nSUMMARY\nBuilds things.\nEXPERIENCE\n" + "Did work at Acme. " * 80 + "\nSKILLS\nPython, SQL"


@pytest.mark.skipif(not CV.exists(), reason="cv.pdf not present")
def test_extract_real_cv():
    text = rag.extract_text(CV)
    assert len(text) > 200
    assert "EXPERIENCE" in text


def test_missing_pdf():
    with pytest.raises(rag.CVError, match="not found"):
        rag.extract_text(Path("nope.pdf"))


def test_corrupt_pdf(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(rag.CVError):
        rag.extract_text(bad)


def test_empty_pdf(tmp_path):
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(200, 200)
    p = tmp_path / "blank.pdf"
    with p.open("wb") as f:
        w.write(f)
    with pytest.raises(rag.CVError, match="scanned"):
        rag.extract_text(p)


def test_sections_and_chunks():
    names = [n for n, _ in rag.split_sections(SAMPLE)]
    assert names == ["HEADER", "SUMMARY", "EXPERIENCE", "SKILLS"]
    chunks = rag.chunk_text(SAMPLE, size=300, overlap=50)
    assert all(len(c.text) <= 300 + 20 for c in chunks)
    assert sum(c.section == "EXPERIENCE" for c in chunks) > 1
    assert all(c.text.startswith(f"[{c.section}]") for c in chunks)


def test_overlap_repeats_text():
    chunks = [c for c in rag.chunk_text(SAMPLE, size=300, overlap=100) if c.section == "EXPERIENCE"]
    a, b = chunks[0].text, chunks[1].text
    assert a[-30:].strip()[:15] in b


def test_chunk_validation():
    with pytest.raises(ValueError):
        rag.chunk_text("x", size=100, overlap=100)


def test_cosine_and_zero_vectors():
    m = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
    s = rag.cosine_similarity(np.array([1, 0]), m)
    assert s.tolist() == [1.0, 0.0, 0.0]
    assert rag.cosine_similarity(np.array([0, 0]), m).tolist() == [0.0, 0.0, 0.0]
    assert rag.cosine_similarity(np.array([1, 0]), np.zeros((0, 2))).size == 0


def test_rank_top_k_and_min_score():
    chunks = [rag.Chunk(f"c{i}", "S") for i in range(3)]
    idx = rag.CVIndex(chunks, np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32))
    hits = idx.rank(np.array([1, 0]), top_k=2)
    assert [h.chunk.text for h in hits] == ["c0", "c1"]
    assert [h.chunk.text for h in idx.rank(np.array([1, 0]), top_k=3, min_score=0.5)] == ["c0", "c1"]
    assert idx.rank(np.array([1, 0]), top_k=0) == []


class FakeModels:
    def embed_content(self, model, contents, config):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float("python" in c.lower()), 1.0]) for c in contents])

    def generate_content(self, model, contents, config):
        return SimpleNamespace(text=f"echo:{len(contents)}")


def test_search_and_answer_with_fake_client():
    client = SimpleNamespace(models=FakeModels())
    chunks = [rag.Chunk("[SKILLS]\nPython", "SKILLS"), rag.Chunk("[EDU]\nBSc", "EDU")]
    idx = rag.CVIndex(chunks, rag.embed_texts(client, "m", [c.text for c in chunks], "RETRIEVAL_DOCUMENT"))
    hits = idx.search(client, "m", "Does he know Python?", top_k=1)
    assert hits[0].chunk.section == "SKILLS"
    assert idx.search(client, "m", "   ") == []
    assert rag.answer(client, "m", "q", hits).startswith("echo:")


def test_prompt_handles_no_hits_and_history():
    p = rag.build_prompt("Where?", [], [{"role": "user", "content": "hi"}])
    assert "no relevant excerpts" in p and "user: hi" in p


def test_error_translation():
    from google.genai import errors

    e429 = errors.APIError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    assert "rate limit" in str(rag.friendly_api_error(e429)).lower()
    e404 = errors.APIError(404, {"error": {"message": "model x not found", "status": "NOT_FOUND"}})
    assert "model" in str(rag.friendly_api_error(e404)).lower()
    e400 = errors.APIError(400, {"error": {"message": "API key not valid", "status": "INVALID_ARGUMENT"}})
    assert "api key" in str(rag.friendly_api_error(e400)).lower()
    assert "network" in str(rag.friendly_api_error(ConnectionError("x"))).lower()


def test_missing_key():
    with pytest.raises(rag.CVError, match="GEMINI_API_KEY"):
        rag.make_client("")


def test_retry_on_transient_5xx(monkeypatch):
    from google.genai import errors

    monkeypatch.setattr(rag, "RETRY_DELAYS", (0, 0))
    calls = {"n": 0}

    class Flaky(FakeModels):
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            if calls["n"] < 3:
                raise errors.APIError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})
            return SimpleNamespace(text="ok")

    assert rag.answer(SimpleNamespace(models=Flaky()), "m", "q", []) == "ok"
    assert calls["n"] == 3


def test_no_retry_on_quota(monkeypatch):
    from google.genai import errors

    monkeypatch.setattr(rag, "RETRY_DELAYS", (0, 0))
    calls = {"n": 0}

    class Quota(FakeModels):
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            raise errors.APIError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    with pytest.raises(rag.CVError, match="rate limit"):
        rag.answer(SimpleNamespace(models=Quota()), "m", "q", [])
    assert calls["n"] == 1


def test_fallback_model_used_when_primary_overloaded(monkeypatch):
    from google.genai import errors

    monkeypatch.setattr(rag, "RETRY_DELAYS", (0, 0))
    used = []

    class Down(FakeModels):
        def generate_content(self, model, contents, config):
            used.append(model)
            if model == "primary":
                raise errors.APIError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})
            return SimpleNamespace(text="from-fallback")

    out = rag.answer(SimpleNamespace(models=Down()), "primary", "q", [], fallback_model="backup")
    assert out == "from-fallback"
    assert used == ["primary", "primary", "primary", "backup"]


def test_fallback_not_used_for_quota(monkeypatch):
    from google.genai import errors

    used = []

    class Quota(FakeModels):
        def generate_content(self, model, contents, config):
            used.append(model)
            raise errors.APIError(429, {"error": {"message": "q", "status": "RESOURCE_EXHAUSTED"}})

    with pytest.raises(rag.CVError, match="rate limit"):
        rag.answer(SimpleNamespace(models=Quota()), "primary", "q", [], fallback_model="backup")
    assert used == ["primary"]


def test_chat_to_text():
    msgs = [
        {"role": "user", "content": "Where does he work?"},
        {"role": "assistant", "content": "At Acme.", "sources": [{"section": "EXPERIENCE"}, {"section": "EXPERIENCE"}, {"section": "SUMMARY"}]},
    ]
    out = rag.chat_to_text(msgs)
    assert "You:\nWhere does he work?" in out
    assert "Assistant:\nAt Acme." in out
    assert "(CV sections used: Experience, Summary)" in out
    assert rag.chat_to_text([]).startswith("Chat with My CV")


def test_heading_glued_to_previous_line():
    text = "John Doe\nSenior Software Engineer SUMMARY\nBuilds things.\nEXPERIENCE\nWorked at Acme."
    sections = rag.split_sections(text)
    assert [n for n, _ in sections] == ["HEADER", "SUMMARY", "EXPERIENCE"]
    assert "Senior Software Engineer" in sections[0][1]
    assert "SUMMARY" not in sections[0][1]
    assert sections[1][1] == "Builds things."
