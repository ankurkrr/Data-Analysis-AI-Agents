from fastapi.testclient import TestClient
from app.main import app


def test_capabilities_endpoint():
    client = TestClient(app)
    resp = client.get("/api/health/capabilities")
    assert resp.status_code == 200
    data = resp.json()

    # Basic shape checks
    assert "llm" in data
    assert "embedder" in data
    assert "pdf_tools" in data
    assert "db" in data

    # LLM keys
    assert isinstance(data["llm"].get("forced_fake"), bool)
    assert isinstance(data["llm"].get("allow_auto_fake"), bool)
    assert isinstance(data["llm"].get("fake_output_possible"), bool)

    # Embedder keys
    assert isinstance(data["embedder"].get("forced_fake"), bool)
    assert isinstance(data["embedder"].get("sentence_transformers_available"), bool)
    assert isinstance(data["embedder"].get("faiss_available"), bool)

    # PDF tools keys
    assert isinstance(data["pdf_tools"].get("camelot_available"), bool)
    assert isinstance(data["pdf_tools"].get("pdfplumber_available"), bool)
    assert isinstance(data["pdf_tools"].get("pytesseract_available"), bool)

    # DB key
    assert isinstance(data["db"].get("mysql_connector_installed"), bool)
