import os
import json
from fastapi.testclient import TestClient


class FakeDB:
    def __init__(self):
        self.requests = {}
        self.results = {}
        self.events = []

    def log_request(self, request_uuid, payload):
        self.requests[request_uuid] = payload

    def log_result(self, request_uuid, result, tools_raw=None):
        llm_mode = None
        llm_fake = False
        try:
            if isinstance(result, dict):
                meta = result.get("metadata", {})
                llm_mode = meta.get("llm_mode")
                llm_fake = bool(meta.get("llm_fake", False))
        except Exception:
            pass
        self.results[request_uuid] = {"result": result, "llm_mode": llm_mode, "llm_fake": llm_fake}

    def log_event(self, request_uuid, event_type, details=None):
        self.events.append({"request_uuid": request_uuid, "event_type": event_type, "details": details})

    def get_result(self, request_uuid):
        return self.results.get(request_uuid)


def test_forecast_endpoint_end_to_end(monkeypatch):
    # Force fake LLM and fake embedder to keep the test environment lightweight
    os.environ["FORCE_FAKE_LLM"] = "1"
    os.environ["FORCE_FAKE_EMBEDDER"] = "1"

    # Dummy fetcher returns a small transcript fixture included in tests/data
    class DummyFetcher:
        def fetch_quarterly_documents(self, ticker, quarters, sources):
            return {
                "reports": [],
                "transcripts": [{"name": "test_transcript_1", "local_path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_fixtures", "test_transcript_1.txt")}]
            }

    # Replace DocumentFetcher used by ForecastAgent
    monkeypatch.setattr("app.services.document_fetcher.DocumentFetcher", lambda: DummyFetcher())

    # Use FakeDB instead of MySQLClient in the API endpoints
    fake_db = FakeDB()
    monkeypatch.setattr("app.api.endpoints.MySQLClient", lambda: fake_db)

    # Create TestClient and POST to the endpoint
    from app.main import app

    client = TestClient(app)

    resp = client.post("/api/forecast/tcs", json={"quarters": 3, "sources": ["screener"], "include_market": False})

    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Check response structure
    assert "metadata" in data
    meta = data["metadata"]
    assert meta.get("llm_mode") == "fake"
    assert meta.get("llm_fake") is True

    request_id = meta.get("request_id")
    assert request_id is not None

    # Ensure DB fake recorded request and result
    assert request_id in fake_db.requests
    stored = fake_db.get_result(request_id)
    assert stored is not None
    assert stored["llm_mode"] == "fake"
    assert stored["llm_fake"] is True
