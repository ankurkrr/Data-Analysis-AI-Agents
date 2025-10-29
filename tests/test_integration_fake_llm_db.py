import os
import json
from uuid import uuid4

from app.agents.forecast_agent import ForecastAgent


class FakeDB:
    def __init__(self):
        self.requests = {}
        self.results = {}
        self.events = []

    def log_request(self, request_uuid, payload):
        self.requests[request_uuid] = payload

    def log_result(self, request_uuid, result, tools_raw=None):
        # mimic MySQLClient.log_result extraction
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

    def get_result(self, request_uuid):
        return self.results.get(request_uuid)

    def log_event(self, request_uuid, event_type, details=None):
        self.events.append({"request_uuid": request_uuid, "event_type": event_type, "details": details})


def test_forced_fake_llm_persists_llm_mode(monkeypatch):
    # Force the fake LLM mode
    os.environ["FORCE_FAKE_LLM"] = "1"

    # Replace DocumentFetcher to avoid external network
    class DummyFetcher:
        def fetch_quarterly_documents(self, ticker, quarters, sources):
            return {"reports": [{"name": "dummy", "local_path": "./tests/data/sample_report_q3.pdf"}], "transcripts": []}

    monkeypatch.setattr("app.services.document_fetcher.DocumentFetcher", lambda: DummyFetcher())

    # Replace agent executor creation so we don't invoke LangChain
    def fake_executor_factory(self, ticker, request_id, quarters, documents):
        class Exec:
            def invoke(self, *_args, **_kwargs):
                # Return a fake LLM output that contains the marker
                fake_output = json.dumps({"__fake_output": True, "forecast": {"note": "fake"}})
                return {"output": fake_output, "intermediate_steps": []}
        return Exec()

    monkeypatch.setattr(ForecastAgent, "_create_agent_executor", fake_executor_factory)

    # Use fake DB to capture writes
    fake_db = FakeDB()

    agent = ForecastAgent()
    # Run agent
    request_id = str(uuid4())
    result = agent.run(ticker="TCS", request_id=request_id, quarters=1, sources=["screener"], include_market=False)

    # Simulate endpoint behavior that logs the result to DB
    fake_db.log_result(request_id, result)
    # Simulate event logging
    meta = result.get("metadata", {}) if isinstance(result, dict) else {}
    if meta.get("llm_mode") and meta.get("llm_mode") != "real":
        fake_db.log_event(request_id, "llm_fallback", {"mode": meta.get("llm_mode"), "llm_fake": bool(meta.get("llm_fake", False))})

    stored = fake_db.get_result(request_id)
    assert stored is not None
    assert stored["llm_mode"] == "fake"
    assert stored["llm_fake"] is True
