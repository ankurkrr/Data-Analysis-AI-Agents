import pytest
from app.llm.openrouter_llm import OpenRouterLLM

class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {"choices":[{"message":{"content":"Mocked response"}}]}
        self.text = "OK"

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")

def test_openrouter_llm_success(monkeypatch):
    # Mock requests.post used in OpenRouterLLM
    def fake_post(url, headers=None, json=None):
        return FakeResponse()

    monkeypatch.setattr('app.llm.openrouter_llm.requests.post', fake_post)

    llm = OpenRouterLLM(openrouter_api_key='test-key', model='test-model', max_retries=1)
    output = llm._call("Hello")
    assert isinstance(output, str)
    assert "Mocked response" in output

def test_openrouter_llm_rate_limit_then_success(monkeypatch):
    calls = {'n':0}
    def fake_post(url, headers=None, json=None):
        calls['n'] += 1
        if calls['n'] == 1:
            return FakeResponse(status_code=429, json_data=None)
        return FakeResponse()

    monkeypatch.setattr('app.llm.openrouter_llm.requests.post', fake_post)
    llm = OpenRouterLLM(openrouter_api_key='test-key', model='test-model', max_retries=2, retry_delay=0.1)
    output = llm._call("Hello")
    assert "Mocked response" in output
