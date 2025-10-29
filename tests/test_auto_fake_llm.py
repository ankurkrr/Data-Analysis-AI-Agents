import importlib
import os
import sys


def test_allow_auto_fake_fallback(monkeypatch, caplog):
    """Simulate OpenRouterLLM failing and ensure ResilientOpenRouterLLM falls back
    to fake LLM and annotates the JSON with __fake_output when ALLOW_AUTO_FAKE=1.
    """
    # Ensure environment toggle enabled
    monkeypatch.setenv("ALLOW_AUTO_FAKE", "1")
    monkeypatch.delenv("FORCE_FAKE_LLM", raising=False)

    # Reload module to pick up env var behavior
    if "app.llm.openrouter_llm" in sys.modules:
        del sys.modules["app.llm.openrouter_llm"]
    mod = importlib.import_module("app.llm.openrouter_llm")

    # Monkeypatch the real LLM to raise on call
    class DummyFailingReal:
        def _call(self, prompt, stop=None):
            raise RuntimeError("simulated LLM failure")

    # Replace OpenRouterLLM in the module with the dummy failing instance
    monkeypatch.setattr(mod, "OpenRouterLLM", lambda: DummyFailingReal())

    # Re-create the resilient wrapper
    rl = mod.ResilientOpenRouterLLM()

    # Capture logs
    caplog.clear()
    caplog.set_level("WARNING")

    out = rl._call("hello")

    # The output should be JSON with the __fake_output flag
    import json
    parsed = json.loads(out)
    assert parsed.get("__fake_output") is True
    # Ensure warning was logged
    assert any("ALLOW_AUTO_FAKE" in rec.message or "returning fake LLM response" in rec.message for rec in caplog.records)


def test_force_fake_llm_overrides_auto(monkeypatch):
    # If FORCE_FAKE_LLM is set, get_llm() should return the fake LLM directly
    monkeypatch.setenv("FORCE_FAKE_LLM", "1")
    monkeypatch.delenv("ALLOW_AUTO_FAKE", raising=False)
    if "app.llm.openrouter_llm" in sys.modules:
        del sys.modules["app.llm.openrouter_llm"]
    mod = importlib.import_module("app.llm.openrouter_llm")
    llm = mod.get_llm()
    # Fake LLM returns JSON with metadata key
    out = llm._call("test")
    import json
    parsed = json.loads(out)
    assert "metadata" in parsed
