import importlib
import sys
import os


def test_force_fake_embedder(monkeypatch):
    """Verify that setting FORCE_FAKE_EMBEDDER forces the module to use the
    lightweight fake embedder implementation.
    """
    # Ensure env var is set before importing the module
    monkeypatch.setenv("FORCE_FAKE_EMBEDDER", "1")

    # Remove module if already loaded so reload will pick up the env var behavior
    if "app.tools.qualitative_analysis_tool" in sys.modules:
        del sys.modules["app.tools.qualitative_analysis_tool"]

    mod = importlib.import_module("app.tools.qualitative_analysis_tool")

    tool = mod.QualitativeAnalysisTool()

    # The embedded instance should be the fake embedder
    assert hasattr(tool, "embedder"), "QualitativeAnalysisTool should have 'embedder' attribute"
    assert isinstance(tool.embedder, mod._FakeEmbedder), "embedder should be _FakeEmbedder when FORCE_FAKE_EMBEDDER=1"


def test_no_env_by_default(monkeypatch):
    """If the env var is not set, instantiating the tool should not forcibly
    create the fake embedder (it may still be used later as a fallback).
    This test simply ensures the toggle is optional and not enabled by default.
    """
    monkeypatch.delenv("FORCE_FAKE_EMBEDDER", raising=False)
    if "app.tools.qualitative_analysis_tool" in sys.modules:
        del sys.modules["app.tools.qualitative_analysis_tool"]

    mod = importlib.import_module("app.tools.qualitative_analysis_tool")
    tool = mod.QualitativeAnalysisTool()

    # When not forced, embedder may be None (lazy load) or some other instance; ensure it's not forcibly the fake one
    assert not isinstance(tool.embedder, mod._FakeEmbedder), "embedder should not be forced to _FakeEmbedder when env var not set"
