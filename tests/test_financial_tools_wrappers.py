import json
from app.tools.financial_extractor_tool import extract_financial_metrics, validate_and_enrich_metrics_tool


def test_tool_wrappers_callables():
    text = "Total revenue was ₹1,000 crore. Net profit was 100 crore. Operating margin 10%."
    res = extract_financial_metrics(text)
    assert isinstance(res, dict)
    assert "metrics" in res
    assert res["count"] >= 1

    # Test validate_and_enrich tool by simulating a metrics JSON string
    metrics = {"total_revenue": {"value": 1000}, "operating_profit": {"value": 100}}
    # Call the class method directly to avoid LangChain callback handling in tests
    from app.tools.financial_extractor_tool import FinancialDataExtractorTool
    extractor = FinancialDataExtractorTool()
    v = extractor.validate_and_enrich_metrics(metrics, text)
    assert isinstance(v, dict)
    assert "metrics" in v
