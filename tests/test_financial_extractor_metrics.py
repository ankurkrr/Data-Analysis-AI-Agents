import json
from app.tools.financial_extractor_tool import FinancialDataExtractorTool


def test_extract_metrics_from_text_basic():
    text = (
        "Total revenue from operations was ₹6,058.3 crore for the quarter. "
        "Net profit was 1,234 crore. Operating margin stood at 23.5%. EPS was 12.34. "
        "EBITDA stood at 7,000 crore. Return on equity was 18.2%. Free cash flow of 500 crore. "
        "Debt-to-equity ratio was 0.45."
    )

    extractor = FinancialDataExtractorTool()
    res = extractor.extract_metrics_from_text(text)
    metrics = res.get("metrics", {})

    assert "total_revenue" in metrics
    # allow small parsing rounding differences for large numbers
    assert abs(metrics["total_revenue"]["value"] - 6058.3) < 0.5
    assert "net_profit" in metrics
    assert abs(metrics["net_profit"]["value"] - 1234.0) < 0.5
    assert "operating_margin" in metrics
    assert abs(metrics["operating_margin"]["value"] - 23.5) < 1e-3
    assert "eps" in metrics
    assert abs(metrics["eps"]["value"] - 12.34) < 1e-6
    assert "ebitda" in metrics
    assert "roe" in metrics
    assert "free_cash_flow" in metrics
    assert "debt_to_equity" in metrics


def test_extract_with_million_unit_conversion():
    text = "Total revenue was 5000 million in the period."
    extractor = FinancialDataExtractorTool()
    res = extractor.extract_metrics_from_text(text)
    metrics = res.get("metrics", {})
    # 5000 million = 500 crore
    assert "total_revenue" in metrics
    assert abs(metrics["total_revenue"]["value"] - 500.0) < 1e-6


def test_validate_and_enrich_deterministic():
    # If LLM unavailable, validate_and_enrich should compute operating_margin from provided metrics
    extractor = FinancialDataExtractorTool()
    metrics_input = {
        "operating_profit": {"value": 200.0},
        "total_revenue": {"value": 2000.0}
    }
    res = extractor.validate_and_enrich_metrics(metrics_input, "Some report text")
    assert res.get("status") == "fallback"
    assert "operating_margin" in res.get("metrics", {})
    assert abs(res["metrics"]["operating_margin"]["value"] - 10.0) < 1e-6


def test_rate_of_change_computation():
    extractor = FinancialDataExtractorTool()
    metrics_input = {
        "total_revenue": {"value": 110.0},
        "total_revenue_prev": {"value": 100.0}
    }
    res = extractor.validate_and_enrich_metrics(metrics_input, "")
    assert res.get("status") in ("fallback", "ok")
    assert "total_revenue_qoq_pct" in res.get("metrics", {})
    pct = res["metrics"]["total_revenue_qoq_pct"]["value"]
    assert abs(pct - 10.0) < 1e-6
