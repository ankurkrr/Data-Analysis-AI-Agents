import os
import pytest
from app.services import document_fetcher


def test_fetch_tcs_ir_reports_mock(monkeypatch, tmp_path):
    """Mock TCS IR page and PDF responses to validate fetch_tcs_ir_reports behavior without network."""
    # Prepare fake URLs and HTML
    pdf_url = "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/q2fy26/sample.pdf"
    page_html = f'<html><body><a href="{pdf_url}">Financial Results</a></body></html>'

    class FakePageResp:
        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code
            self.headers = {"content-type": "text/html"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception("HTTP error")
        
        def iter_content(self, chunk_size=8192):
            # Provide an iter_content method so code that tries to stream will not fail
            yield self.text.encode("utf-8")

    class FakePDFResp:
        def __init__(self, content=None, status_code=200):
            # Ensure PDF content is large enough to pass size checks
            if content is None:
                content = b"%PDF-1.4\n%mockpdf\n" + (b"0" * 2000)
            self._content = content
            self.status_code = status_code
            self.headers = {"content-type": "application/pdf"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception("HTTP error")

        def iter_content(self, chunk_size=8192):
            # yield the content in one chunk
            yield self._content

    def fake_get(url, *args, **kwargs):
        # If stream=True, return PDF-like response for direct downloads
        if kwargs.get('stream'):
            # If requesting the direct PDF url, return PDF content
            if url == pdf_url:
                return FakePDFResp()
            # If requesting the main page as a stream, return PDF fallback content
            if url.startswith("https://www.tcs.com/investor-relations/financial-statements"):
                return FakePDFResp()
        # Return page HTML for the TCS base URL
        if url.startswith("https://www.tcs.com/investor-relations/financial-statements"):
            return FakePageResp(page_html)
        # Return PDF content for the expected PDF URL
        if url == pdf_url:
            return FakePDFResp()
        # Default fallback
        return FakePageResp("")

    # Monkeypatch requests.get used inside document_fetcher
    monkeypatch.setattr(document_fetcher, "requests", document_fetcher.requests)
    monkeypatch.setattr(document_fetcher.requests, "get", fake_get)

    # Use temporary download directory
    monkeypatch.setattr(document_fetcher, "DOWNLOAD_DIR", str(tmp_path))

    reports = document_fetcher.fetch_tcs_ir_reports(year="2025-26", quarters=["Q2"], max_reports=1)

    assert isinstance(reports, list)
    assert len(reports) == 1
    r = reports[0]
    assert r["source_url"].endswith("sample.pdf")
    assert os.path.exists(r["local_path"]) and os.path.getsize(r["local_path"]) > 10

