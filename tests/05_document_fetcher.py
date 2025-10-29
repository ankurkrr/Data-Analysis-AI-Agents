# tests/05_document_fetcher.py
import tests.bootstrap_path
from app.services.document_fetcher import fetch_quarterly_documents
res = fetch_quarterly_documents("TCS", 2)
print("reports:", res.get("reports"))
print("transcripts:", res.get("transcripts"))
