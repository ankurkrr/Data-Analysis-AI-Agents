# tests/06_fin_extractor_run.py
import tests.bootstrap_path
from app.tools.financial_extractor_tool import extract_financial_data
reports = [{"name":"SAMP","local_path":"tests/data/sample_report_q1.pdf"}]
res = extract_financial_data(reports)
import json
print(json.dumps(res, indent=2)[:2000])
