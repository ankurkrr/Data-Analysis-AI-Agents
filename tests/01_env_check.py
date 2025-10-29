# tests/01_env_check.py
import sys, platform, os
from importlib import util

print("Python:", sys.version)
print("Platform:", platform.platform())
print("CWD:", os.getcwd())

packages = ["langchain","requests","sentence_transformers","faiss","pdfplumber","camelot","pytesseract","pdf2image","mysql.connector","pydantic"]
for p in packages:
    found = util.find_spec(p) is not None
    print(f"{p}: {'OK' if found else 'MISSING'}")
