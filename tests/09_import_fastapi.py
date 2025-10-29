# tests/09_import_fastapi.py
import importlib, traceback
try:
    m = importlib.import_module("app.main")
    print("Imported app.main OK")
except Exception:
    traceback.print_exc()
