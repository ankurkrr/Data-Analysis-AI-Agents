# tests/04_llm_wrapper.py
import tests.bootstrap_path
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app/llm/openrouter_llm.py')))
from dotenv import load_dotenv
load_dotenv()
from app.llm.openrouter_llm import OpenRouterLLM

llm = OpenRouterLLM()  # uses env
print("model:", llm.model)
out = llm._call("Return a short JSON: {\"ok\": true, \"msg\": \"hi\"}")
print("LLM response (first 200 chars):", out[:200])
