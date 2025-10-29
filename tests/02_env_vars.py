# tests/02_env_vars.py
from dotenv import load_dotenv
import os
load_dotenv()
print("OPENROUTER_API_KEY:", bool(os.getenv("OPENROUTER_API_KEY")))
print("LLM_MODEL:", os.getenv("LLM_MODEL"))
print("MYSQL_HOST:", os.getenv("MYSQL_HOST"))
print("MYSQL_DB:", os.getenv("MYSQL_DB"))
