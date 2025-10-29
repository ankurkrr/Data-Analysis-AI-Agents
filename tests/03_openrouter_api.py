# tests/03_openrouter_api.py
from dotenv import load_dotenv
import os, requests, json
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit("OPENROUTER_API_KEY not set")

url = "https://openrouter.ai/api/v1"
payload = {
    "model": os.getenv("LLM_MODEL", "openai/gpt-oss-20b:free"),
    "messages": [{"role":"user", "content":"Say hello in one short sentence."}],
    "temperature": 0.2,
    "max_tokens": 50
}
headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
r = requests.post(url, headers=headers, json=payload, timeout=30)
print("HTTP", r.status_code)
try:
    data = r.json()
    print("Choices present:", "choices" in data)
    print(json.dumps(data.get("choices", data), indent=2)[:1000])
except Exception as e:
    print("JSON parse error:", e, r.text[:1000])
