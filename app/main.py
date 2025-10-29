from fastapi import FastAPI
from app.api.endpoints import router as api_router
from dotenv import load_dotenv
import os

load_dotenv()  # load .env file at runtime

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")

app = FastAPI(title="TCS Financial Forecasting Agent")

app.include_router(api_router, prefix="/api")

@app.get("/health")
async def health():
    return {"status": "ok"}
