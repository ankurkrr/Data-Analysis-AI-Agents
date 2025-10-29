from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Optional
import logging

from app.agents.forecast_agent import ForecastAgent
from app.db.mysql_client import MySQLClient

router = APIRouter()
# Lazy-initialized services to avoid import-time errors (e.g. DB down)
agent: Optional[ForecastAgent] = None
db: Optional[MySQLClient] = None


def ensure_services():
    """Initialize agent and DB lazily. Raises HTTPException on failure."""
    global agent, db
    if agent is None:
        try:
            agent = ForecastAgent()
        except Exception as e:
            logging.exception("Failed to initialize ForecastAgent")
            raise HTTPException(status_code=500, detail=f"ForecastAgent initialization failed: {e}")

    if db is None:
        try:
            db = MySQLClient()
        except Exception as e:
            logging.exception("Failed to initialize MySQLClient")
            raise HTTPException(status_code=500, detail=f"MySQLClient initialization failed: {e}")

class ForecastRequest(BaseModel):
    quarters: int = 3
    sources: list = ["screener", "company-ir"]
    include_market: bool = False

@router.post("/forecast/tcs")
async def forecast_tcs(req: ForecastRequest):
    # Ensure services are available (lazy init). If initialization fails, ensure_services raises HTTPException.
    ensure_services()

    request_id = str(uuid4())
    payload = req.dict()
    try:
        db.log_request(request_id, payload)
    except Exception:
        # If logging the request fails, continue but record warning
        logging.exception("Failed to log request to DB; continuing with agent run")

    try:
        result = agent.run(ticker="TCS", request_id=request_id, quarters=req.quarters, sources=req.sources, include_market=req.include_market)
        try:
            db.log_result(request_id, result)
        except Exception:
            logging.exception("Failed to log result to DB")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{request_id}")
async def status(request_id: str):
    ensure_services()
    try:
        return db.get_result(request_id)
    except Exception:
        logging.exception("Failed to fetch result from DB")
        raise HTTPException(status_code=500, detail="Failed to fetch result")
