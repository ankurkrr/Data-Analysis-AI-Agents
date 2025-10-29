from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from uuid import uuid4
from typing import Optional
import logging
import json
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-initialized services
agent: Optional[object] = None
db: Optional[object] = None

def ensure_services():
    """Initialize agent and DB lazily with detailed error logging"""
    global agent, db
    
    if agent is None:
        try:
            logger.info("Initializing ForecastAgent...")
            from app.agents.forecast_agent import ForecastAgent
            agent = ForecastAgent()
            logger.info("✓ ForecastAgent initialized")
        except Exception as e:
            logger.error(f"✗ ForecastAgent init failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"ForecastAgent initialization failed: {str(e)}"
            )

    if db is None:
        try:
            logger.info("Initializing MySQLClient...")
            from app.db.mysql_client import MySQLClient
            db = MySQLClient()
            logger.info("✓ MySQLClient initialized")
        except Exception as e:
            logger.error(f"✗ MySQLClient init failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"MySQLClient initialization failed: {str(e)}"
            )

class ForecastRequest(BaseModel):
    quarters: int = 3
    sources: list = ["screener", "company-ir"]
    include_market: bool = False

@router.post("/forecast/tcs")
async def forecast_tcs(req: ForecastRequest):
    """Generate TCS forecast with timeout protection"""
    request_id = str(uuid4())
    start_time = datetime.utcnow()
    
    logger.info(f"="*60)
    logger.info(f"NEW REQUEST: {request_id}")
    logger.info(f"Params: quarters={req.quarters}, sources={req.sources}")
    logger.info(f"="*60)
    
    # Ensure services available
    try:
        ensure_services()
    except HTTPException as e:
        logger.error(f"Service initialization failed: {e.detail}")
        raise
    
    payload = req.dict()
    
    # Log request
    try:
        db.log_request(request_id, payload)
        logger.info(f"✓ Request logged to DB")
    except Exception as e:
        logger.warning(f"Failed to log request: {str(e)}")
    
    # Run agent with timeout
    try:
        logger.info("Starting agent execution...")
        
        # Run with asyncio timeout (60 seconds for Postman compatibility)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    agent.run,
                    ticker="TCS",
                    request_id=request_id,
                    quarters=req.quarters,
                    sources=req.sources,
                    include_market=req.include_market
                ),
                timeout=120.0  # 2 minutes max
            )
        except asyncio.TimeoutError:
            logger.error(f"Agent execution timed out after 120s")
            raise HTTPException(
                status_code=504,
                detail="Request timed out. Try reducing quarters or check document sources."
            )
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"✓ Agent completed in {duration:.1f}s")
        
        # Log result
        try:
            db.log_result(request_id, result)
            logger.info(f"✓ Result logged to DB")
        except Exception as e:
            logger.warning(f"Failed to log result: {str(e)}")
        
        # Check for errors in result
        if isinstance(result, dict) and "error" in result:
            logger.warning(f"Agent returned error: {result.get('error')}")
            return JSONResponse(
                status_code=500,
                content=result
            )
        
        logger.info(f"="*60)
        logger.info(f"REQUEST COMPLETED: {request_id}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"="*60)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
        error_result = {
            "error": "agent_execution_failed",
            "message": str(e),
            "request_id": request_id,
            "ticker": "TCS"
        }
        try:
            db.log_result(request_id, error_result)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{request_id}")
async def status(request_id: str):
    """Get status of a previous request"""
    logger.info(f"Status check for: {request_id}")
    ensure_services()
    
    try:
        result = db.get_result(request_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No result found for request_id: {request_id}"
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/capabilities")
async def capabilities():
    """Return runtime capabilities"""
    import os
    import importlib.util

    def _check_import(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return False

    caps = {
        "llm": {
            "forced_fake": bool(os.getenv("FORCE_FAKE_LLM")),
            "allow_auto_fake": bool(os.getenv("ALLOW_AUTO_FAKE")),
            "api_key_set": bool(os.getenv("OPENROUTER_API_KEY"))
        },
        "embedder": {
            "forced_fake": bool(os.getenv("FORCE_FAKE_EMBEDDER")),
            "sentence_transformers_available": _check_import("sentence_transformers"),
            "faiss_available": _check_import("faiss")
        },
        "pdf_tools": {
            "camelot_available": _check_import("camelot"),
            "pdfplumber_available": _check_import("pdfplumber"),
            "pytesseract_available": _check_import("pytesseract")
        },
        "db": {
            "mysql_connector_installed": _check_import("mysql.connector"),
            "connection_configured": all([
                os.getenv("MYSQL_HOST"),
                os.getenv("MYSQL_USER"),
                os.getenv("MYSQL_DB")
            ])
        }
    }

    return caps

from fastapi.responses import JSONResponse