# app/main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.api.endpoints import router as api_router
import logging
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TCS Financial Forecasting Agent",
    description="AI-powered financial forecasting for TCS",
    version="1.0.0"
)

# Add startup event to log initialization
@app.on_event("startup")
async def startup_event():
    logger.info("="*50)
    logger.info("Starting TCS Forecast Agent")
    logger.info("="*50)
    try:
        # Test imports
        from app.agents.forecast_agent import ForecastAgent
        from app.db.mysql_client import MySQLClient
        logger.info("✓ All imports successful")
        
        # Test DB connection
        db = MySQLClient()
        logger.info("✓ Database connection successful")
        
    except Exception as e:
        logger.error(f"✗ Startup failed: {str(e)}", exc_info=True)
        raise

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "path": str(request.url)
        }
    )

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "TCS Financial Forecasting Agent",
        "endpoints": {
            "health": "/health",
            "forecast": "/api/forecast/tcs",
            "capabilities": "/api/health/capabilities"
        }
    }

@app.get("/health")
async def health():
    logger.info("Health check requested")
    try:
        from app.db.mysql_client import MySQLClient
        db = MySQLClient()
        return {
            "status": "ok",
            "database": "connected",
            "timestamp": __import__("datetime").datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )