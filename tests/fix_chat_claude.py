
Assignment:
Task: Financial Forecasting Agent for TCS
Your task is to build a FastAPI application that acts as an AI agent capable of generating a business outlook forecast for Tata Consultancy Services (TCS).
The agent's primary function is to move beyond simple Q&A. It must automatically find and analyze financial documents from past 1-2 quarters to generate a reasoned, qualitative forecast for the future.
Source: You are expected to be resourceful. Find and download the necessary documents (e.g., quarterly financial reports, earnings call transcripts) for the last 1-2 quarters from a source like https://www.screener.in/company/TCS/consolidated/#documents
Usage of AI
At Elevation AI, we embrace AI-first solutions. For this assignment, if you have used AI, we’re keen to understand how. Please document:
Your AI stack used and reasoning approach
The specific tools/models employed (e.g., OCR, RAG stack, embeddings, vector DB, LLM provider, function-calling).
What the AI actually achieved end-to-end (data sources retrieved, metrics extracted, synthesis quality).
Guardrails and evaluation (prompting strategy, retries, grounding checks).
Limits and tradeoffs you encountered—and how you mitigated them.
Core Requirements
You will build an agent with access to at least two specialized, purpose-built tools:
FinancialDataExtractorTool: A robust tool designed to understand quarterly financial reports and extract key financial metrics (e.g., Total Revenue, Net Profit, Operating Margin).


QualitativeAnalysisTool: A RAG-based tool that performs semantic search and analysis across 2-3 past earnings call transcripts to identify recurring themes, management sentiment, and forward-looking statements.
Deliverables
Generate a Forecast: The primary endpoint of your API must be able to handle a complex analytical task.
Example Task: "Analyze the financial reports and transcripts for the last three quarters and provide a qualitative forecast for the upcoming quarter. Your forecast must identify key financial trends (e.g., revenue growth, margin pressure), summarize management's stated outlook, and highlight any significant risks or opportunities mentioned."


Provide Structured Output: The agent's final output must be a structured JSON object. This demonstrates your ability to control the LLM and deliver predictable, machine-readable results.


Log the Results: The agent must be served via a FastAPI endpoint, and all incoming requests and the final JSON output must be logged to a MySQL database.
Optional, not Necessary
MarketDataTool: As an optional bonus, you can implement a third tool that fetches live market data (e.g., current stock price) and incorporates it as another point of context in the analysis.
Technical Stack & Expectations
Programming Language: Python 3.10+
Backend Framework: FastAPI
LLM Framework: LangChain
AI Provider: Any
Database: MySQL 8.0
What to Submit & The Importance of the README
Your submission will be evaluated not just on the code, but on how easy it is for us to understand and run. Another engineer must be able to clone your repository, follow your instructions, and have the service running locally without any guesswork.
Please provide a link to a Git repository containing:
Source Code: All your Python scripts.
requirements.txt: A file listing all necessary libraries.
README.md: This must include:
Project Overview: Your architectural approach, design choices, and how your agent chains thoughts and tools to create a forecast.
Agent & Tool Design: A detailed explanation of each tool and the master prompt you used to guide your agent's reasoning.
Setup Instructions: Clear, step-by-step instructions on setting up the environment, installing dependencies, and configuring all credentials (LLMs and MySQL). This must be unambiguous.
How to Run: The exact commands to start the FastAPI service.
How We Will Evaluate Your Submission
Reasoning: Does the agent successfully perform a multi-step analysis? Can it synthesize data from multiple documents and tools into a coherent forecast?
Engineering & Architecture: How well-designed are your tools and agentic chain? Is the logic for extracting financial data robust?
Code Quality & Readability: Is your code clean, modular, and easy to maintain? Does it follow best practices for a production-ready service?
Clarity and Reproducibility of Documentation: Can we run your project just by reading your README? How clear are your explanations of your design?


Architecture:
tcs-forecast-agent/
├─ app/
│  ├─ main.py
│  ├─ api/
│  │  ├─ endpoints.py
│  ├─ agents/
│  │  ├─ forecast_agent.py
│  ├─ data/
│  │  ├─ sample_pdf
│  ├─ services/
│  │  ├─ document_fetcher.py
│  ├─ db/
│  │  ├─ mysql_client.py
│  ├─ utils/
│  │  └─ number_parsing.py
│  ├─ llm/
│  │  ├─ openrouter_llm.py
|  |- tools/
│  ├─ financial_extractor_tool.py
│  ├─ qualitative_analysis_tool.py
│  └─ data/               # sample pdfs + transcripts for tests
│  ├─ utils/
│  │  └─ debug_server.py. 
│  │  └─ check_response.py.
│  │  └─ test_agent_flow.py.
│  │  └─ test_api_manual.py.
│  │  └─ verify_agent_code.py.
├─ requirements.txt
├─ README.md
└─ .env 

Codes:
# app/agents/forecast_agent.py
"""
app/agents/forecast_agent.py - Properly integrated LangChain agent
"""
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from app.llm.openrouter_llm import OpenRouterLLM
from app.tools.financial_extractor_tool import FinancialDataExtractorTool
from app.tools.qualitative_analysis_tool import QualitativeAnalysisTool
import json
import datetime
from typing import Dict, Any, List
from jsonschema import validate, ValidationError

# JSON Schema for final forecast
FORECAST_SCHEMA = {
    "type": "object",
    "required": ["metadata", "numeric_trends", "qualitative_summary", "forecast", "risks_and_opportunities", "sources"],
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "request_id": {"type": "string"},
                "analysis_date": {"type": "string"},
                "quarters_analyzed": {"type": "array"}
            },
            "required": ["ticker", "request_id", "analysis_date"]
        },
        "numeric_trends": {"type": "object"},
        "qualitative_summary": {"type": "object"},
        "forecast": {"type": "object"},
        "risks_and_opportunities": {"type": "object"},
        "sources": {"type": "array"}
    }
}

# ReAct Agent Prompt Template
REACT_PROMPT = """You are a senior financial analyst AI agent with access to specialized tools for analyzing TCS's financial performance.

Your goal: Generate a comprehensive business outlook forecast for TCS by analyzing financial reports and earnings transcripts.

TOOLS:
You have access to the following tools:

{tools}

TOOL USAGE FORMAT:
To use a tool, use the following format:

Thought: [Your reasoning about what to do next]
Action: [The tool name from {tool_names}]
Action Input: [The input to the tool as valid JSON]
Observation: [The tool's response will appear here]

After receiving observations, you can either:
1. Continue with another Thought/Action/Action Input/Observation cycle
2. Provide the final answer

When you have gathered sufficient information, respond with:

Thought: I now have all the information needed to create the forecast
Final Answer: [Your complete JSON forecast following the schema]

IMPORTANT RULES:
- Always think step-by-step before taking an action
- Use FinancialDataExtractorTool FIRST to get numeric metrics
- Then use QualitativeAnalysisTool to get management insights
- You can call tools multiple times if needed
- Base all claims on actual tool outputs - never fabricate data
- The Final Answer MUST be valid JSON matching the forecast schema

FORECAST JSON SCHEMA:
{{
    "metadata": {{"ticker": "TCS", "request_id": "<id>", "analysis_date": "<iso-date>", "quarters_analyzed": ["Q1", "Q2", "Q3"]}},
    "numeric_trends": {{
        "total_revenue": {{"values": [{{"period": "Q1-2024", "value": 12345, "unit": "INR_Cr"}}], "trend": "increasing", "qoq_change_pct": 3.5}},
        "net_profit": {{"values": [...], "trend": "...", "qoq_change_pct": ...}},
        "operating_margin": {{"values": [...], "trend": "..."}}
    }},
    "qualitative_summary": {{
        "themes": ["digital transformation demand", "attrition stabilizing"],
        "management_sentiment": {{"score": 0.6, "summary": "Cautiously optimistic"}},
        "forward_guidance": ["Expecting 5-7% growth", "Focus on AI/ML investments"]
    }},
    "forecast": {{
        "outlook_text": "Based on analysis...",
        "numeric_projection": {{"metric": "revenue", "low": 13000, "high": 13500, "unit": "INR_Cr"}},
        "confidence": 0.75
    }},
    "risks_and_opportunities": {{
        "risks": ["Attrition pressure", "Macro headwinds"],
        "opportunities": ["AI/GenAI adoption", "Deal pipeline strength"]
    }},
    "sources": [
        {{"tool": "FinancialDataExtractorTool", "document": "Q3-FY24-Results.pdf", "location": "page 2"}},
        {{"tool": "QualitativeAnalysisTool", "document": "Q3-Earnings-Transcript.txt", "location": "chunk_3"}}
    ]
}}

REQUEST CONTEXT:
Ticker: {ticker}
Request ID: {request_id}
Quarters to analyze: {quarters}
Documents available: {documents_summary}

Begin your analysis!

{agent_scratchpad}"""


class ForecastAgent:
    def __init__(self):
        self.llm = OpenRouterLLM()
        self.financial_tool_instance = FinancialDataExtractorTool()
        self.qualitative_tool_instance = QualitativeAnalysisTool()
        self.agent_executor = None
        self._tool_call_history = []
        
    def _create_tools(self, documents: Dict[str, List[Dict]]) -> List[Tool]:
        """Create LangChain Tool wrappers for our custom tools"""
        
        def financial_extractor_wrapper(input_str: str) -> str:
            """Extract financial metrics from quarterly reports"""
            try:
                # Parse input (should be JSON with report indices or "all")
                input_data = json.loads(input_str) if input_str.strip() else {"action": "extract_all"}
                
                reports = documents.get("reports", [])
                if not reports:
                    return json.dumps({"error": "No reports available", "reports_count": 0})
                
                # Call the actual tool
                result = self.financial_tool_instance.extract(reports)
                self._tool_call_history.append({
                    "tool": "FinancialDataExtractorTool",
                    "input": input_data,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                })
                
                # Return structured summary that agent can reason about
                summary = {
                    "status": "success",
                    "reports_analyzed": len(result.get("results", [])),
                    "metrics_found": self._summarize_metrics(result),
                    "raw_output": result  # Full output for synthesis later
                }
                return json.dumps(summary, indent=2)
                
            except Exception as e:
                return json.dumps({"error": str(e), "status": "failed"})
        
        def qualitative_analyzer_wrapper(input_str: str) -> str:
            """Analyze earnings call transcripts for themes and sentiment"""
            try:
                input_data = json.loads(input_str) if input_str.strip() else {"action": "analyze_all"}
                
                transcripts = documents.get("transcripts", [])
                if not transcripts:
                    return json.dumps({"error": "No transcripts available", "transcripts_count": 0})
                
                # Call the actual tool
                result = self.qualitative_tool_instance.analyze(transcripts)
                self._tool_call_history.append({
                    "tool": "QualitativeAnalysisTool",
                    "input": input_data,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                })
                
                # Return structured summary
                summary = {
                    "status": "success",
                    "transcripts_analyzed": len(transcripts),
                    "themes_identified": len(result.get("themes", [])),
                    "theme_names": [t["theme"] for t in result.get("themes", [])],
                    "sentiment": result.get("management_sentiment", {}),
                    "forward_guidance_count": len(result.get("forward_guidance", [])),
                    "risks_identified": len(result.get("risks", [])),
                    "raw_output": result
                }
                return json.dumps(summary, indent=2)
                
            except Exception as e:
                return json.dumps({"error": str(e), "status": "failed"})
        
        # Create LangChain Tool objects
        tools = [
            Tool(
                name="FinancialDataExtractorTool",
                func=financial_extractor_wrapper,
                description="""Use this tool to extract financial metrics from TCS quarterly reports.
                Input: JSON with optional parameters like {"action": "extract_all"} or {"report_indices": [0, 1]}
                Output: JSON containing extracted metrics like revenue, profit, margins with their values and confidence scores.
                Call this tool FIRST to get numeric data."""
            ),
            Tool(
                name="QualitativeAnalysisTool",
                func=qualitative_analyzer_wrapper,
                description="""Use this tool to analyze earnings call transcripts for qualitative insights.
                Input: JSON with optional parameters like {"action": "analyze_all"} or {"focus": "guidance"}
                Output: JSON containing themes, sentiment analysis, forward guidance, and identified risks.
                Call this tool AFTER financial extraction to get management's perspective."""
            )
        ]
        
        return tools
    
    def _summarize_metrics(self, financial_result: Dict) -> Dict:
        """Helper to summarize extracted metrics for agent reasoning"""
        summary = {}
        for result in financial_result.get("results", []):
            metrics = result.get("metrics", {})
            for key, data in metrics.items():
                if key not in summary:
                    summary[key] = []
                summary[key].append({
                    "value": data.get("value"),
                    "unit": data.get("unit"),
                    "confidence": data.get("confidence"),
                    "source_doc": result.get("doc_meta", {}).get("name")
                })
        return summary
    
    def _create_agent_executor(self, ticker: str, request_id: str, quarters: int, documents: Dict) -> AgentExecutor:
        """Create the LangChain agent executor"""
        
        tools = self._create_tools(documents)
        
        # Create the prompt
        docs_summary = {
            "reports": [{"name": r.get("name"), "source": r.get("source_url", "local")} for r in documents.get("reports", [])],
            "transcripts": [{"name": t.get("name"), "source": t.get("source_url", "local")} for t in documents.get("transcripts", [])]
        }
        
        prompt = PromptTemplate(
            template=REACT_PROMPT,
            input_variables=["ticker", "request_id", "quarters", "documents_summary", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                "tool_names": ", ".join([tool.name for tool in tools])
            }
        )
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            max_execution_time=300,  # 5 minutes timeout
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    def _validate_and_repair_forecast(self, forecast_json: str, context: Dict, max_attempts: int = 3) -> Dict:
        """Validate forecast against schema and attempt repairs if needed"""
        
        attempts = []
        
        for attempt in range(1, max_attempts + 1):
            attempts.append({"attempt": attempt, "raw": forecast_json[:500]})
            
            # Try to parse JSON
            try:
                parsed = json.loads(forecast_json)
            except json.JSONDecodeError as e:
                if attempt == max_attempts:
                    return {
                        "error": "json_parse_failed",
                        "details": str(e),
                        "attempts": attempts
                    }
                
                # Ask LLM to fix JSON
                repair_prompt = f"""The following text should be valid JSON but has parsing errors:

{forecast_json}

Error: {str(e)}

Please return ONLY the corrected JSON with no additional text or explanation."""
                
                try:
                    forecast_json = self.llm._call(repair_prompt)
                except Exception:
                    continue
                continue
            
            # Validate against schema
            try:
                validate(instance=parsed, schema=FORECAST_SCHEMA)
                return {"status": "success", "forecast": parsed, "attempts": attempts}
            except ValidationError as ve:
                if attempt == max_attempts:
                    return {
                        "error": "schema_validation_failed",
                        "details": str(ve),
                        "parsed_json": parsed,
                        "attempts": attempts
                    }
                
                # Ask LLM to fix schema issues
                repair_prompt = f"""The following JSON doesn't match the required schema:

{json.dumps(parsed, indent=2)}

Schema validation error: {str(ve)}

Required schema:
{json.dumps(FORECAST_SCHEMA, indent=2)}

Please return a corrected JSON that strictly follows the schema. Use the context below for data:
{json.dumps(context, indent=2)[:2000]}"""
                
                try:
                    forecast_json = self.llm._call(repair_prompt)
                except Exception:
                    continue
        
        return {
            "error": "max_attempts_exceeded",
            "attempts": attempts
        }
    
    def run(self, ticker: str, request_id: str, quarters: int = 3, sources: List[str] = None, include_market: bool = False) -> Dict[str, Any]:
        """
        Main entry point: Run the LangChain agent to generate forecast
        """
        from app.services.document_fetcher import DocumentFetcher
        
        # Step 1: Fetch documents
        fetcher = DocumentFetcher()
        documents = fetcher.fetch_quarterly_documents(ticker, quarters, sources)
        
        if not documents.get("reports") and not documents.get("transcripts"):
            return {
                "error": "no_documents_found",
                "message": "Could not fetch any financial documents or transcripts",
                "ticker": ticker,
                "request_id": request_id
            }
        
        # Step 2: Create and run agent
        try:
            agent_executor = self._create_agent_executor(ticker, request_id, quarters, documents)
            
            # Run the agent
            result = agent_executor.invoke({
                "ticker": ticker,
                "request_id": request_id,
                "quarters": quarters,
                "documents_summary": json.dumps({
                    "reports_count": len(documents.get("reports", [])),
                    "transcripts_count": len(documents.get("transcripts", []))
                })
            })
            
            # Extract the agent's output
            agent_output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Step 3: Validate and repair the forecast
            validation_result = self._validate_and_repair_forecast(
                agent_output,
                context={"documents": documents, "tool_history": self._tool_call_history}
            )
            
            # Step 4: Build final response
            final_response = {
                "metadata": {
                    "ticker": ticker,
                    "request_id": request_id,
                    "analysis_date": datetime.datetime.utcnow().isoformat(),
                    "quarters_analyzed": [r.get("name") for r in documents.get("reports", [])]
                },
                "agent_execution": {
                    "intermediate_steps_count": len(intermediate_steps),
                    "tool_calls": self._tool_call_history,
                    "iterations": len([step for step in intermediate_steps if isinstance(step[0], AgentAction)])
                },
                "forecast": validation_result.get("forecast") if validation_result.get("status") == "success" else None,
                "validation": validation_result,
                "raw_agent_output": agent_output[:1000],  # Truncate for logging
                "documents_processed": {
                    "reports": [{"name": r.get("name"), "path": r.get("local_path")} for r in documents.get("reports", [])],
                    "transcripts": [{"name": t.get("name"), "path": t.get("local_path")} for t in documents.get("transcripts", [])]
                }
            }
            
            return final_response
            
        except Exception as e:
            return {
                "error": "agent_execution_failed",
                "message": str(e),
                "ticker": ticker,
                "request_id": request_id,
                "tool_calls": self._tool_call_history
            }

app/api/endpoints.py:
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from app.agents.forecast_agent import ForecastAgent
from app.db.mysql_client import MySQLClient

router = APIRouter()
agent = ForecastAgent()
db = MySQLClient()

class ForecastRequest(BaseModel):
    quarters: int = 3
    sources: list = ["screener", "company-ir"]
    include_market: bool = False

@router.post("/forecast/tcs")
async def forecast_tcs(req: ForecastRequest):
    request_id = str(uuid4())
    payload = req.dict()
    db.log_request(request_id, payload)
    try:
        result = agent.run(ticker="TCS", request_id=request_id, quarters=req.quarters, sources=req.sources, include_market=req.include_market)
        db.log_result(request_id, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{request_id}")
async def status(request_id: str):
    return db.get_result(request_id)

db/mysql_client.py:
import mysql.connector
import os
import json
from dotenv import load_dotenv

load_dotenv()

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "tcs_forecast"),
}

class MySQLClient:
    def __init__(self):
        db_name = MYSQL_CONFIG["database"]

        # Step 1: Connect to MySQL *without* database to ensure it exists
        temp_conn = mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"]
        )
        temp_cursor = temp_conn.cursor()
        temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        temp_conn.commit()
        temp_cursor.close()
        temp_conn.close()

        # Step 2: Connect to the actual database
        self.conn = mysql.connector.connect(**MYSQL_CONFIG)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            request_uuid VARCHAR(64) UNIQUE,
            payload JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            request_uuid VARCHAR(64),
            result_json JSON,
            tools_raw JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (request_uuid)
        )
        """)
        self.conn.commit()
        cur.close()

    def log_request(self, request_uuid: str, payload: dict):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO requests (request_uuid, payload) VALUES (%s, %s)",
            (request_uuid, json.dumps(payload)),
        )
        self.conn.commit()
        cur.close()

    def log_result(self, request_uuid: str, result: dict, tools_raw: dict = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO results (request_uuid, result_json, tools_raw) VALUES (%s, %s, %s)",
            (request_uuid, json.dumps(result), json.dumps(tools_raw or {})),
        )
        self.conn.commit()
        cur.close()

    def get_result(self, request_uuid: str):
        cur = self.conn.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM results WHERE request_uuid=%s ORDER BY created_at DESC LIMIT 1",
            (request_uuid,),
        )
        r = cur.fetchone()
        cur.close()
        return r

# app/llm/openrouter_llm.py
# app/llm/openrouter_llm.py
from langchain.llms.base import LLM
from typing import Any, List, Optional
import requests
import os
from pydantic import Field

class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter free models"""

    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    model: str = Field(default="deepseek/deepseek-chat-v3.1:free")
    base_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions")

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Send user prompt to OpenRouter model"""
        if not self.openrouter_api_key:
            raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(self.base_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise ValueError(f"OpenRouter API Error: {response.status_code} - {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]





# app/services/document_fetcher.py
# app/services/document_fetcher.py
import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin, urlparse
import hashlib
import time

DOWNLOAD_DIR = "data/downloads"
SCREENER_COMPANY_URL_TEMPLATE = "https://www.screener.in/company/{ticker}/consolidated/"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def _download_file(url: str, dest_dir: str = DOWNLOAD_DIR) -> str:
    """
    Download a file and return local path. Name by SHA1(url)+basename to avoid collisions.
    """
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        # guess filename
        parsed = urlparse(url)
        base = os.path.basename(parsed.path) or "file"
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        fname = f"{url_hash}_{base}"
        local_path = os.path.join(dest_dir, fname)
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(1024*64):
                if chunk:
                    f.write(chunk)
        return local_path
    except Exception:
        return ""

def _is_pdf_link(href: str) -> bool:
    if not href:
        return False
    href = href.split('?')[0].lower()
    return href.endswith(".pdf")

def _looks_like_transcript_text(text: str) -> bool:
    if not text:
        return False
    text = text.lower()
    keys = ["transcript", "earnings call", "concall", "conference call", "management commentary", "transcribed"]
    return any(k in text for k in keys)

def fetch_quarterly_documents(ticker: str, quarters: int, sources: List[str]=None) -> Dict[str, List[Dict]]:
    """
    Scrape Screener.in company consolidated page for documents.
    Returns:
       {"reports":[{"name":..., "local_path":...}], "transcripts":[{"name":..., "local_path":...}]}
    """
    url = SCREENER_COMPANY_URL_TEMPLATE.format(ticker=ticker)
    reports = []
    transcripts = []

    try:
        headers = {"User-Agent": "tcs-forecast-agent/0.1 (+https://example.com)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Screener has a div with id 'documents' or a section — search for anchors containing '.pdf'
        # Find all anchors inside the page
        anchors = soup.find_all("a", href=True)
        pdf_links = []
        for a in anchors:
            href = a["href"]
            # absolute url
            full = urljoin(url, href)
            if _is_pdf_link(full):
                # check if anchor text suggests 'results', 'quarterly' etc
                text = (a.get_text() or "").strip()
                pdf_links.append({"href": full, "text": text})
            else:
                # also capture anchors that look like pdf but use query param
                if "pdf" in full.lower() and ".pdf" in full.lower():
                    pdf_links.append({"href": full, "text": (a.get_text() or "").strip()})

        # deduplicate by href
        seen = set()
        pdf_links_unique = []
        for p in pdf_links:
            if p["href"] not in seen:
                seen.add(p["href"])
                pdf_links_unique.append(p)

        # Sort: prefer those whose anchor text mentions 'quarter' or 'results' or 'consolidated'
        def score_pdf_link(p):
            text = p["text"].lower()
            s = 0
            if "quarter" in text or "q" in text:
                s += 2
            if "results" in text or "consolidated" in text:
                s += 2
            if "annual" in text:
                s -= 1
            return -s  # negative for reverse sort

        pdf_links_unique = sorted(pdf_links_unique, key=score_pdf_link)

        # Download top N PDFs
        for idx, p in enumerate(pdf_links_unique[:max(quarters*2, 6)]):
            local = _download_file(p["href"])
            if local:
                name = p["text"] or os.path.basename(local)
                reports.append({"name": name, "local_path": local, "source_url": p["href"]})
            time.sleep(0.5)

        # Now try to find transcripts: anchors whose text looks like transcript keywords or link targets containing 'transcript' or 'concall'
        anchors = soup.find_all("a", href=True)
        transcript_candidates = []
        for a in anchors:
            txt = (a.get_text() or "").strip()
            href = urljoin(url, a["href"])
            if _looks_like_transcript_text(txt) or 'transcript' in href.lower() or 'concall' in href.lower() or 'conference-call' in href.lower():
                transcript_candidates.append({"href": href, "text": txt})

        # De-duplicate and download if pdf; otherwise store external link metadata and try to download if pointing to a .txt or .html that looks like transcript
        seen_t = set()
        for t in transcript_candidates:
            if t["href"] in seen_t:
                continue
            seen_t.add(t["href"])
            href = t["href"]
            # If it's a PDF, download
            if _is_pdf_link(href):
                local = _download_file(href)
                if local:
                    transcripts.append({"name": t["text"] or os.path.basename(local), "local_path": local, "source_url": href})
            else:
                # try fetching the page and parse text, save as .txt
                try:
                    r2 = requests.get(href, timeout=20)
                    r2.raise_for_status()
                    soup2 = BeautifulSoup(r2.text, "html.parser")
                    # heuristics: find divs that look like transcript text
                    body_text = soup2.get_text(separator="\n")
                    # Save a local txt file
                    if len(body_text) > 200:
                        fname = os.path.join(DOWNLOAD_DIR, hashlib.sha1(href.encode()).hexdigest()[:8] + "_transcript.txt")
                        with open(fname, "w", encoding="utf-8") as f:
                            f.write(body_text)
                        transcripts.append({"name": t["text"] or href, "local_path": fname, "source_url": href})
                except Exception:
                    # skip if cannot download
                    pass

        # If transcripts empty, try third-party search fallback (DuckDuckGo unofficial via html query)
        if not transcripts:
            # naive attempt: search quick for 'TCS earnings call transcript' on google is not allowed here; skip
            pass

    except Exception:
        # If any error, fall back to returning any files in tests/data for local dev
        # Provide a fallback to local test files (developer should place sample files)
        fallback_reports = [
            {"name":"Q1_SAMPLE","local_path":"tests/data/sample_report_q1.pdf"},
            {"name":"Q4_SAMPLE","local_path":"tests/data/sample_report_q4.pdf"},
            {"name":"Q3_SAMPLE","local_path":"tests/data/sample_report_q3.pdf"},
        ]
        fallback_transcripts = [
            {"name":"Q1_TRANSCRIPT","local_path":"tests/data/sample_transcript_q1.txt"},
            {"name":"Q4_TRANSCRIPT","local_path":"tests/data/sample_transcript_q4.txt"},
            {"name":"Q3_TRANSCRIPT","local_path":"tests/data/sample_transcript.txt"}
        ]
        return {"reports": fallback_reports[:quarters], "transcripts": fallback_transcripts[:max(1, quarters-1)]}

    # Final limit to requested quarters
    return {"reports": reports[:quarters], "transcripts": transcripts[:max(1, quarters-1)]}

class DocumentFetcher:
    def __init__(self):
        pass

    def fetch_quarterly_documents(self, ticker, quarters, sources=None):
        return fetch_quarterly_documents(ticker, quarters, sources)

# app/tools/financial_extractor_tool.py
"""
app/tools/financial_extractor_tool.py - Refactored as a proper class-based tool
"""
import os
import re
import json
from typing import List, Dict, Any, Optional
import pdfplumber
from app.utils.number_parsing import parse_inr_number
from pdf2image import convert_from_path
import pytesseract

# Camelot import with fallback
try:
    import camelot
    _HAS_CAMELOT = True
except Exception:
    _HAS_CAMELOT = False


class FinancialDataExtractorTool:
    """
    Robust financial data extraction tool using multiple methods:
    1. Camelot table extraction (best for structured PDFs)
    2. pdfplumber text extraction (fallback)
    3. OCR with pytesseract (last resort)
    """
    
    def __init__(self):
        self.extraction_methods = ["camelot", "pdfplumber", "ocr"]
        
    def extract(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main extraction method called by LangChain agent
        
        Args:
            reports: List of report dicts with 'local_path', 'name', 'source_url'
            
        Returns:
            Structured dict with extracted metrics and metadata
        """
        results = []
        
        for report in reports:
            path = report.get("local_path")
            if not path or not os.path.exists(path):
                results.append({
                    "doc_meta": report,
                    "error": "file_not_found",
                    "metrics": {}
                })
                continue
            
            # Extract from this report
            extraction_result = self._extract_from_single_report(path, report)
            results.append(extraction_result)
        
        return {
            "tool": "FinancialDataExtractorTool",
            "status": "completed",
            "reports_processed": len(results),
            "results": results
        }
    
    def _extract_from_single_report(self, pdf_path: str, metadata: Dict) -> Dict[str, Any]:
        """Extract metrics from a single PDF report"""
        
        metrics = {}
        extraction_log = {
            "camelot": {"attempted": False, "metrics_found": 0, "hits": []},
            "pdfplumber": {"attempted": False, "metrics_found": 0, "snippets": []},
            "ocr": {"attempted": False, "metrics_found": 0, "text_length": 0}
        }
        
        # Method 1: Camelot table extraction
        if _HAS_CAMELOT:
            extraction_log["camelot"]["attempted"] = True
            camelot_metrics = self._extract_with_camelot(pdf_path)
            for metric in camelot_metrics:
                key = self._normalize_metric_key(metric["label"])
                if key and key not in metrics:
                    metrics[key] = {
                        "value": metric["value"],
                        "unit": metric.get("unit", "INR_Cr"),
                        "confidence": metric.get("confidence", 0.85),
                        "source": {"method": "camelot", "page": metric.get("page")},
                        "label": metric["label"]
                    }
                    extraction_log["camelot"]["metrics_found"] += 1
            extraction_log["camelot"]["hits"] = camelot_metrics
        
        # Method 2: pdfplumber text extraction (if key metrics still missing)
        required_metrics = ["total_revenue", "net_profit", "operating_profit", "ebitda"]
        missing_metrics = [m for m in required_metrics if m not in metrics]
        
        if missing_metrics:
            extraction_log["pdfplumber"]["attempted"] = True
            text = self._extract_text_with_pdfplumber(pdf_path)
            if text:
                pdfplumber_metrics = self._parse_metrics_from_text(text)
                for metric in pdfplumber_metrics:
                    key = self._normalize_metric_key(metric["label"])
                    if key and key in missing_metrics and key not in metrics:
                        metrics[key] = {
                            "value": metric["value"],
                            "unit": metric.get("unit", "INR_Cr"),
                            "confidence": 0.65,
                            "source": {"method": "pdfplumber"},
                            "label": metric["label"]
                        }
                        extraction_log["pdfplumber"]["metrics_found"] += 1
                
                extraction_log["pdfplumber"]["snippets"] = pdfplumber_metrics[:5]
        
        # Method 3: OCR (last resort if still missing critical metrics)
        critical_missing = any(m not in metrics for m in ["total_revenue", "net_profit"])
        if critical_missing:
            extraction_log["ocr"]["attempted"] = True
            ocr_text = self._extract_with_ocr(pdf_path, max_pages=5)
            if ocr_text:
                extraction_log["ocr"]["text_length"] = len(ocr_text)
                ocr_metrics = self._parse_metrics_from_text(ocr_text)
                for metric in ocr_metrics:
                    key = self._normalize_metric_key(metric["label"])
                    if key and key not in metrics:
                        metrics[key] = {
                            "value": metric["value"],
                            "unit": metric.get("unit", "INR_Cr"),
                            "confidence": 0.45,
                            "source": {"method": "ocr"},
                            "label": metric["label"]
                        }
                        extraction_log["ocr"]["metrics_found"] += 1
        
        return {
            "doc_meta": metadata,
            "metrics": metrics,
            "extraction_log": extraction_log,
            "metrics_count": len(metrics)
        }
    
    def _extract_with_camelot(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Use Camelot to extract table data"""
        results = []
        
        try:
            # Try both lattice and stream flavors
            tables = []
            for flavor in ['lattice', 'stream']:
                try:
                    tables.extend(camelot.read_pdf(pdf_path, pages='all', flavor=flavor))
                except Exception:
                    pass
            
            for table in tables:
                df = table.df
                page = table.page
                
                # Scan for financial metric labels
                for r_idx in range(df.shape[0]):
                    for c_idx in range(df.shape[1]):
                        cell = str(df.iat[r_idx, c_idx])
                        
                        # Check if this cell contains a financial label
                        if self._is_financial_label(cell):
                            # Look for numeric value in same row (to the right)
                            numeric_val = None
                            for k in range(c_idx + 1, min(df.shape[1], c_idx + 5)):
                                candidate = str(df.iat[r_idx, k])
                                val = parse_inr_number(candidate)
                                if val is not None:
                                    numeric_val = val
                                    break
                            
                            # Also check same column (below)
                            if numeric_val is None:
                                for k in range(r_idx + 1, min(df.shape[0], r_idx + 3)):
                                    candidate = str(df.iat[k, c_idx])
                                    val = parse_inr_number(candidate)
                                    if val is not None:
                                        numeric_val = val
                                        break
                            
                            if numeric_val is not None:
                                results.append({
                                    "label": cell.strip(),
                                    "value": numeric_val,
                                    "unit": "INR_Cr",
                                    "page": page,
                                    "confidence": 0.85
                                })
        except Exception:
            pass
        
        return results
    
    def _extract_text_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:10]:  # Limit to first 10 pages
                    page_text = page.extract_text()
                    if page_text:
                        text += "\n\n" + page_text
        except Exception:
            pass
        return text
    
    def _extract_with_ocr(self, pdf_path: str, dpi: int = 200, max_pages: int = 5) -> str:
        """Extract text using OCR (slowest method)"""
        text = ""
        try:
            pages = convert_from_path(pdf_path, dpi=dpi)
            for page in pages[:max_pages]:
                page_text = pytesseract.image_to_string(page)
                text += "\n\n" + page_text
        except Exception:
            pass
        return text
    
    def _parse_metrics_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse financial metrics from plain text"""
        metrics = []
        
        # Common financial labels to search for
        labels = [
            "Total Revenue", "Revenue", "Net Revenue",
            "Net Profit", "Profit After Tax", "PAT",
            "Operating Profit", "EBIT", "Operating Income",
            "EBITDA",
            "EPS", "Earnings Per Share"
        ]
        
        for label in labels:
            # Find label in text (case insensitive)
            pattern = re.escape(label)
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                # Extract surrounding context (next 300 chars)
                start = match.start()
                context = text[start:start + 300]
                
                # Try to find a number in this context
                value = parse_inr_number(context)
                if value is not None:
                    metrics.append({
                        "label": label,
                        "value": value,
                        "unit": "INR_Cr",
                        "context": context[:150]
                    })
        
        return metrics
    
    def _is_financial_label(self, text: str) -> bool:
        """Check if text looks like a financial metric label"""
        if not text or len(text) < 3:
            return False
        
        text_lower = text.lower()
        keywords = [
            "revenue", "profit", "income", "ebitda", "ebit",
            "margin", "earnings", "eps", "pat", "sales"
        ]
        
        return any(keyword in text_lower for keyword in keywords)
    
    def _normalize_metric_key(self, label: str) -> Optional[str]:
        """Normalize various label formats to standard metric keys"""
        if not label:
            return None
        
        label_lower = label.lower()
        
        # Revenue
        if "revenue" in label_lower or "sales" in label_lower:
            return "total_revenue"
        
        # Net Profit
        if ("net" in label_lower and "profit" in label_lower) or "pat" in label_lower or "profit after tax" in label_lower:
            return "net_profit"
        
        # Operating Profit
        if ("operating" in label_lower and ("profit" in label_lower or "income" in label_lower)) or label_lower == "ebit":
            return "operating_profit"
        
        # EBITDA
        if "ebitda" in label_lower:
            return "ebitda"
        
        # EPS
        if "eps" in label_lower or "earnings per share" in label_lower:
            return "eps"
        
        # Operating Margin
        if "operating" in label_lower and "margin" in label_lower:
            return "operating_margin"
        
        return None


# Legacy function wrapper for backward compatibility
def extract_financial_data(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Legacy function - delegates to the class-based tool"""
    tool = FinancialDataExtractorTool()
    return tool.extract(reports)

"""
app/tools/qualitative_analysis_tool.py - Fixed version without decorator
"""

from typing import List, Dict, Any
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class QualitativeAnalysisTool:
    """
    RAG-based qualitative analysis tool for earnings call transcripts.
    Uses sentence-transformers for embeddings and FAISS for vector search.
    """
    
    def __init__(self, embed_model_name: str = EMBED_MODEL):
        """Initialize the tool with an embedding model"""
        self.embedder = SentenceTransformer(embed_model_name)
        self.index = None
        self.chunks = []
    
    def _chunk_text(self, text: str, chunk_words: int = 300) -> List[str]:
        """
        Split text into chunks of approximately chunk_words length.
        
        Args:
            text: Input text to chunk
            chunk_words: Target number of words per chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_words):
            chunk = " ".join(words[i:i + chunk_words])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def index_transcripts(self, transcripts: List[Dict[str, Any]]) -> bool:
        """
        Build FAISS index from transcript documents.
        
        Args:
            transcripts: List of dicts with 'name' and 'local_path' keys
            
        Returns:
            True if successful, False otherwise
        """
        self.chunks = []
        texts = []
        
        for transcript in transcripts:
            path = transcript.get("local_path")
            if not path:
                continue
                
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue
            
            # Chunk the transcript
            chunks = self._chunk_text(text, chunk_words=300)
            
            for i, chunk in enumerate(chunks):
                meta = {
                    "source": transcript.get("name", "unknown"),
                    "chunk_id": f"{transcript.get('name', 'unknown')}_chunk_{i}"
                }
                self.chunks.append({"meta": meta, "text": chunk})
                texts.append(chunk)
        
        if not texts:
            return False
        
        # Generate embeddings
        try:
            embeddings = self.embedder.encode(texts, show_progress_bar=False)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            
            return True
        except Exception:
            return False
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dicts with chunk_id, source, text, and score
        """
        if self.index is None or not self.chunks:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedder.encode([query])
            
            # Search
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), 
                min(top_k, len(self.chunks))
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append({
                        "chunk_id": chunk["meta"]["chunk_id"],
                        "source": chunk["meta"]["source"],
                        "text": chunk["text"][:600],  # Truncate for readability
                        "score": float(distance)
                    })
            
            return results
        except Exception:
            return []
    
    def analyze(self, transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main analysis method: indexes transcripts and extracts insights.
        
        Args:
            transcripts: List of transcript dicts with 'name' and 'local_path'
            
        Returns:
            Dict with themes, sentiment, forward_guidance, and risks
        """
        # Index the transcripts
        success = self.index_transcripts(transcripts)
        
        if not success:
            return {
                "tool": "QualitativeAnalysisTool",
                "themes": [],
                "management_sentiment": {
                    "score": 0.0,
                    "summary": "insufficient_data"
                },
                "forward_guidance": [],
                "risks": []
            }
        
        # Define search queries for different themes
        theme_queries = {
            "demand": "demand, growth, digital transformation, revenue growth, market demand",
            "attrition": "attrition, employee turnover, resignations, hiring, talent, retention",
            "guidance": "guidance, outlook, expect, forecast, projection, next quarter",
            "margins": "margin, profitability, costs, efficiency, operating margin",
            "deals": "deals, pipeline, bookings, wins, contracts, clients"
        }
        
        # Retrieve chunks for each theme
        themes = []
        for theme_name, query in theme_queries.items():
            results = self.retrieve(query, top_k=5)
            if results:
                themes.append({
                    "theme": theme_name,
                    "count": len(results),
                    "examples": results[:3]  # Top 3 examples
                })
        
        # Basic sentiment analysis (rule-based)
        sentiment_score = 0.0
        sentiment_summary = "neutral"
        
        # Positive indicators
        positive_queries = ["strong performance", "growth", "optimistic", "positive"]
        negative_queries = ["challenges", "headwinds", "concerns", "pressure"]
        
        positive_count = 0
        negative_count = 0
        
        for query in positive_queries:
            results = self.retrieve(query, top_k=3)
            positive_count += len(results)
        
        for query in negative_queries:
            results = self.retrieve(query, top_k=3)
            negative_count += len(results)
        
        # Calculate sentiment
        if positive_count > negative_count:
            sentiment_score = min(0.8, positive_count / max(positive_count + negative_count, 1))
            sentiment_summary = "positive" if sentiment_score > 0.6 else "cautiously optimistic"
        elif negative_count > positive_count:
            sentiment_score = -min(0.8, negative_count / max(positive_count + negative_count, 1))
            sentiment_summary = "negative" if sentiment_score < -0.6 else "cautious"
        else:
            sentiment_score = 0.0
            sentiment_summary = "neutral"
        
        # Extract forward guidance
        forward_guidance_results = self.retrieve(
            "guidance, outlook, expect, forecast, next quarter, full year", 
            top_k=5
        )
        
        # Identify risks
        risks = []
        risk_themes = ["attrition", "competition", "macro", "regulation"]
        for theme_name in risk_themes:
            theme_data = next((t for t in themes if t["theme"] == theme_name), None)
            if theme_data and theme_data["count"] > 0:
                risks.append({
                    "name": theme_name,
                    "evidence": [ex["chunk_id"] for ex in theme_data["examples"]]
                })
        
        return {
            "tool": "QualitativeAnalysisTool",
            "themes": themes,
            "management_sentiment": {
                "score": sentiment_score,
                "summary": sentiment_summary
            },
            "forward_guidance": forward_guidance_results,
            "risks": risks
        }
    

"""
app/utils/number_parsing.py - Fixed to handle Indian number format
"""
import re


def parse_inr_number(text: str):
    """
    Parse Indian Rupee numbers from text.
    Handles formats like:
    - ₹ 12,345.67 Cr
    - 1,23,456 (Indian format)
    - 123,456 (Western format)
    - 9876.54
    """
    if not text:
        return None
    
    # Pattern 1: ₹ symbol with number
    m = re.search(r'₹\s*([0-9,\.]+)\s*(Cr|Crore|CR|cr|Million|Mn)?', text)
    if m:
        num_str = m.group(1)
        try:
            # Remove all commas and parse
            return float(num_str.replace(',', ''))
        except:
            return None
    
    # Pattern 2: Indian number format (1,23,456 or 12,34,567)
    # Indian format has commas every 2 digits after the first 3
    m2 = re.search(r'\b([0-9]{1,3}(?:,[0-9]{2})+(?:,[0-9]{3})?)\b', text)
    if m2:
        num_str = m2.group(1)
        try:
            return float(num_str.replace(',', ''))
        except:
            return None
    
    # Pattern 3: Western number format with commas (123,456 or 1,234,567)
    m3 = re.search(r'\b([0-9]{1,3}(?:,[0-9]{3})+)\b', text)
    if m3:
        num_str = m3.group(1)
        try:
            return float(num_str.replace(',', ''))
        except:
            return None
    
    # Pattern 4: Plain number (no commas)
    m4 = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\b', text)
    if m4:
        try:
            return float(m4.group(1))
        except:
            return None
    
    return None

check_response:
"""
check_response.py - Analyze the actual API response
"""
import json
import glob

# Find the most recent test output
files = glob.glob("test_output_*.json")
if not files:
    print("No test output files found")
    exit(1)

latest = max(files)
print(f"Analyzing: {latest}")
print("=" * 60)

with open(latest, 'r') as f:
    data = json.load(f)

print("\nTop-level keys:")
for key in data.keys():
    print(f"  - {key}")

print("\nFull structure (first 100 lines):")
output = json.dumps(data, indent=2)
lines = output.split('\n')
for i, line in enumerate(lines[:100]):
    print(line)
    if i == 99 and len(lines) > 100:
        print(f"\n... ({len(lines) - 100} more lines)")

# Check for error
if 'error' in data:
    print("\n" + "=" * 60)
    print("ERROR DETECTED:")
    print("=" * 60)
    print(f"Error type: {data.get('error')}")
    print(f"Message: {data.get('message', 'No message')}")
    
    if 'synthesis' in data and isinstance(data['synthesis'], dict):
        if 'error' in data['synthesis']:
            print(f"\nSynthesis error: {data['synthesis'].get('error')}")
            print(f"Details: {data['synthesis'].get('last_exc')}")
    
    if 'synthesis_attempts' in data:
        print(f"\nSynthesis attempts: {len(data['synthesis_attempts'])}")
        for i, attempt in enumerate(data['synthesis_attempts'], 1):
            print(f"\n  Attempt {i}:")
            print(f"    Raw output (first 200 chars): {attempt.get('raw', '')[:200]}")





"""
check_server.py - See what's happening when server starts
"""
import subprocess
import time
import sys

print("=" * 60)
print("Starting server and capturing output...")
print("=" * 60)
print()

# Start server and capture output
process = subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

print("Server starting... (will show output for 30 seconds)")
print("If you see the error, press CTRL+C")
print("-" * 60)
print()

try:
    start_time = time.time()
    while time.time() - start_time < 30:
        line = process.stdout.readline()
        if line:
            print(line.rstrip())
        
        # Check if process died
        if process.poll() is not None:
            # Process ended
            remaining = process.stdout.read()
            if remaining:
                print(remaining)
            print()
            print("=" * 60)
            print("Server process ended!")
            print("=" * 60)
            sys.exit(1)
        
        # Check if server started successfully
        if "Application startup complete" in line:
            print()
            print("=" * 60)
            print("✓ Server started successfully!")
            print("=" * 60)
            print()
            print("Press CTRL+C to stop")
            # Keep running
            process.wait()
            break
        
        time.sleep(0.1)
    
except KeyboardInterrupt:
    print("\n\nStopping server...")
    process.terminate()
    process.wait()
    print("Server stopped.")

"""
debug_server.py - Diagnose server startup issues
"""
import sys
import os
from pathlib import Path

print("=" * 60)
print("TCS Forecast Agent - Server Diagnostic Tool")
print("=" * 60)

# Check 1: Python version
print("\n[1] Python Version")
print(f"    Version: {sys.version}")
if sys.version_info < (3, 10):
    print("    ⚠️  WARNING: Python 3.10+ recommended")
else:
    print("    ✓ OK")

# Check 2: Working directory
print("\n[2] Working Directory")
print(f"    Path: {os.getcwd()}")
if Path("app/main.py").exists():
    print("    ✓ app/main.py found")
else:
    print("    ✗ app/main.py NOT found - are you in the project root?")
    sys.exit(1)

# Check 3: Required modules
print("\n[3] Required Modules")
required_modules = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "langchain",
    "mysql.connector",
    "requests",
    "dotenv"
]

missing_modules = []
for module in required_modules:
    try:
        if module == "mysql.connector":
            __import__("mysql.connector")
        elif module == "dotenv":
            __import__("dotenv")
        else:
            __import__(module)
        print(f"    ✓ {module}")
    except ImportError:
        print(f"    ✗ {module} - NOT INSTALLED")
        missing_modules.append(module)

if missing_modules:
    print(f"\n    ⚠️  Missing modules: {', '.join(missing_modules)}")
    print(f"    Run: pip install {' '.join(missing_modules)}")
    sys.exit(1)

# Check 4: Environment file
print("\n[4] Environment Configuration")
if Path(".env").exists():
    print("    ✓ .env file found")
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check critical variables
    env_vars = {
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "MYSQL_HOST": os.getenv("MYSQL_HOST", "localhost"),
        "MYSQL_USER": os.getenv("MYSQL_USER", "root"),
        "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD"),
        "MYSQL_DB": os.getenv("MYSQL_DB", "tcs_forecast")
    }
    
    for key, value in env_vars.items():
        if value:
            if "PASSWORD" in key or "KEY" in key:
                print(f"    ✓ {key}: ***hidden***")
            else:
                print(f"    ✓ {key}: {value}")
        else:
            print(f"    ⚠️  {key}: NOT SET")
else:
    print("    ⚠️  .env file NOT found")
    print("    Create .env from .env.example")

# Check 5: MySQL Connection
print("\n[5] MySQL Connection")
try:
    import mysql.connector
    from dotenv import load_dotenv
    load_dotenv()
    
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        connect_timeout=5
    )
    print("    ✓ MySQL connection successful")
    
    # Check if database exists
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES LIKE 'tcs_forecast'")
    if cursor.fetchone():
        print("    ✓ Database 'tcs_forecast' exists")
    else:
        print("    ⚠️  Database 'tcs_forecast' does NOT exist")
        print("    Creating database...")
        cursor.execute("CREATE DATABASE tcs_forecast")
        print("    ✓ Database created")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"    ✗ MySQL connection failed: {str(e)}")
    print("    Troubleshooting:")
    print("    - Check MySQL is running: sudo systemctl status mysql")
    print("    - Verify credentials in .env")
    print("    - Try: mysql -u root -p")

# Check 6: Try importing app
print("\n[6] Application Import")
try:
    from app.main import app
    print("    ✓ Successfully imported app.main")
except Exception as e:
    print(f"    ✗ Failed to import app.main: {str(e)}")
    print(f"\n    Full error:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 7: Port availability
print("\n[7] Port Availability")
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

ports_to_check = [8000, 8080, 8082]
for port in ports_to_check:
    if is_port_in_use(port):
        print(f"    ⚠️  Port {port} is already in use")
    else:
        print(f"    ✓ Port {port} is available")

# Check 8: Try starting server
print("\n[8] Server Startup Test")
print("    Attempting to start server on port 8082...")
print("    (This will run for 5 seconds then stop)")

import subprocess
import time

try:
    # Start server
    process = subprocess.Popen(
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a bit
    time.sleep(3)
    
    # Check if it's running
    if process.poll() is None:
        print("    ✓ Server started successfully!")
        
        # Try to hit health endpoint
        import requests
        try:
            response = requests.get("http://localhost:8082/health", timeout=2)
            if response.status_code == 200:
                print("    ✓ Health endpoint responding")
            else:
                print(f"    ⚠️  Health endpoint returned status {response.status_code}")
        except Exception as e:
            print(f"    ⚠️  Could not reach health endpoint: {e}")
        
        # Stop server
        process.terminate()
        process.wait(timeout=5)
        print("    ✓ Server stopped cleanly")
        
    else:
        # Process died
        stdout, stderr = process.communicate()
        print("    ✗ Server failed to start")
        print(f"\n    STDOUT:\n{stdout}")
        print(f"\n    STDERR:\n{stderr}")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ Error during startup test: {e}")
    import traceback
    traceback.print_exc()
    if 'process' in locals():
        process.terminate()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All diagnostics passed!")
print("=" * 60)
print("\nYour server should work. Try:")
print("  uvicorn app.main:app --host 0.0.0.0 --port 8082")
print("\nOr run the manual test again:")
print("  python test_api_manual.py")

"""
tests/test_agent_flow.py - Comprehensive integration tests
"""
import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.forecast_agent import ForecastAgent, FORECAST_SCHEMA
from app.tools.financial_extractor_tool import FinancialDataExtractorTool
from app.tools.qualitative_analysis_tool import QualitativeAnalysisTool
from app.utils.number_parsing import parse_inr_number
from jsonschema import validate


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestNumberParsing:
    """Test the number parsing utility"""
    
    def test_parse_inr_with_rupee_symbol(self):
        assert parse_inr_number("₹ 12,345.67 Cr") == 12345.67
    
    def test_parse_inr_with_commas(self):
        assert parse_inr_number("1,23,456") == 123456
    
    def test_parse_inr_plain_number(self):
        assert parse_inr_number("9876.54") == 9876.54
    
    def test_parse_inr_invalid(self):
        assert parse_inr_number("No numbers here") is None
        assert parse_inr_number("") is None
        assert parse_inr_number(None) is None


class TestFinancialExtractor:
    """Test the financial data extraction tool"""
    
    @pytest.fixture
    def extractor(self):
        return FinancialDataExtractorTool()
    
    @pytest.fixture
    def sample_reports(self):
        """Create sample report metadata"""
        return [
            {
                "name": "Q3-FY24-Results",
                "local_path": "tests/data/sample_report_q3.pdf",
                "source_url": "https://example.com/q3.pdf"
            },
            {
                "name": "Q2-FY24-Results",
                "local_path": "tests/data/sample_report_q2.pdf",
                "source_url": "https://example.com/q2.pdf"
            }
        ]
    
    def test_extractor_initialization(self, extractor):
        assert extractor is not None
        assert "camelot" in extractor.extraction_methods
        assert "pdfplumber" in extractor.extraction_methods
        assert "ocr" in extractor.extraction_methods
    
    def test_normalize_metric_key(self, extractor):
        assert extractor._normalize_metric_key("Total Revenue") == "total_revenue"
        assert extractor._normalize_metric_key("Net Profit") == "net_profit"
        assert extractor._normalize_metric_key("EBITDA") == "ebitda"
        assert extractor._normalize_metric_key("Operating Profit") == "operating_profit"
        assert extractor._normalize_metric_key("EPS") == "eps"
        assert extractor._normalize_metric_key("Random Text") is None
    
    def test_is_financial_label(self, extractor):
        assert extractor._is_financial_label("Total Revenue") is True
        assert extractor._is_financial_label("Net Profit") is True
        assert extractor._is_financial_label("EBITDA") is True
        assert extractor._is_financial_label("Random Text") is False
        assert extractor._is_financial_label("") is False
    
    def test_extract_structure(self, extractor, sample_reports):
        """Test that extract returns proper structure even with missing files"""
        result = extractor.extract(sample_reports)
        
        assert result["tool"] == "FinancialDataExtractorTool"
        assert result["status"] == "completed"
        assert "results" in result
        assert len(result["results"]) == len(sample_reports)
        
        # Check each result has required fields
        for res in result["results"]:
            assert "doc_meta" in res
            assert "metrics" in res
            assert isinstance(res["metrics"], dict)


class TestQualitativeAnalyzer:
    """Test the qualitative analysis tool"""
    
    @pytest.fixture
    def analyzer(self):
        return QualitativeAnalysisTool()
    
    @pytest.fixture
    def sample_transcripts(self):
        """Create sample transcript metadata"""
        # Create temporary transcript files for testing
        os.makedirs("tests/data", exist_ok=True)
        
        transcript1_path = "tests/data/test_transcript_1.txt"
        with open(transcript1_path, "w") as f:
            f.write("""
            Q3 FY24 Earnings Call Transcript
            
            Management Commentary:
            We are pleased to report strong demand across all verticals.
            Digital transformation continues to be a key driver.
            Attrition has stabilized at 12.5% this quarter.
            
            Guidance:
            We expect 5-7% revenue growth for the full year.
            Focus on AI and GenAI investments will continue.
            
            Risks:
            Macro headwinds remain a concern.
            Competition in the market is intensifying.
            """)
        
        return [
            {"name": "Q3-Transcript", "local_path": transcript1_path}
        ]
    
    def test_analyzer_initialization(self, analyzer):
        assert analyzer is not None
        assert analyzer.embedder is not None
        assert analyzer.index is None  # Not built yet
        assert analyzer.chunks == []
    
    def test_chunk_text(self, analyzer):
        text = " ".join([f"word{i}" for i in range(500)])
        chunks = analyzer._chunk_text(text, chunk_words=100)
        
        assert len(chunks) == 5
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_index_and_retrieve(self, analyzer, sample_transcripts):
        # Index the transcripts
        success = analyzer.index_transcripts(sample_transcripts)
        assert success is True
        assert analyzer.index is not None
        assert len(analyzer.chunks) > 0
        
        # Test retrieval
        results = analyzer.retrieve("demand and growth", top_k=3)
        assert len(results) <= 3
        assert all("chunk_id" in r and "text" in r for r in results)
    
    def test_analyze_output_structure(self, analyzer, sample_transcripts):
        result = analyzer.analyze(sample_transcripts)
        
        assert result["tool"] == "QualitativeAnalysisTool"
        assert "themes" in result
        assert "management_sentiment" in result
        assert "forward_guidance" in result
        assert "risks" in result
        
        # Check sentiment structure
        sentiment = result["management_sentiment"]
        assert "score" in sentiment
        assert "summary" in sentiment
        assert isinstance(sentiment["score"], (int, float))


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestForecastAgentIntegration:
    """Test the complete agent flow"""
    
    @pytest.fixture
    def agent(self):
        return ForecastAgent()
    
    @pytest.fixture
    def mock_documents(self):
        """Create mock document structure"""
        return {
            "reports": [
                {
                    "name": "Q3-FY24-Results",
                    "local_path": "tests/data/sample_report_q3.pdf",
                    "source_url": "https://example.com/q3.pdf"
                }
            ],
            "transcripts": [
                {
                    "name": "Q3-Transcript",
                    "local_path": "tests/data/test_transcript_1.txt"
                }
            ]
        }
    
    def test_agent_initialization(self, agent):
        assert agent is not None
        assert agent.llm is not None
        assert agent.financial_tool_instance is not None
        assert agent.qualitative_tool_instance is not None
    
    def test_create_tools(self, agent, mock_documents):
        tools = agent._create_tools(mock_documents)
        
        assert len(tools) == 2
        assert tools[0].name == "FinancialDataExtractorTool"
        assert tools[1].name == "QualitativeAnalysisTool"
        assert callable(tools[0].func)
        assert callable(tools[1].func)
    
    def test_tool_execution(self, agent, mock_documents):
        """Test that tools can be called and return proper JSON"""
        tools = agent._create_tools(mock_documents)
        
        # Test financial tool
        fin_tool = tools[0]
        fin_result = fin_tool.func('{"action": "extract_all"}')
        fin_data = json.loads(fin_result)
        
        assert "status" in fin_data
        assert "reports_analyzed" in fin_data or "error" in fin_data
        
        # Test qualitative tool
        qual_tool = tools[1]
        qual_result = qual_tool.func('{"action": "analyze_all"}')
        qual_data = json.loads(qual_result)
        
        assert "status" in qual_data
        assert "transcripts_analyzed" in qual_data or "error" in qual_data
    
    @patch('app.services.document_fetcher.DocumentFetcher')
    def test_agent_run_with_mock_documents(self, mock_fetcher_class, agent):
        """Test full agent run with mocked document fetching"""
        
        # Load environment variables for API key
        from dotenv import load_dotenv
        load_dotenv()
        
        # Skip test if no API key
        import os
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set in environment")
        
        # Setup mock
        mock_fetcher = Mock()
        mock_fetcher.fetch_quarterly_documents.return_value = {
            "reports": [
                {"name": "Q3-Report", "local_path": "tests/data/sample.pdf"}
            ],
            "transcripts": [
                {"name": "Q3-Transcript", "local_path": "tests/data/sample.txt"}
            ]
        }
        mock_fetcher_class.return_value = mock_fetcher
        
        # Run agent
        result = agent.run(
            ticker="TCS",
            request_id="test-123",
            quarters=3
        )
        
        # Verify structure
        assert "metadata" in result
        assert result["metadata"]["ticker"] == "TCS"
        assert result["metadata"]["request_id"] == "test-123"
        assert "agent_execution" in result or "error" in result
    
    def test_validate_and_repair_forecast(self, agent):
        """Test forecast validation and repair logic"""
        
        # Valid forecast
        valid_forecast = json.dumps({
            "metadata": {
                "ticker": "TCS",
                "request_id": "test-123",
                "analysis_date": "2024-10-28T00:00:00",
                "quarters_analyzed": ["Q1", "Q2", "Q3"]
            },
            "numeric_trends": {},
            "qualitative_summary": {},
            "forecast": {},
            "risks_and_opportunities": {},
            "sources": []
        })
        
        result = agent._validate_and_repair_forecast(
            valid_forecast,
            context={},
            max_attempts=1
        )
        
        assert result["status"] == "success"
        assert "forecast" in result
        
        # Invalid JSON
        invalid_json = "This is not JSON {incomplete"
        result = agent._validate_and_repair_forecast(
            invalid_json,
            context={},
            max_attempts=1
        )
        
        assert "error" in result


# ============================================================================
# END-TO-END TESTS
# ============================================================================

class TestEndToEndFlow:
    """Test the complete end-to-end flow"""
    
    def test_schema_validation(self):
        """Test that our schema is valid"""
        sample_forecast = {
            "metadata": {
                "ticker": "TCS",
                "request_id": "e2e-test",
                "analysis_date": "2024-10-28T10:00:00Z",
                "quarters_analyzed": ["Q1-FY24", "Q2-FY24", "Q3-FY24"]
            },
            "numeric_trends": {
                "total_revenue": {
                    "values": [
                        {"period": "Q1-FY24", "value": 58229, "unit": "INR_Cr"},
                        {"period": "Q2-FY24", "value": 59162, "unit": "INR_Cr"},
                        {"period": "Q3-FY24", "value": 60583, "unit": "INR_Cr"}
                    ],
                    "trend": "increasing",
                    "qoq_change_pct": 2.4
                },
                "net_profit": {
                    "values": [
                        {"period": "Q1-FY24", "value": 11074, "unit": "INR_Cr"},
                        {"period": "Q2-FY24", "value": 11342, "unit": "INR_Cr"},
                        {"period": "Q3-FY24", "value": 11735, "unit": "INR_Cr"}
                    ],
                    "trend": "increasing",
                    "qoq_change_pct": 3.5
                }
            },
            "qualitative_summary": {
                "themes": ["Digital transformation demand", "Attrition stabilizing", "AI/GenAI investments"],
                "management_sentiment": {
                    "score": 0.65,
                    "summary": "Cautiously optimistic with strong demand signals"
                },
                "forward_guidance": [
                    "Expecting 5-7% revenue growth for FY24",
                    "Focus on AI and automation capabilities",
                    "Deal pipeline remains robust"
                ]
            },
            "forecast": {
                "outlook_text": "Based on consistent QoQ revenue growth and improving margins, TCS is expected to maintain momentum in Q4-FY24.",
                "numeric_projection": {
                    "metric": "revenue",
                    "low": 61000,
                    "high": 62500,
                    "unit": "INR_Cr"
                },
                "confidence": 0.72
            },
            "risks_and_opportunities": {
                "risks": [
                    "Macro economic headwinds in key markets",
                    "Competition intensifying in AI space",
                    "Client budget constraints"
                ],
                "opportunities": [
                    "GenAI adoption accelerating",
                    "Digital transformation initiatives",
                    "Cloud migration projects"
                ]
            },
            "sources": [
                {
                    "tool": "FinancialDataExtractorTool",
                    "document": "Q3-FY24-Results.pdf",
                    "location": "page 2"
                },
                {
                    "tool": "QualitativeAnalysisTool",
                    "document": "Q3-Earnings-Transcript.txt",
                    "location": "chunk_5"
                }
            ]
        }
        
        # This should not raise any exceptions
        validate(instance=sample_forecast, schema=FORECAST_SCHEMA)
        assert True  # If we got here, validation passed
    
    @pytest.mark.integration
    @patch('app.llm.openrouter_llm.requests.post')
    def test_full_api_flow_mock(self, mock_post):
        """Test full flow with mocked LLM responses"""
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "metadata": {
                            "ticker": "TCS",
                            "request_id": "mock-test",
                            "analysis_date": "2024-10-28T00:00:00",
                            "quarters_analyzed": ["Q3"]
                        },
                        "numeric_trends": {},
                        "qualitative_summary": {},
                        "forecast": {},
                        "risks_and_opportunities": {},
                        "sources": []
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        from app.agents.forecast_agent import ForecastAgent
        agent = ForecastAgent()
        
        # This would normally call the real API, but we've mocked it
        # In a real test, you'd patch the document fetcher too
        assert agent is not None


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_chunking_performance(self):
        """Test that chunking large texts is reasonably fast"""
        import time
        
        analyzer = QualitativeAnalysisTool()
        
        # Generate large text (100k words)
        large_text = " ".join([f"word{i}" for i in range(100000)])
        
        start = time.time()
        chunks = analyzer._chunk_text(large_text, chunk_words=300)
        duration = time.time() - start
        
        assert len(chunks) > 0
        assert duration < 5.0  # Should complete in under 5 seconds
    
    def test_embedding_performance(self):
        """Test that embedding generation is reasonably fast"""
        import time
        
        analyzer = QualitativeAnalysisTool()
        
        # Create moderate-sized transcripts
        transcripts = [
            {
                "name": f"transcript_{i}",
                "local_path": "tests/data/test_transcript_1.txt"
            }
            for i in range(3)
        ]
        
        start = time.time()
        success = analyzer.index_transcripts(transcripts)
        duration = time.time() - start
        
        assert success
        assert duration < 30.0  # Should complete in under 30 seconds


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""
    
    # Create test data directory
    os.makedirs("tests/data", exist_ok=True)
    
    # Create sample transcript
    transcript_path = "tests/data/test_transcript_1.txt"
    if not os.path.exists(transcript_path):
        with open(transcript_path, "w") as f:
            f.write("""
            TCS Q3 FY24 Earnings Call Transcript
            
            Management: We are pleased with our performance this quarter.
            Revenue grew by 3.2% quarter-over-quarter.
            Digital transformation demand remains strong.
            Attrition has stabilized at 12.5%.
            
            Q&A Session:
            Analyst: What is your outlook for next quarter?
            Management: We expect continued growth in the 5-7% range.
            Focus on AI and GenAI will drive future revenue.
            """)
    
    yield
    
    # Cleanup (optional)
    # shutil.rmtree("tests/data", ignore_errors=True)


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_agent_flow.py -v
    pytest.main([__file__, "-v", "--tb=short"])

"""
test_api_manual.py - Manual testing script for the FastAPI application

Usage:
    python test_api_manual.py

This script will:
1. Start the FastAPI server in a subprocess
2. Wait for it to be ready
3. Send test requests
4. Display results
5. Clean up
"""

import requests
import json
import time
import subprocess
import sys
import os
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8082"  # Changed to 8082
TIMEOUT = 300  # 5 minutes for agent execution


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.END}")
    print(f"{Colors.HEADER}{title}{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.END}\n")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def print_info(message):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {message}{Colors.END}")


def wait_for_server(url, timeout=300):
    """Wait for the FastAPI server to be ready"""
    print_info(f"Waiting for server at {url}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print_success(f"Server is ready! ({response.json()})")
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    
    print_error(f"Server did not become ready within {timeout} seconds")
    return False


def test_health_endpoint():
    """Test the health check endpoint"""
    print_section("Test 1: Health Check Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print_success("Health check passed")
            return True
        else:
            print_error("Health check failed")
            return False
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        return False


def test_forecast_endpoint_minimal():
    """Test the forecast endpoint with minimal parameters"""
    print_section("Test 2: Forecast Endpoint (Minimal Request)")
    
    payload = {
        "quarters": 2,
        "sources": ["screener"],
        "include_market": False
    }
    
    print(f"Request Payload:\n{json.dumps(payload, indent=2)}")
    print_info("Sending request... (this may take 1-3 minutes)")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/api/forecast/tcs",
            json=payload,
            timeout=TIMEOUT
        )
        duration = time.time() - start_time
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Duration: {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            
            # Pretty print key sections
            print(f"\n{Colors.BOLD}Response Structure:{Colors.END}")
            print(f"- metadata: {Colors.GREEN}✓{Colors.END}" if "metadata" in result else f"- metadata: {Colors.RED}✗{Colors.END}")
            print(f"- agent_execution: {Colors.GREEN}✓{Colors.END}" if "agent_execution" in result else f"- agent_execution: {Colors.RED}✗{Colors.END}")
            print(f"- forecast: {Colors.GREEN}✓{Colors.END}" if "forecast" in result else f"- forecast: {Colors.RED}✗{Colors.END}")
            print(f"- documents_processed: {Colors.GREEN}✓{Colors.END}" if "documents_processed" in result else f"- documents_processed: {Colors.RED}✗{Colors.END}")
            
            # Print metadata
            if "metadata" in result:
                print(f"\n{Colors.BOLD}Metadata:{Colors.END}")
                print(json.dumps(result["metadata"], indent=2))
            
            # Print agent execution summary
            if "agent_execution" in result:
                print(f"\n{Colors.BOLD}Agent Execution:{Colors.END}")
                ae = result["agent_execution"]
                print(f"- Tool calls: {ae.get('tool_calls', 0)}")
                print(f"- Iterations: {ae.get('iterations', 0)}")
                print(f"- Intermediate steps: {ae.get('intermediate_steps_count', 0)}")
            
            # Print forecast summary
            if result.get("forecast"):
                print(f"\n{Colors.BOLD}Forecast Preview:{Colors.END}")
                forecast = result["forecast"]
                if "qualitative_summary" in forecast:
                    qs = forecast["qualitative_summary"]
                    print(f"- Themes: {qs.get('themes', [])}")
                    print(f"- Sentiment: {qs.get('management_sentiment', {})}")
            
            # Save full response
            output_file = f"test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n{Colors.CYAN}Full response saved to: {output_file}{Colors.END}")
            
            # Get request_id for status check
            request_id = result.get("metadata", {}).get("request_id")
            if request_id:
                return True, request_id
            
            print_success("Forecast endpoint test passed")
            return True, None
        else:
            print_error(f"Forecast endpoint test failed: {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print_error(f"Request timed out after {TIMEOUT} seconds")
        return False, None
    except Exception as e:
        print_error(f"Forecast endpoint error: {str(e)}")
        return False, None


def test_status_endpoint(request_id):
    """Test the status endpoint"""
    print_section("Test 3: Status Endpoint")
    
    if not request_id:
        print_info("Skipping - no request_id available")
        return False
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/status/{request_id}",
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response:\n{json.dumps(result, indent=2)}")
            print_success("Status endpoint test passed")
            return True
        else:
            print_error(f"Status endpoint test failed: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Status endpoint error: {str(e)}")
        return False


def test_forecast_endpoint_full():
    """Test the forecast endpoint with full parameters"""
    print_section("Test 4: Forecast Endpoint (Full Request)")
    
    payload = {
        "quarters": 3,
        "sources": ["screener", "company-ir"],
        "include_market": True
    }
    
    print(f"Request Payload:\n{json.dumps(payload, indent=2)}")
    print_info("Sending request... (this may take 2-4 minutes)")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/api/forecast/tcs",
            json=payload,
            timeout=TIMEOUT
        )
        duration = time.time() - start_time
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Duration: {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            
            # Check for forecast data
            if result.get("forecast"):
                forecast = result["forecast"]
                print(f"\n{Colors.BOLD}Forecast Quality Checks:{Colors.END}")
                
                # Check for required sections
                checks = {
                    "numeric_trends": "numeric_trends" in forecast,
                    "qualitative_summary": "qualitative_summary" in forecast,
                    "forecast_projection": "forecast" in forecast,
                    "risks_and_opportunities": "risks_and_opportunities" in forecast,
                    "sources": "sources" in forecast
                }
                
                for check_name, passed in checks.items():
                    status = f"{Colors.GREEN}✓{Colors.END}" if passed else f"{Colors.RED}✗{Colors.END}"
                    print(f"{status} {check_name}")
                
                # Save full response
                output_file = f"test_output_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n{Colors.CYAN}Full response saved to: {output_file}{Colors.END}")
                
                print_success("Full forecast endpoint test passed")
                return True
            else:
                print_error("Forecast data missing in response")
                return False
        else:
            print_error(f"Forecast endpoint test failed: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print_error(f"Request timed out after {TIMEOUT} seconds")
        return False
    except Exception as e:
        print_error(f"Forecast endpoint error: {str(e)}")
        return False


def verify_database_logging():
    """Verify that data was logged to MySQL"""
    print_section("Test 5: Database Logging Verification")
    
    try:
        import mysql.connector
        from dotenv import load_dotenv
        
        load_dotenv()
        
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DB", "tcs_forecast")
        )
        
        cursor = conn.cursor(dictionary=True)
        
        # Check requests table
        cursor.execute("SELECT COUNT(*) as count FROM requests")
        request_count = cursor.fetchone()["count"]
        print(f"Requests logged: {request_count}")
        
        # Check results table
        cursor.execute("SELECT COUNT(*) as count FROM results")
        result_count = cursor.fetchone()["count"]
        print(f"Results logged: {result_count}")
        
        # Get latest request
        cursor.execute("""
            SELECT request_uuid, created_at 
            FROM requests 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        latest = cursor.fetchone()
        if latest:
            print(f"\nLatest request:")
            print(f"- UUID: {latest['request_uuid']}")
            print(f"- Created: {latest['created_at']}")
        
        cursor.close()
        conn.close()
        
        if request_count > 0 and result_count > 0:
            print_success("Database logging verified")
            return True
        else:
            print_error("No data found in database")
            return False
            
    except ImportError:
        print_info("mysql-connector-python not installed, skipping database check")
        return None
    except Exception as e:
        print_error(f"Database verification error: {str(e)}")
        return False


def run_all_tests():
    """Run all tests in sequence"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     TCS Financial Forecasting Agent - API Test Suite      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Check if server is already running
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=2)
        print_info("Server is already running")
        server_was_running = True
        server_process = None
    except requests.exceptions.RequestException:
        print_info("Starting FastAPI server...")
        server_was_running = False
        
        # Start server in background
        server_process = subprocess.Popen(
            ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready
        if not wait_for_server(API_BASE_URL):
            print_error("Failed to start server")
            if server_process:
                server_process.terminate()
            return False
    
    # Run tests
    results = []
    
    try:
        # Test 1: Health check
        results.append(("Health Check", test_health_endpoint()))
        
        # Test 2: Minimal forecast request
        test_passed, request_id = test_forecast_endpoint_minimal()
        results.append(("Minimal Forecast", test_passed))
        
        # Test 3: Status check
        if request_id:
            results.append(("Status Check", test_status_endpoint(request_id)))
        
        # Test 4: Full forecast request
        results.append(("Full Forecast", test_forecast_endpoint_full()))
        
        # Test 5: Database verification
        db_result = verify_database_logging()
        if db_result is not None:
            results.append(("Database Logging", db_result))
        
    finally:
        # Cleanup: stop server if we started it
        if not server_was_running and server_process:
            print_info("\nStopping FastAPI server...")
            server_process.terminate()
            server_process.wait(timeout=5)
            print_success("Server stopped")
    
    # Print summary
    print_section("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Colors.GREEN}PASSED{Colors.END}" if result else f"{Colors.RED}FAILED{Colors.END}"
        print(f"{test_name:.<50} {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All tests passed!{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Some tests failed{Colors.END}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
verify_agent_code.py - Check if the agent is using LangChain properly
"""
import inspect

print("=" * 60)
print("Verifying Agent Implementation")
print("=" * 60)

try:
    from app.agents.forecast_agent import ForecastAgent
    
    agent = ForecastAgent()
    
    # Check methods
    print("\n[1] ForecastAgent Methods:")
    methods = [m for m in dir(agent) if not m.startswith('_') and callable(getattr(agent, m))]
    for method in methods:
        print(f"    ✓ {method}")
    
    # Check for key methods
    required_methods = ['run', '_create_tools', '_create_agent_executor']
    print("\n[2] Required Methods Check:")
    for method in required_methods:
        if hasattr(agent, method):
            print(f"    ✓ {method}")
        else:
            print(f"    ✗ {method} - MISSING!")
    
    # Check run method signature
    print("\n[3] Run Method Signature:")
    sig = inspect.signature(agent.run)
    print(f"    Parameters: {list(sig.parameters.keys())}")
    
    # Check if using LangChain
    print("\n[4] LangChain Integration Check:")
    run_source = inspect.getsource(agent.run)
    
    checks = {
        "create_react_agent": "create_react_agent" in run_source or "_create_agent_executor" in run_source,
        "AgentExecutor": "AgentExecutor" in run_source or "agent_executor" in run_source,
        "Tool objects": "Tool(" in run_source or "_create_tools" in run_source,
        "invoke": "invoke(" in run_source
    }
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check_name}")
    
    # Check return structure
    print("\n[5] Return Structure Check:")
    print("    Checking what run() returns...")
    
    # Look at the return statements
    import re
    returns = re.findall(r'return\s+{[^}]+}', run_source, re.MULTILINE | re.DOTALL)
    if returns:
        print(f"    Found {len(returns)} return statement(s)")
        for i, ret in enumerate(returns[:2], 1):
            print(f"\n    Return {i} (first 200 chars):")
            print(f"    {ret[:200]}...")
    
    print("\n" + "=" * 60)
    print("✓ Agent verification complete")
    print("=" * 60)
    
    # Summary
    if all(checks.values()):
        print("\n✓ Agent is using LangChain properly!")
    else:
        print("\n✗ Agent may not be using LangChain correctly")
        print("   Review the checks above")
    
except Exception as e:
    print(f"\n✗ Error importing agent: {e}")
    import traceback
    traceback.print_exc()
