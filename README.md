# TCS Financial Forecasting Agent

An AI-powered financial analysis agent that generates business outlook forecasts for Tata Consultancy Services (TCS) using LangChain, FastAPI, and advanced document processing techniques.

## 📋 Table of Contents

* [Overview](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#overview)
* [Architecture](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#architecture)
* [AI Stack &amp; Approach](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#ai-stack--approach)
* [Features](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#features)
* [Prerequisites](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#prerequisites)
* [Installation](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#installation)
* [Configuration](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#configuration)
* [Running the Application](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#running-the-application)
* [API Usage](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#api-usage)
* [Testing](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#testing)
* [Project Structure](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#project-structure)
* [Agent &amp; Tool Design](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#agent--tool-design)
* [Limitations &amp; Tradeoffs](https://claude.ai/chat/cc70cfb3-c71a-439a-9b91-e5e3b1ca78d7#limitations--tradeoffs)

---

## 🎯 Overview

This project implements an **agentic AI system** that autonomously:

1. **Fetches** quarterly financial reports and earnings call transcripts from web sources
2. **Extracts** numerical financial metrics using multi-method PDF processing
3. **Analyzes** qualitative insights using RAG-based semantic search
4. **Synthesizes** a structured JSON forecast with confidence scores and source citations
5. **Logs** all requests and results to MySQL for audit trails

The agent uses **LangChain's ReAct framework** to reason step-by-step, deciding which tools to call and when, rather than following a rigid pipeline.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   /health    │  │ /api/forecast│  │ /api/status  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
└─────────┼──────────────────┼──────────────────────────────┘
          │                  │
          ▼                  ▼
    ┌─────────────────────────────────┐
    │   ForecastAgent (LangChain)     │
    │   ┌─────────────────────────┐   │
    │   │  ReAct Agent Executor   │   │
    │   │  (Thought-Action Loop)  │   │
    │   └───────┬─────────────────┘   │
    │           │                      │
    │     ┌─────┴──────┐              │
    │     ▼            ▼              │
    │  ┌─────┐    ┌─────────┐        │
    │  │Tool1│    │  Tool2  │        │
    │  └─────┘    └─────────┘        │
    └─────────────────────────────────┘
              │            │
    ┌─────────▼────┐  ┌───▼──────────┐
    │ Financial    │  │ Qualitative  │
    │ Extractor    │  │ Analyzer     │
    │              │  │              │
    │ • Camelot    │  │ • FAISS      │
    │ • pdfplumber │  │ • Embeddings │
    │ • OCR        │  │ • RAG Search │
    └──────────────┘  └──────────────┘
              │            │
              ▼            ▼
         ┌────────────────────┐
         │  OpenRouter LLM    │
         │  (deepseek-chat)   │
         └────────────────────┘
                  │
                  ▼
         ┌────────────────────┐
         │   MySQL Database   │
         │  • requests table  │
         │  • results table   │
         └────────────────────┘
```

### Key Components:

* **Agent Layer** : LangChain ReAct agent with autonomous reasoning
* **Tool Layer** : Specialized tools for financial extraction and qualitative analysis
* **LLM Layer** : OpenRouter API (free deepseek model)
* **Storage Layer** : MySQL for request/response logging
* **Document Layer** : Web scraping + multi-method PDF processing

---

## 🤖 AI Stack & Approach

### LLM Provider

* **OpenRouter API** with `deepseek/deepseek-chat-v3.1:free`
* Reasoning: Free tier, good performance for financial analysis tasks
* Alternative: Can easily swap to OpenAI GPT-4, Anthropic Claude, etc.

### Agent Framework

* **LangChain ReAct Agent** - Enables autonomous reasoning:
  ```
  Thought → Action → Observation → Thought → ...
  ```
* Allows agent to decide tool calling sequence dynamically
* Handles errors and retries intelligently

### Document Processing Stack

**Financial Data Extraction** (3-tier fallback):

1. **Camelot** (primary): Table extraction from structured PDFs
   * Best accuracy for formatted financial statements
   * Handles both lattice and stream modes
2. **pdfplumber** (fallback): Text-based extraction
   * Works when tables aren't structured
   * Pattern matching for metric labels
3. **OCR via pytesseract** (last resort): Image-based extraction
   * For scanned documents or poor-quality PDFs
   * Highest latency but most robust

**Qualitative Analysis** (RAG-based):

* **sentence-transformers** (`all-MiniLM-L6-v2`): Generates embeddings
* **FAISS** : Vector similarity search (CPU-optimized)
* **Chunking strategy** : 300-word chunks with overlap
* **Retrieval** : Top-k semantic search for themes, sentiment, guidance

### Output Synthesis

* **JSON Schema Validation** : Strict schema enforcement
* **Retry with Repair** : LLM self-corrects invalid outputs (up to 3 attempts)
* **Source Attribution** : Every claim linked to specific documents/chunks

### Guardrails

1. **Schema validation** ensures predictable output format
2. **Confidence scoring** for each extracted metric (0-1 scale)
3. **Source citation** prevents hallucination
4. **Extraction method logging** for transparency
5. **Timeout limits** (5 min per request)

---

## ✨ Features

✅  **Autonomous Agent** : ReAct-based reasoning, not hardcoded pipelines

✅  **Multi-Method PDF Processing** : Camelot → pdfplumber → OCR fallback

✅  **RAG-Powered Analysis** : FAISS vector search across transcripts

✅  **Structured JSON Output** : Schema-validated forecasts

✅  **Web Scraping** : Auto-fetches documents from Screener.in

✅  **MySQL Logging** : Full audit trail of requests and results

✅  **Confidence Scoring** : Transparency in metric extraction reliability

✅  **Source Attribution** : Every claim cites its origin

---

## 📦 Prerequisites

* **Python 3.10+**
* **MySQL 8.0+**
* **System dependencies** :

```bash
  # Ubuntu/Debiansudo apt-get install -y tesseract-ocr poppler-utils ghostscript# macOSbrew install tesseract poppler ghostscript
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd tcs-forecast-agent
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup MySQL Database

```bash
# Login to MySQL
mysql -u root -p

# Create database
CREATE DATABASE tcs_forecast;

# Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON tcs_forecast.* TO 'your_user'@'localhost';
FLUSH PRIVILEGES;
```

The application will auto-create tables on first run.

---

## ⚙️ Configuration

### 1. Create `.env` File

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env`:

```bash
# OpenRouter API (get free key from https://openrouter.ai)
OPENROUTER_API_KEY=your_api_key_here

# MySQL Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=tcs_forecast

# Optional: Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Getting OpenRouter API Key:

1. Visit https://openrouter.ai/
2. Sign up (free)
3. Go to "Keys" section
4. Create new key
5. Copy to `.env`

---

## 🏃 Running the Application

### Start the Server

```bash
# Development mode (with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server will be available at: `http://localhost:8000`

### Verify It's Running

```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

---

## 📡 API Usage

### 1. Generate Forecast

 **Endpoint** : `POST /api/forecast/tcs`

 **Request Body** :

```json
{
  "quarters": 3,
  "sources": ["screener", "company-ir"],
  "include_market": false
}
```

 **Parameters** :

* `quarters` (int): Number of past quarters to analyze (1-4)
* `sources` (list): Document sources to fetch from
* `include_market` (bool): Include live market data (optional feature)

 **Example with curl** :

```bash
curl -X POST "http://localhost:8000/api/forecast/tcs" \
  -H "Content-Type: application/json" \
  -d '{
    "quarters": 3,
    "sources": ["screener"],
    "include_market": false
  }'
```

 **Example with Python** :

```python
import requests

response = requests.post(
    "http://localhost:8000/api/forecast/tcs",
    json={
        "quarters": 3,
        "sources": ["screener"],
        "include_market": False
    }
)

forecast = response.json()
print(forecast["metadata"])
print(forecast["forecast"])
```

 **Response Structure** :

```json
{
  "metadata": {
    "ticker": "TCS",
    "request_id": "uuid-here",
    "analysis_date": "2024-10-28T...",
    "quarters_analyzed": ["Q1-FY24", "Q2-FY24", "Q3-FY24"]
  },
  "agent_execution": {
    "tool_calls": [...],
    "iterations": 4,
    "intermediate_steps_count": 8
  },
  "forecast": {
    "metadata": {...},
    "numeric_trends": {
      "total_revenue": {
        "values": [...],
        "trend": "increasing",
        "qoq_change_pct": 3.5
      }
    },
    "qualitative_summary": {
      "themes": ["Digital transformation", "AI adoption"],
      "management_sentiment": {
        "score": 0.65,
        "summary": "Cautiously optimistic"
      }
    },
    "forecast": {
      "outlook_text": "Based on...",
      "numeric_projection": {...},
      "confidence": 0.72
    },
    "risks_and_opportunities": {...},
    "sources": [...]
  }
}
```

### 2. Check Request Status

 **Endpoint** : `GET /api/status/{request_id}`

```bash
curl http://localhost:8000/api/status/your-request-uuid
```

---

## 🧪 Testing

### Run All Tests

```bash
# Make test script executable
chmod +x run_tests.sh

# Run complete test suite
./run_tests.sh
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/test_agent_flow.py::TestNumberParsing -v
pytest tests/test_agent_flow.py::TestFinancialExtractor -v

# Integration tests
pytest tests/test_agent_flow.py::TestForecastAgentIntegration -v

# End-to-end tests
pytest tests/test_agent_flow.py::TestEndToEndFlow -v

# Performance tests
pytest tests/test_agent_flow.py::TestPerformance -v
```

### Run Manual API Tests

```bash
# Start server first, then:
python test_api_manual.py
```

This will:

* Test all endpoints
* Generate sample requests
* Save responses to JSON files
* Verify database logging
* Print detailed results

### Coverage Report

```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

---

## 📁 Project Structure

```
tcs-forecast-agent/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   └── endpoints.py           # API route handlers
│   ├── agents/
│   │   └── forecast_agent.py      # LangChain ReAct agent (REFACTORED)
│   ├── tools/
│   │   ├── financial_extractor_tool.py   # Financial data extraction
│   │   └── qualitative_analysis_tool.py  # RAG-based transcript analysis
│   ├── services/
│   │   └── document_fetcher.py    # Web scraping for documents
│   ├── db/
│   │   └── mysql_client.py        # Database operations
│   ├── llm/
│   │   └── openrouter_llm.py      # LLM wrapper for OpenRouter
│   └── utils/
│       └── number_parsing.py      # INR number parsing utilities
├── tests/
│   ├── test_agent_flow.py         # Comprehensive test suite
│   └── data/                      # Test fixtures
│       ├── sample_report_q3.pdf
│       └── test_transcript_1.txt
├── data/
│   └── downloads/                 # Downloaded documents (auto-created)
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── .env                           # Your credentials (gitignored)
├── README.md                      # This file
├── run_tests.sh                   # Test runner script
└── test_api_manual.py             # Manual API testing script
```

---

## 🛠️ Agent & Tool Design

### Master Agent Prompt

The agent uses a **ReAct prompt** that enables step-by-step reasoning:

```
You are a senior financial analyst AI agent with access to specialized tools.

Your goal: Generate a comprehensive business outlook forecast for TCS.

TOOLS:
1. FinancialDataExtractorTool - Extract metrics from quarterly reports
2. QualitativeAnalysisTool - Analyze earnings transcripts

PROCESS:
Thought: [Reason about what to do next]
Action: [Tool name]
Action Input: [JSON parameters]
Observation: [Tool output]
... (repeat until sufficient information)

Thought: I now have all the information needed
Final Answer: [Complete JSON forecast]
```

### Tool 1: FinancialDataExtractorTool

 **Purpose** : Extract numerical financial metrics with high confidence

**Methods** (3-tier fallback):

1. **Camelot** : Parse PDF tables directly
2. **pdfplumber** : Text-based pattern matching
3. **OCR** : Image-to-text for scanned docs

 **Output** :

```json
{
  "tool": "FinancialDataExtractorTool",
  "results": [{
    "doc_meta": {"name": "Q3-Results", "local_path": "..."},
    "metrics": {
      "total_revenue": {
        "value": 60583,
        "unit": "INR_Cr",
        "confidence": 0.85,
        "source": {"method": "camelot", "page": 2}
      }
    }
  }]
}
```

 **Design Rationale** :

* Multiple methods ensure robustness across PDF formats
* Confidence scoring enables downstream filtering
* Source tracking prevents hallucination

### Tool 2: QualitativeAnalysisTool

 **Purpose** : Extract themes, sentiment, and forward guidance from transcripts

 **Architecture** :

* **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
* **Vector Store** : FAISS (CPU-optimized, in-memory)
* **Chunking** : 300 words per chunk
* **Retrieval** : Top-5 semantic search per query

 **Queries** :

* Demand trends: "demand, growth, digital transformation"
* Attrition: "attrition, employee turnover, resignations"
* Guidance: "guidance, outlook, expect, forecast"

 **Output** :

```json
{
  "tool": "QualitativeAnalysisTool",
  "themes": [{
    "theme": "demand",
    "count": 5,
    "examples": [{"chunk_id": "...", "text": "..."}]
  }],
  "management_sentiment": {
    "score": 0.65,
    "summary": "Cautiously optimistic"
  },
  "forward_guidance": [...]
}
```

 **Design Rationale** :

* RAG prevents hallucination (answers grounded in actual text)
* Semantic search outperforms keyword matching
* Chunk-level attribution enables verification

---

## ⚖️ Limitations & Tradeoffs

### Current Limitations

1. **Web Scraping Brittleness**
   * Screener.in HTML structure can change
   * Rate limiting may block aggressive requests
   * **Mitigation** : Fallback to local test files, respect robots.txt
2. **PDF Processing Accuracy**
   * Some PDFs have complex layouts (multi-column, nested tables)
   * OCR accuracy depends on image quality
   * **Mitigation** : 3-tier fallback, confidence scoring
3. **LLM Output Consistency**
   * Free-tier models occasionally produce malformed JSON
   * **Mitigation** : Schema validation + 3-attempt repair loop
4. **Embedding Model Limitations**
   * `all-MiniLM-L6-v2` is fast but less accurate than larger models
   * **Tradeoff** : Speed vs accuracy (can upgrade to `mpnet-base-v2`)
5. **No Real-Time Market Data**
   * `include_market` parameter is a stub (not implemented)
   * **Future work** : Integrate with Yahoo Finance API

### Performance Tradeoffs

| Component     | Speed          | Accuracy      | Resource Usage |
| ------------- | -------------- | ------------- | -------------- |
| Camelot       | Fast (2-5s)    | High (0.85)   | Medium         |
| pdfplumber    | Fast (1-3s)    | Medium (0.65) | Low            |
| OCR           | Slow (10-30s)  | Low (0.45)    | High           |
| FAISS         | Fast (<1s)     | High          | Low (CPU)      |
| LLM Synthesis | Medium (5-15s) | Variable      | Medium         |

 **Total Request Time** : 30-120 seconds (depending on document quality)

### Design Decisions

1. **Why LangChain ReAct over custom pipeline?**
   * **Pro** : Flexible, handles errors, extensible
   * **Con** : Adds latency (~10% overhead)
   * **Decision** : Flexibility > speed for this use case
2. **Why OpenRouter free tier?**
   * **Pro** : Zero cost, decent quality
   * **Con** : Rate limits, slower than GPT-4
   * **Decision** : Cost-effective for POC, easily upgradeable
3. **Why FAISS (CPU) over Pinecone/Weaviate?**
   * **Pro** : No external dependencies, fast enough
   * **Con** : In-memory only, no persistence
   * **Decision** : Simplicity > scalability for demo
4. **Why MySQL over MongoDB?**
   * **Pro** : SQL is standard, easier to query
   * **Con** : JSON storage less elegant than NoSQL
   * **Decision** : Assignment requirement (MySQL 8.0)

---

## 🔮 Future Enhancements

* [ ] Implement `MarketDataTool` for live stock prices
* [ ] Add caching layer (Redis) for document processing
* [ ] Upgrade to production LLM (GPT-4, Claude Opus)
* [ ] Add async processing (Celery) for long-running requests
* [ ] Implement request queuing and status polling
* [ ] Add monitoring/observability (Prometheus, Grafana)
* [ ] Support for other companies (generic ticker)
* [ ] PDF caching to avoid re-downloading
* [ ] Streaming responses for real-time feedback

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

This is an assignment project, but feedback is welcome! Open an issue for bugs or suggestions.

---

## 📧 Contact

For questions about this implementation:

* Open an issue in the repository
* Check the test outputs for debugging hints
* Review agent execution logs in MySQL database

---

**Built with ❤️ using LangChain, FastAPI, and OpenRouter**
