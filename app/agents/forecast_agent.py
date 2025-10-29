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
from datetime import datetime, timezone
from typing import Dict, Any, List
from jsonschema import validate, ValidationError
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="camelot")


load_dotenv()  # load .env file at runtime

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")

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
                    "timestamp": datetime.now(timezone.utc).isoformat()
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
                    "timestamp": datetime.now(timezone.utc).isoformat()
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
                    "analysis_date": datetime.now(timezone.utc).isoformat(),
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