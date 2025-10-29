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

# Do not raise at import time if OPENROUTER_API_KEY is missing. The LLM
# wrapper will raise when an LLM call is attempted. This keeps the module
# importable in CI/dev environments where the LLM calls are mocked.

# JSON Schema for final forecast with strict validation
FORECAST_SCHEMA = {
    "type": "object",
    "required": ["metadata", "numeric_trends", "qualitative_summary", "forecast", "risks_and_opportunities", "sources"],
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "request_id": {"type": "string"},
                "analysis_date": {"type": "string", "format": "date-time"},
                "quarters_analyzed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                }
            },
            "required": ["ticker", "request_id", "analysis_date", "quarters_analyzed"],
            "additionalProperties": False
        },
        "numeric_trends": {
            "type": "object",
            "properties": {
                "revenue": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "period": {"type": "string"},
                                    "value": {"type": "number"},
                                    "unit": {"type": "string"}
                                },
                                "required": ["period", "value", "unit"]
                            }
                        },
                        "trend": {"type": "string", "enum": ["increasing", "decreasing", "stable"]},
                        "qoq_change_pct": {"type": "number"}
                    },
                    "required": ["values", "trend"]
                },
                "net_profit": {"type": "object", "properties": {"values": {"type": "array"}}},
                "operating_margin": {"type": "object", "properties": {"values": {"type": "array"}}}
            },
            "required": ["revenue", "net_profit", "operating_margin"]
        },
        "qualitative_summary": {
            "type": "object",
            "properties": {
                "themes": {"type": "array", "items": {"type": "string"}},
                "management_sentiment": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "summary": {"type": "string"}
                    },
                    "required": ["score", "summary"]
                },
                "forward_guidance": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["themes", "management_sentiment", "forward_guidance"]
        },
        "forecast": {
            "type": "object",
            "properties": {
                "outlook_text": {"type": "string", "minLength": 50},
                "numeric_projection": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "low": {"type": "number"},
                        "high": {"type": "number"},
                        "unit": {"type": "string"}
                    },
                    "required": ["metric", "low", "high", "unit"]
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["outlook_text", "numeric_projection", "confidence"]
        },
        "risks_and_opportunities": {
            "type": "object",
            "properties": {
                "risks": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "opportunities": {"type": "array", "items": {"type": "string"}, "minItems": 1}
            },
            "required": ["risks", "opportunities"]
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "document": {"type": "string"},
                    "location": {"type": "string"}
                },
                "required": ["tool", "document", "location"]
            },
            "minItems": 1
        }
    }
}

# ReAct Agent Prompt Template
REACT_PROMPT = """You are a financial analyst generating TCS forecasts using two required tools.

REQUIRED TOOLS (Use in exact order):
1. FinancialDataExtractorTool - Get numeric data
2. QualitativeAnalysisTool - Get management insights

{tools}

WORKFLOW:
1. FIRST use FinancialDataExtractorTool to extract financial metrics
2. THEN use QualitativeAnalysisTool to analyze management commentary
3. Combine both results into final JSON following exact schema below

FORMAT:
You must ALWAYS follow this EXACT format for each step:

Thought: [your reasoning]
Action: [tool name from {tool_names}]
Action Input: {{"action": "extract_all"}} or similar
Observation: [tool response]

Final Answer: [your final JSON output only after using both tools]

Example:
Thought: First, I need to extract financial metrics from the reports.
Action: FinancialDataExtractorTool
Action Input: {{"action": "extract_all"}}
Observation: [tool output]

Thought: Now I need to analyze management commentary.
Action: QualitativeAnalysisTool 
Action Input: {{"action": "analyze_all"}}
Observation: [tool output]

Thought: I have both numeric data and qualitative insights. Now I'll combine them into the final forecast.
Final Answer: [JSON forecast following schema exactly]

REQUIRED JSON OUTPUT STRUCTURE:
{{
    "metadata": {{
        "ticker": "{ticker}",
        "request_id": "{request_id}",
        "analysis_date": "<current-iso-date>",
        "quarters_analyzed": ["Q1", "Q2", "Q3"]
    }},
    "numeric_trends": {{
        "revenue": {{
            "values": [
                {{"period": "Q1-2024", "value": 123, "unit": "INR_Cr"}}
            ],
            "trend": "increasing",
            "qoq_change_pct": 3.5
        }},
        "net_profit": {{ <similar-structure> }},
        "operating_margin": {{ <similar-structure> }}
    }},
    "qualitative_summary": {{
        "themes": ["theme1", "theme2"],
        "management_sentiment": {{
            "score": 0.75,
            "summary": "Brief sentiment summary"
        }},
        "forward_guidance": ["guidance1", "guidance2"]
    }},
    "forecast": {{
        "outlook_text": "Detailed outlook based on both numeric and qualitative analysis",
        "numeric_projection": {{
            "metric": "revenue",
            "low": 13000,
            "high": 13500,
            "unit": "INR_Cr"
        }},
        "confidence": 0.75
    }},
    "risks_and_opportunities": {{
        "risks": ["risk1", "risk2"],
        "opportunities": ["opp1", "opp2"]
    }},
    "sources": [
        {{
            "tool": "FinancialDataExtractorTool",
            "document": "doc-name",
            "location": "page-2"
        }}
    ]
}}

STRICT REQUIREMENTS:
1. Must call both tools in order
2. Final Answer must be EXACT JSON structure above
3. All fields are required
4. Use actual values from tool responses
5. No placeholder text allowed

REQUEST:
Ticker: {ticker}
ID: {request_id}
Quarters: {quarters}
Docs: {documents_summary}

{agent_scratchpad}

{agent_scratchpad}"""


class ForecastAgent:
    def __init__(self):
        # Use factory function to allow a fake LLM in CI/dev via FORCE_FAKE_LLM
        from app.llm.openrouter_llm import get_llm
        self.llm = get_llm()
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

    def _compact_context_for_llm(self, context: Dict, max_chars: int = 2000) -> Dict:
        """Create a small, safe summary of the context suitable for passing to the LLM.

        This avoids sending full document texts or large raw outputs which can blow
        past model context limits.
        """
        compact = {
            "reports": [],
            "transcripts": [],
            "tool_history": []
        }

        docs = context.get("documents", {}) if isinstance(context.get("documents"), dict) else {}
        for r in docs.get("reports", []):
            name = r.get("name")
            # estimate size using available fields but don't include content
            size = 0
            if isinstance(r.get("content"), str):
                size = len(r.get("content"))
            elif isinstance(r.get("text"), str):
                size = len(r.get("text"))
            compact["reports"].append({"name": name, "size_chars": size})

        for t in docs.get("transcripts", []):
            name = t.get("name")
            size = 0
            if isinstance(t.get("content"), str):
                size = len(t.get("content"))
            elif isinstance(t.get("text"), str):
                size = len(t.get("text"))
            compact["transcripts"].append({"name": name, "size_chars": size})

        for h in context.get("tool_history", []):
            compact["tool_history"].append({
                "tool": h.get("tool"),
                "timestamp": h.get("timestamp")
            })

        s = json.dumps(compact, indent=2)
        if len(s) > max_chars:
            # aggressive truncation: keep only counts and first 3 names
            compact_small = {
                "reports_count": len(compact.get("reports", [])),
                "transcripts_count": len(compact.get("transcripts", [])),
                "reports_sample": [r.get("name") for r in compact.get("reports", [])[:3]],
                "transcripts_sample": [t.get("name") for t in compact.get("transcripts", [])[:3]],
                "tool_history": compact.get("tool_history", [])[:5]
            }
            return compact_small

        return compact
    
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
            input_variables=[
                "ticker", "request_id", "quarters", 
                "documents_summary", "agent_scratchpad"
            ],
            partial_variables={
                "tools": "\n".join(f"{tool.name}: {tool.description}" for tool in tools),
                "tool_names": ", ".join(tool.name for tool in tools)
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
            max_iterations=15,  # Increased from 10
            max_execution_time=600,  # Increased to 10 minutes
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            early_stopping_method="force"  # Force stop at max iterations instead of generating invalid output
        )
        
        return agent_executor
    
    def _validate_and_repair_forecast(self, forecast_json: str, context: Dict, max_attempts: int = 3) -> Dict:
        """Validate forecast against schema and attempt repairs if needed"""
        
        attempts = []
        repair_prompt_template = """Fix this forecast JSON to match the required schema exactly.
        
ORIGINAL JSON:
{json_str}

VALIDATION ERROR:
{error}

REQUIRED STRUCTURE:
{
    "metadata": {"ticker": "...", "request_id": "...", "analysis_date": "...", "quarters_analyzed": ["..."]},
    "numeric_trends": {
        "revenue": {"values": [{"period": "...", "value": 123, "unit": "..."}], "trend": "...", "qoq_change_pct": 0.0},
        "net_profit": {"values": [...], "trend": "...", "qoq_change_pct": 0.0},
        "operating_margin": {"values": [...], "trend": "..."}
    },
    "qualitative_summary": {
        "themes": ["..."],
        "management_sentiment": {"score": 0.0, "summary": "..."},
        "forward_guidance": ["..."]
    },
    "forecast": {
        "outlook_text": "...",
        "numeric_projection": {"metric": "...", "low": 0.0, "high": 0.0, "unit": "..."},
        "confidence": 0.0
    },
    "risks_and_opportunities": {
        "risks": ["..."],
        "opportunities": ["..."]
    },
    "sources": [{"tool": "...", "document": "...", "location": "..."}]
}

CONTEXT:
{context}

Return ONLY the fixed JSON with no additional text."""
        
        for attempt in range(1, max_attempts + 1):
            attempts.append({"attempt": attempt, "raw": forecast_json[:500]})
            
            # Try to parse JSON - first clean up any markdown code block markers
            try:
                # Remove markdown code block markers and extract just the JSON
                clean_json = forecast_json
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[-1]
                if "```" in clean_json:
                    clean_json = clean_json.split("```")[0]
                clean_json = clean_json.strip()
                
                parsed = json.loads(clean_json)
            except json.JSONDecodeError as e:
                if attempt == max_attempts:
                    return {
                        "error": "json_parse_failed",
                        "details": str(e),
                        "attempts": attempts
                    }
                
                # Ask LLM to fix JSON
                # Truncate very large agent outputs to avoid exceeding model context
                truncated_forecast = forecast_json
                if isinstance(forecast_json, str) and len(forecast_json) > 15000:
                    truncated_forecast = forecast_json[-15000:]
                    truncated_forecast = "...(truncated)\n" + truncated_forecast

                repair_prompt = f"""The following text should be valid JSON but has parsing errors:

{truncated_forecast}

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
                compact_ctx = self._compact_context_for_llm(context, max_chars=1500)
                repair_prompt = repair_prompt_template.format(
                    json_str=json.dumps(parsed, indent=2),
                    error=str(ve),
                    context=json.dumps(compact_ctx, indent=2)
                )
                
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
        from app.utils.document_chunker import chunk_documents
        
        # Reset tool call history
        self._tool_call_history = []
        
        # Step 1: Fetch documents
        fetcher = DocumentFetcher()
        documents = fetcher.fetch_quarterly_documents(ticker, quarters, sources)
        
        # Step 2: Split into chunks that fit token limits
        document_chunks = chunk_documents(documents, max_chunk_tokens=4000)
        
        if not documents.get("reports") and not documents.get("transcripts"):
            return {
                "error": "no_documents_found",
                "message": "Could not fetch any financial documents or transcripts",
                "ticker": ticker,
                "request_id": request_id
            }
            
        if not document_chunks:
            return {
                "error": "chunking_failed",
                "message": "Failed to split documents into processable chunks",
                "ticker": ticker,
                "request_id": request_id
            }
        
        # Step 3: Process each chunk and merge results
        all_results = []
        
        # Create and run agent on first chunk (main context)
        try:
            agent_executor = self._create_agent_executor(ticker, request_id, quarters, document_chunks[0])
            
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
            
            # Step 3: If the qualitative tool was not called but transcripts exist,
            # run the qualitative analyzer as a resilient fallback and ask the LLM
            # to synthesize a proper final JSON using both outputs.
            tools_used_now = set(t["tool"] for t in self._tool_call_history)

            if "QualitativeAnalysisTool" not in tools_used_now and documents.get("transcripts"):
                try:
                    # Run qualitative analysis directly as a fallback
                    qualitative_result = self.qualitative_tool_instance.analyze(documents.get("transcripts"))
                    self._tool_call_history.append({
                        "tool": "QualitativeAnalysisTool",
                        "input": {"action": "analyze_all", "fallback": True},
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                    # Ask LLM to synthesize a final JSON combining previous agent output
                    # and the qualitative tool output. Keep the prompt focused and strict.
                    # Build a compact qualitative summary to avoid huge prompts
                    qual_compact = {
                        "themes": qualitative_result.get("themes") if isinstance(qualitative_result, dict) else None,
                        "management_sentiment": qualitative_result.get("management_sentiment") if isinstance(qualitative_result, dict) else qualitative_result.get("sentiment") if isinstance(qualitative_result, dict) else None,
                        "forward_guidance": qualitative_result.get("forward_guidance") if isinstance(qualitative_result, dict) else None,
                        "risks": qualitative_result.get("risks") if isinstance(qualitative_result, dict) else None,
                        "transcripts_analyzed": qualitative_result.get("transcripts_analyzed") if isinstance(qualitative_result, dict) else None
                    }

                    # Truncate agent output to a safe length to avoid token overflows
                    safe_agent_output = (agent_output[-8000:] if isinstance(agent_output, str) and len(agent_output) > 8000 else agent_output)

                    synth_prompt = (
                        "The agent failed to call the QualitativeAnalysisTool during its run. "
                        "We have the agent's last produced output (truncated) and a compact qualitative summary. "
                        "Using BOTH sources, produce a single, complete JSON forecast that strictly follows the required schema. "
                        "Return ONLY the JSON object, no explanation.\n\n"
                        f"AGENT_OUTPUT_TRUNCATED:\n{safe_agent_output}\n\nQUALITATIVE_SUMMARY:\n{json.dumps(qual_compact, indent=2)}\n\n"
                    )

                    # Call the LLM to synthesize the final JSON
                    try:
                        synthesized = self.llm._call(synth_prompt)
                        # If LLM produced something, replace agent_output so validation will run on it
                        if synthesized and isinstance(synthesized, str):
                            agent_output = synthesized
                    except Exception:
                        # If synthesis fails, continue and let validation attempt on original output
                        pass

                except Exception as e:
                    # Best-effort fallback - if qualitative tool fails, continue and surface original error
                    self._tool_call_history.append({
                        "tool": "QualitativeAnalysisTool",
                        "input": {"action": "analyze_all", "fallback": True},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": str(e)
                    })

            # Step 3: Validate and repair the forecast
            validation_result = self._validate_and_repair_forecast(
                agent_output,
                context={"documents": documents, "tool_history": self._tool_call_history}
            )
            
            # Step 4: Build final response
            # Determine whether agent output was produced by a fake LLM (resilient fallback or forced fake)
            llm_fake = False
            llm_mode = "real"
            try:
                parsed_agent = json.loads(agent_output)
                if isinstance(parsed_agent, dict) and parsed_agent.get("__fake_output"):
                    llm_fake = True
            except Exception:
                # Not valid JSON; do a string check for marker inserted by fake LLM
                if "__fake_output" in (agent_output or ""):
                    llm_fake = True

            # Also check intermediate steps/observations (some agents/tool chains surface the final
            # JSON inside an observation or intermediate output rather than the top-level `output`).
            # This makes detection robust when the ReAct parsing wraps the JSON in Thought/Action text.
            if not llm_fake and intermediate_steps:
                try:
                    for step in intermediate_steps:
                        # Each step is typically a tuple like (AgentAction, observation) or (AgentFinish, "...")
                        if not step:
                            continue
                        # observation may be at index 1 for AgentAction tuples, or be the sole element for AgentFinish
                        obs_text = None
                        if isinstance(step, (list, tuple)) and len(step) >= 2:
                            obs = step[1]
                            obs_text = obs if isinstance(obs, str) else json.dumps(obs)
                        else:
                            # Fallback: stringify the whole step
                            obs_text = json.dumps(step)

                        if obs_text and "__fake_output" in obs_text:
                            llm_fake = True
                            break
                except Exception:
                    # Best-effort only; detection defaults to previous checks
                    pass

            # Respect explicit FORCE_FAKE_LLM env var: if set, consider mode 'fake'
            if os.getenv("FORCE_FAKE_LLM"):
                llm_mode = "fake"
                # If the environment forces a fake LLM for CI/tests, treat the output as synthetic
                # even if the detection heuristics above didn't find the fake marker.
                llm_fake = True
            elif llm_fake:
                # If output contains fake marker but FORCE_FAKE_LLM not set, assume resilient fallback
                llm_mode = "resilient-fallback"
            else:
                llm_mode = "real"

            # Emit a warning log for monitoring when a fake/fallback LLM produced results
            if llm_mode != "real":
                import logging
                logging.warning(f"LLM produced synthetic output for request_id={request_id}; mode={llm_mode}")

            # Validate that both tools were called
            tools_used = set(t["tool"] for t in self._tool_call_history)
            if not ("FinancialDataExtractorTool" in tools_used and "QualitativeAnalysisTool" in tools_used):
                return {
                    "error": "incomplete_analysis",
                    "message": "Agent must use both financial extraction and qualitative analysis tools",
                    "tools_called": list(tools_used),
                    "tool_history": self._tool_call_history
                }

            final_response = {
                "metadata": {
                    "ticker": ticker,
                    "request_id": request_id,
                    "analysis_date": datetime.now(timezone.utc).isoformat(),
                    "quarters_analyzed": [r.get("name") for r in documents.get("reports", [])],
                    # Indicates whether the returned forecast was produced by a fake/fallback LLM
                    "llm_fake": llm_fake,
                    "llm_mode": llm_mode
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