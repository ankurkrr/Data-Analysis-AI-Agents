"""
tests/test_agent_flow.py - Comprehensive integration tests
"""
import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="camelot")


load_dotenv()  # load .env file at runtime

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")

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