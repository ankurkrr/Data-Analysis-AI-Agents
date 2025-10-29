#!/bin/bash

# run_tests.sh - Comprehensive test runner for TCS Forecast Agent

set -e  # Exit on error

echo "=========================================="
echo "TCS Financial Forecasting Agent - Test Suite"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt

echo ""
echo "=========================================="
echo "1. Running Unit Tests"
echo "=========================================="
python -m pytest tests/test_agent_flow.py::TestNumberParsing -v --tb=short
python -m pytest tests/test_agent_flow.py::TestFinancialExtractor -v --tb=short
python -m pytest tests/test_agent_flow.py::TestQualitativeAnalyzer -v --tb=short

echo ""
echo "=========================================="
echo "2. Running Integration Tests"
echo "=========================================="
python -m pytest tests/test_agent_flow.py::TestForecastAgentIntegration -v --tb=short

echo ""
echo "=========================================="
echo "3. Running End-to-End Tests"
echo "=========================================="
python -m pytest tests/test_agent_flow.py::TestEndToEndFlow -v --tb=short

echo ""
echo "=========================================="
echo "4. Running Performance Tests"
echo "=========================================="
python -m pytest tests/test_agent_flow.py::TestPerformance -v --tb=short

echo ""
echo "=========================================="
echo "5. Code Coverage Report"
echo "=========================================="
python -m pytest tests/test_agent_flow.py --cov=app --cov-report=term-missing --cov-report=html

echo ""
echo -e "${GREEN}=========================================="
echo "All Tests Completed!"
echo "==========================================${NC}"
echo ""
echo "Coverage report available at: htmlcov/index.html"
echo ""

# Optional: Run linting
if command -v pylint &> /dev/null; then
    echo "=========================================="
    echo "6. Running Code Quality Checks"
    echo "=========================================="
    pylint app --disable=C0114,C0115,C0116 || true
fi

echo ""
echo -e "${GREEN}✓ Test suite execution complete${NC}"