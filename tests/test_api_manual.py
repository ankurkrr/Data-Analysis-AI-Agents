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
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="camelot")


load_dotenv()  # load .env file at runtime

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")

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