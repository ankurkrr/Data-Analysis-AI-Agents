import os
import sys
import pytest
import logging
import requests
from bs4 import BeautifulSoup
from app.services import document_fetcher

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.mark.debug  # Add marker for easier filtering
def test_fetch_tcs_ir_reports_download():
    """Test downloading TCS IR reports with detailed logging."""
    logger.info("Starting TCS IR download test")
    
    # Ensure download directory exists in project root
    downloads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    logger.info(f"Download directory ready: {downloads_dir}")
    
    try:
            # Test with specific year and quarter that exists
            url = document_fetcher._get_tcs_ir_url("2023-24", "Q1")
            logger.info(f"Testing URL: {url}")
            
            # Use expanded headers that mimic a real browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Dest": "document",
                "DNT": "1",
            }
            
            try:
                logger.info("Making HTTP request...")
                resp = requests.get(url, headers=headers, timeout=30)
                logger.info(f"Response status: {resp.status_code}")
                logger.info(f"Response headers: {dict(resp.headers)}")
                
                # Log first part of response for debugging
                content = resp.text
                logger.info(f"Response length: {len(content)} chars")
                logger.info(f"First 1000 chars of response: {content[:1000]}...")
                
                # Parse and analyze HTML
                soup = BeautifulSoup(content, "html.parser")
                
                # Log all links found
                all_links = soup.find_all("a", href=True)
                logger.info(f"Total links found: {len(all_links)}")
                logger.info("First 10 links found:")
                for link in all_links[:10]:
                    logger.info(f"Link text: '{link.get_text().strip()}', href: '{link['href']}'")
                
                # Look for PDF links more broadly
                pdf_links = []
                for link in all_links:
                    href = link["href"]
                    text = link.get_text().strip()
                    if (".pdf" in href.lower() or 
                        "financial" in text.lower() or 
                        "result" in text.lower() or
                        "statement" in text.lower()):
                        pdf_links.append(link)
                
                logger.info(f"Found {len(pdf_links)} potential PDF/financial links:")
                for link in pdf_links:
                    logger.info(f"- Text: '{link.get_text().strip()}', href: '{link['href']}'")
                    
            except Exception as e:
                logger.error(f"Error fetching URL: {str(e)}", exc_info=True)
            
            logger.info("Fetching reports for 2023-24 Q1...")
            reports = document_fetcher.fetch_tcs_ir_reports(
                year="2023-24",
                quarters=["Q1"],
                max_reports=1
            )
            logger.info(f"Reports fetched: {reports}")

            # Validate results
            assert isinstance(reports, list), "Expected reports to be a list"
            assert len(reports) > 0, "Expected at least one report"

            for r in reports:
                logger.info(f"Validating report: {r['name']}")
                assert r["local_path"].endswith(".pdf"), f"Expected PDF file, got: {r['local_path']}"
                assert os.path.exists(r["local_path"]), f"File should exist: {r['local_path']}"
                size = os.path.getsize(r["local_path"])
                logger.info(f"File size: {size} bytes")
                assert size > 1000, f"File too small ({size} bytes): {r['local_path']}"
                assert r["year"] == "2023-24", f"Wrong year: {r['year']}"
                assert r["quarter"] == "Q1", f"Wrong quarter: {r['quarter']}"
                assert r["type"] in ["Consolidated", "Standalone"], f"Wrong type: {r['type']}"
                logger.info(f"Report validation successful: {r['name']}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

def test_fetch_quarterly_documents_download():
    """Test downloading documents from Screener.in"""
    try:
        # Use TCS ticker, 1 quarter for speed
        logger.info("Testing Screener.in document fetch...")
        result = document_fetcher.fetch_quarterly_documents("TCS", quarters=1)
        
        assert "reports" in result, "Missing 'reports' in result"
        assert isinstance(result["reports"], list), "Reports should be a list"
        assert len(result["reports"]) > 0, "Should have at least one report"
        
        for r in result["reports"]:
            logger.info(f"Validating report: {r['name']}")
            assert r["local_path"].endswith(".pdf"), f"File should be PDF, got: {r['local_path']}"
            assert os.path.exists(r["local_path"]), f"File should exist: {r['local_path']}"
            size = os.path.getsize(r["local_path"])
            logger.info(f"File size: {size} bytes")
            assert size > 1000, f"File too small ({size} bytes): {r['local_path']}"
            logger.info(f"Report validation successful: {r['name']}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--capture=no"])