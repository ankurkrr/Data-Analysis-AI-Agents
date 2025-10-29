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
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="camelot")


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