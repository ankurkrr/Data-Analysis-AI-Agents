#!/usr/bin/env python3
"""
Quick diagnostic script to identify issues
Run this BEFORE starting the server to check all dependencies
"""
import sys
import os

def check_env_vars():
    """Check required environment variables"""
    print("\n" + "="*60)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("="*60)
    
    required = {
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "MYSQL_HOST": os.getenv("MYSQL_HOST"),
        "MYSQL_USER": os.getenv("MYSQL_USER"),
        "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD"),
        "MYSQL_DB": os.getenv("MYSQL_DB")
    }
    
    all_good = True
    for key, value in required.items():
        if value:
            print(f"✓ {key}: {'*' * 10} (set)")
        else:
            print(f"✗ {key}: NOT SET")
            all_good = False
    
    return all_good

def check_imports():
    """Check critical imports"""
    print("\n" + "="*60)
    print("CHECKING PYTHON IMPORTS")
    print("="*60)
    
    imports = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn server",
        "langchain": "LangChain",
        "mysql.connector": "MySQL connector",
        "pdfplumber": "PDF text extraction",
        "requests": "HTTP requests",
        "beautifulsoup4": "Web scraping",
        "sentence_transformers": "Embeddings (optional)",
        "faiss": "Vector search (optional)",
        "tiktoken": "Token counting"
    }
    
    critical_failed = []
    optional_failed = []
    
    for module, desc in imports.items():
        try:
            __import__(module.replace("-", "_"))
            print(f"✓ {module:25s} - {desc}")
        except ImportError:
            is_optional = "(optional)" in desc
            if is_optional:
                print(f"⚠ {module:25s} - {desc} - NOT INSTALLED")
                optional_failed.append(module)
            else:
                print(f"✗ {module:25s} - {desc} - MISSING!")
                critical_failed.append(module)
    
    return len(critical_failed) == 0, critical_failed, optional_failed

def check_database():
    """Check MySQL connection"""
    print("\n" + "="*60)
    print("CHECKING DATABASE CONNECTION")
    print("="*60)
    
    try:
        import mysql.connector
        from dotenv import load_dotenv
        load_dotenv()
        
        config = {
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_PORT", 3306)),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", "")
        }
        
        print(f"Attempting connection to {config['user']}@{config['host']}:{config['port']}...")
        
        conn = mysql.connector.connect(**config, connect_timeout=5)
        print(f"✓ MySQL connection successful")
        
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        dbs = [db[0] for db in cursor.fetchall()]
        
        db_name = os.getenv("MYSQL_DB", "tcs_forecast")
        if db_name in dbs:
            print(f"✓ Database '{db_name}' exists")
        else:
            print(f"⚠ Database '{db_name}' does not exist (will be created)")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ MySQL connection failed: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"  1. Is MySQL running? Try: sudo systemctl status mysql")
        print(f"  2. Check credentials in .env file")
        print(f"  3. Test with: mysql -u {os.getenv('MYSQL_USER')} -p")
        return False

def check_llm():
    """Check LLM configuration"""
    print("\n" + "="*60)
    print("CHECKING LLM CONFIGURATION")
    print("="*60)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("✗ OPENROUTER_API_KEY not set")
        print("\nGet a free key from: https://openrouter.ai/")
        return False
    
    print(f"✓ API key set: {api_key[:20]}...")
    
    # Test API call
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": os.getenv("LLM_MODEL", "qwen/qwen-2.5-72b-instruct:free"),
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        print("Testing API connection...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✓ LLM API connection successful")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ API test failed: {str(e)}")
        return False

def main():
    print("\n" + "="*60)
    print("TCS FORECAST AGENT - DIAGNOSTIC TOOL")
    print("="*60)
    
    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Loaded .env file")
    except:
        print("✗ Failed to load .env file")
    
    # Run checks
    env_ok = check_env_vars()
    imports_ok, critical_missing, optional_missing = check_imports()
    db_ok = check_database()
    llm_ok = check_llm()
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    if env_ok:
        print("✓ Environment variables: OK")
    else:
        print("✗ Environment variables: MISSING")
    
    if imports_ok:
        print("✓ Python dependencies: OK")
    else:
        print(f"✗ Missing critical packages: {', '.join(critical_missing)}")
        print(f"\n  Install with: pip install {' '.join(critical_missing)}")
    
    if optional_missing:
        print(f"⚠ Optional packages missing: {', '.join(optional_missing)}")
        print(f"  Install with: pip install {' '.join(optional_missing)}")
    
    if db_ok:
        print("✓ Database connection: OK")
    else:
        print("✗ Database connection: FAILED")
    
    if llm_ok:
        print("✓ LLM API: OK")
    else:
        print("✗ LLM API: FAILED")
    
    print("\n" + "="*60)
    
    if all([env_ok, imports_ok, db_ok, llm_ok]):
        print("✓ ALL CHECKS PASSED - Ready to start server!")
        print("\nRun: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Fix issues above before starting")
        return 1

if __name__ == "__main__":
    sys.exit(main())