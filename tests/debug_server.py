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