"""
check_server.py - See what's happening when server starts
"""
import subprocess
import time
import sys

print("=" * 60)
print("Starting server and capturing output...")
print("=" * 60)
print()

# Start server and capture output
process = subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8082"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

print("Server starting... (will show output for 30 seconds)")
print("If you see the error, press CTRL+C")
print("-" * 60)
print()

try:
    start_time = time.time()
    while time.time() - start_time < 30:
        line = process.stdout.readline()
        if line:
            print(line.rstrip())
        
        # Check if process died
        if process.poll() is not None:
            # Process ended
            remaining = process.stdout.read()
            if remaining:
                print(remaining)
            print()
            print("=" * 60)
            print("Server process ended!")
            print("=" * 60)
            sys.exit(1)
        
        # Check if server started successfully
        if "Application startup complete" in line:
            print()
            print("=" * 60)
            print("✓ Server started successfully!")
            print("=" * 60)
            print()
            print("Press CTRL+C to stop")
            # Keep running
            process.wait()
            break
        
        time.sleep(0.1)
    
except KeyboardInterrupt:
    print("\n\nStopping server...")
    process.terminate()
    process.wait()
    print("Server stopped.")