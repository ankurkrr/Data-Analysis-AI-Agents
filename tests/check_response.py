"""
check_response.py - Analyze the actual API response
"""
import json
import glob
import sys

# Allow passing a specific file path as the first argument. If not provided, find the most recent test output
if len(sys.argv) > 1 and sys.argv[1].strip():
    latest = sys.argv[1].strip()
    print(f"Analyzing provided file: {latest}")
else:
    # Try looking in repository root and in the tests/ folder for test output files
    files = glob.glob("test_output_*.json") + glob.glob("tests/test_output_*.json")
    if not files:
        print("No test output files found. Generate them by running the test scripts that produce test_output_*.json (e.g. tests/test_api_manual.py or tests/fix_chat_claude.py).")
        print("You can also pass a file path to this script: python -m tests.check_response path/to/file.json")
        sys.exit(1)

    latest = max(files)
    print(f"Analyzing most recent file: {latest}")
print("=" * 60)

with open(latest, 'r') as f:
    data = json.load(f)

print("\nTop-level keys:")
for key in data.keys():
    print(f"  - {key}")

print("\nFull structure (first 100 lines):")
output = json.dumps(data, indent=2)
lines = output.split('\n')
for i, line in enumerate(lines[:100]):
    print(line)
    if i == 99 and len(lines) > 100:
        print(f"\n... ({len(lines) - 100} more lines)")

# Check for error
if 'error' in data:
    print("\n" + "=" * 60)
    print("ERROR DETECTED:")
    print("=" * 60)
    print(f"Error type: {data.get('error')}")
    print(f"Message: {data.get('message', 'No message')}")
    
    if 'synthesis' in data and isinstance(data['synthesis'], dict):
        if 'error' in data['synthesis']:
            print(f"\nSynthesis error: {data['synthesis'].get('error')}")
            print(f"Details: {data['synthesis'].get('last_exc')}")
    
    if 'synthesis_attempts' in data:
        print(f"\nSynthesis attempts: {len(data['synthesis_attempts'])}")
        for i, attempt in enumerate(data['synthesis_attempts'], 1):
            print(f"\n  Attempt {i}:")
            print(f"    Raw output (first 200 chars): {attempt.get('raw', '')[:200]}")