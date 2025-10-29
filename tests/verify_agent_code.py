"""
verify_agent_code.py - Check if the agent is using LangChain properly
"""
import inspect
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="camelot")


load_dotenv()  # load .env file at runtime

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")

print("=" * 60)
print("Verifying Agent Implementation")
print("=" * 60)

try:
    from app.agents.forecast_agent import ForecastAgent
    
    agent = ForecastAgent()
    
    # Check methods
    print("\n[1] ForecastAgent Methods:")
    methods = [m for m in dir(agent) if not m.startswith('_') and callable(getattr(agent, m))]
    for method in methods:
        print(f"    ✓ {method}")
    
    # Check for key methods
    required_methods = ['run', '_create_tools', '_create_agent_executor']
    print("\n[2] Required Methods Check:")
    for method in required_methods:
        if hasattr(agent, method):
            print(f"    ✓ {method}")
        else:
            print(f"    ✗ {method} - MISSING!")
    
    # Check run method signature
    print("\n[3] Run Method Signature:")
    sig = inspect.signature(agent.run)
    print(f"    Parameters: {list(sig.parameters.keys())}")
    
    # Check if using LangChain
    print("\n[4] LangChain Integration Check:")
    run_source = inspect.getsource(agent.run)
    
    checks = {
        "create_react_agent": "create_react_agent" in run_source or "_create_agent_executor" in run_source,
        "AgentExecutor": "AgentExecutor" in run_source or "agent_executor" in run_source,
        "Tool objects": "Tool(" in run_source or "_create_tools" in run_source,
        "invoke": "invoke(" in run_source
    }
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check_name}")
    
    # Check return structure
    print("\n[5] Return Structure Check:")
    print("    Checking what run() returns...")
    
    # Look at the return statements
    import re
    returns = re.findall(r'return\s+{[^}]+}', run_source, re.MULTILINE | re.DOTALL)
    if returns:
        print(f"    Found {len(returns)} return statement(s)")
        for i, ret in enumerate(returns[:2], 1):
            print(f"\n    Return {i} (first 200 chars):")
            print(f"    {ret[:200]}...")
    
    print("\n" + "=" * 60)
    print("✓ Agent verification complete")
    print("=" * 60)
    
    # Summary
    if all(checks.values()):
        print("\n✓ Agent is using LangChain properly!")
    else:
        print("\n✗ Agent may not be using LangChain correctly")
        print("   Review the checks above")
    
except Exception as e:
    print(f"\n✗ Error importing agent: {e}")
    import traceback
    traceback.print_exc()