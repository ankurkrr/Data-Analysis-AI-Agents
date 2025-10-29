"""
fix_imports.py - Automatic dependency fix script
Run this to fix import errors and ensure compatible versions
"""
import subprocess
import sys
import os

def run_command(cmd):
    """Run a shell command and print output"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 13:
        print("⚠️  Warning: Python 3.13 detected. Some packages may have compatibility issues.")
        print("💡 Recommended: Use Python 3.10 or 3.11 for best compatibility")
        return False
    elif version.major == 3 and version.minor >= 10:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.10+ required")
        return False

def fix_langchain_imports():
    """Fix LangChain import issues"""
    print("\n📦 Fixing LangChain dependencies...")
    
    commands = [
        # Uninstall old versions
        "pip uninstall -y langchain langchain-community langchain-core",
        
        # Install compatible versions
        "pip install langchain==0.1.20",
        "pip install langchain-community==0.0.38", 
        "pip install langchain-core==0.1.52",
        "pip install huggingface-hub==0.20.3",
        
        # Ensure other dependencies
        "pip install sentence-transformers==2.2.2",
        "pip install faiss-cpu==1.7.4",
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"⚠️  Command failed: {cmd}")
            return False
    
    return True

def verify_imports():
    """Test if imports work"""
    print("\n🔍 Verifying imports...")
    
    tests = [
        ("LangChain Tool", "from langchain.tools import Tool"),
        ("LangChain Agents", "from langchain.agents import AgentExecutor, create_react_agent"),
        ("HuggingFace Hub", "from langchain_community.llms import HuggingFaceHub"),
        ("Sentence Transformers", "from sentence_transformers import SentenceTransformer"),
        ("FAISS", "import faiss"),
    ]
    
    all_passed = True
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_passed = False
    
    return all_passed

def create_fixed_forecast_agent():
    """Create a fixed version of forecast_agent.py"""
    print("\n📝 Creating fixed forecast_agent.py...")
    
    fixed_imports = '''"""
app/agents/forecast_agent.py - Fixed version with correct imports
"""
# FIXED: Correct imports for LangChain 0.1.20+
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool  # Changed from langchain.agents
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish

# Import HuggingFace LLM
from app.llm.huggingface_llm import create_free_llm
'''
    
    # Check if file exists
    file_path = "app/agents/forecast_agent.py"
    if os.path.exists(file_path):
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the old imports
        if "from langchain.agents import Tool" in content:
            content = content.replace(
                "from langchain.agents import Tool, AgentExecutor, create_react_agent",
                "from langchain.agents import AgentExecutor, create_react_agent\nfrom langchain.tools import Tool"
            )
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Fixed imports in {file_path}")
            return True
        else:
            print(f"✅ Imports already correct in {file_path}")
            return True
    else:
        print(f"⚠️  File not found: {file_path}")
        print("Please ensure you're running this from the project root directory")
        return False

def main():
    """Main fix routine"""
    print("=" * 60)
    print("🔧 TCS Forecasting Agent - Import Fix Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Fix LangChain
    if not fix_langchain_imports():
        print("\n❌ Failed to fix dependencies")
        print("Try running manually:")
        print("  pip uninstall -y langchain langchain-community")
        print("  pip install langchain==0.1.20 langchain-community==0.0.38")
        return
    
    # Fix forecast_agent.py
    if not create_fixed_forecast_agent():
        print("\n⚠️  Could not automatically fix forecast_agent.py")
        print("Please manually update the imports:")
        print("  FROM: from langchain.agents import Tool, AgentExecutor, create_react_agent")
        print("  TO:   from langchain.agents import AgentExecutor, create_react_agent")
        print("        from langchain.tools import Tool")
    
    # Verify
    print("\n" + "=" * 60)
    if verify_imports():
        print("\n✅ All imports fixed successfully!")
        print("\nYou can now run:")
        print("  uvicorn app.main:app --reload")
    else:
        print("\n❌ Some imports still failing")
        print("\nTry these manual fixes:")
        print("1. Ensure Python 3.10 or 3.11 (not 3.13)")
        print("2. Create fresh virtual environment:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # or venv\\Scripts\\activate on Windows")
        print("3. Install from requirements.txt:")
        print("   pip install -r requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 