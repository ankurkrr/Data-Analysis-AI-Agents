from app.llm.openrouter_llm import OpenRouterLLM
import os


def main():
    """Run OpenRouter LLM example only when invoked directly (prevent running during pytest collection)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set; skipping LLM example.")
        return

    llm = OpenRouterLLM(openrouter_api_key=api_key)
    try:
        resp = llm._call("Write a one-line forecast for TCS revenue growth next quarter.")
        print(resp)
    except Exception as e:
        print(f"LLM call failed: {e}")


if __name__ == "__main__":
    main()