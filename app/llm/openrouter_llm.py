from langchain.llms.base import LLM
from typing import Any, List, Optional
import requests
import os
from pydantic import Field
import time
import random

class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter with retry logic"""
    
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "google/gemini-2.0-flash-exp:free"))
    base_url: str = Field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") + "/chat/completions")
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=5.0)  # 5 seconds for free models

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Send user prompt to OpenRouter model with retry logic"""
        if not self.openrouter_api_key:
            raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.base_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    # Rate limit hit - wait and retry
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit. Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ValueError(f"OpenRouter API Rate Limit: {response.text}")
                else:
                    raise ValueError(f"OpenRouter API Error: {response.status_code} - {response.text}")
                    
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
        
        raise ValueError(f"OpenRouter API failed after {self.max_retries} retries: {last_error}")
