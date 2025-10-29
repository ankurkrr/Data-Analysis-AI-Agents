from langchain.llms.base import LLM
from typing import Any, List, Optional
import requests
import os
from pydantic import Field
import time
import random
import logging
import json

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
        """Send user prompt to OpenRouter model with retry and model-rotation logic.

        Behavior:
        - Supports rotating across models set in the env var OPENROUTER_MODEL_LIST (comma-separated).
        - Will attempt each configured model and retry with exponential backoff on 429s or network errors.
        - If no model list is provided, falls back to the single model in self.model.
        - Provides clearer error messages including the response body when shapes are unexpected.
        """
        if not self.openrouter_api_key:
            raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment.")

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        # Build the models rotation list from env var or the single configured model
        models_env = os.getenv("OPENROUTER_MODEL_LIST", "").strip()
        if models_env:
            models = [m.strip() for m in models_env.split(",") if m.strip()]
        else:
            models = [self.model]

        # Total attempts = (max_retries + 1) rounds across available models
        total_attempts = (self.max_retries + 1) * max(1, len(models))
        last_error = None

        for attempt in range(total_attempts):
            # Choose model in round-robin across attempts
            model = models[attempt % len(models)]
            # How many times we've retried this specific model (for backoff)
            model_retry_round = attempt // len(models)

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }

            try:
                logging.debug("OpenRouter call attempt=%d model=%s", attempt + 1, model)
                response = requests.post(self.base_url, headers=headers, json=payload)

                # Success
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        raise ValueError(f"OpenRouter API returned non-JSON response: {response.text}")

                    # Chat-like shape
                    if isinstance(data, dict):
                        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                            choice = data["choices"][0]
                            if isinstance(choice, dict):
                                if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                                    return choice["message"]["content"]
                                if "text" in choice:
                                    return choice["text"]

                    # Alternative shapes
                    if isinstance(data, dict) and "output" in data:
                        out = data["output"]
                        if isinstance(out, list) and out:
                            first = out[0]
                            if isinstance(first, dict):
                                if "content" in first:
                                    return first["content"]
                                if "text" in first:
                                    return first["text"]
                        if isinstance(out, str):
                            return out

                    raise ValueError(f"OpenRouter API returned unexpected JSON shape: {json.dumps(data)[:1500]}")

                # Rate limited on this model - try next model/round until exhausted
                elif response.status_code == 429:
                    last_error = response.text
                    if attempt < total_attempts - 1:
                        wait_time = self.retry_delay * (2 ** model_retry_round) + random.uniform(0, 1)
                        logging.warning("Rate limit for model %s. Waiting %.1fs before trying next model/round.", model, wait_time)
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ValueError(f"OpenRouter API Rate Limit: {response.text}")

                else:
                    # Other HTTP errors - include body for easier debugging
                    raise ValueError(f"OpenRouter API Error: {response.status_code} - {response.text}")

            except requests.RequestException as e:
                last_error = e
                if attempt < total_attempts - 1:
                    wait_time = self.retry_delay * (2 ** model_retry_round) + random.uniform(0, 1)
                    logging.warning("Network error calling OpenRouter (model=%s): %s. Retrying after %.1fs", model, str(e), wait_time)
                    time.sleep(wait_time)
                    continue

        # If we exit loop, all attempts failed
        raise ValueError(f"OpenRouter API failed after {total_attempts} attempts across models {models}: {last_error}")


class FakeOpenRouterLLM(LLM):
    """A simple deterministic LLM used for CI/dev when a real API key is not
    available. It returns a minimal but valid JSON forecast skeleton so the
    rest of the pipeline can be exercised without calling external APIs.
    """
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Return a minimal JSON following the forecast schema so downstream
        # validation can proceed in tests/dev.
        minimal = {
            "__fake_output": True,
            "metadata": {"ticker": "TCS", "request_id": "fake-req", "analysis_date": "1970-01-01T00:00:00Z", "quarters_analyzed": []},
            "numeric_trends": {},
            "qualitative_summary": {},
            "forecast": {"outlook_text": "[FAKE] No LLM available in test mode", "numeric_projection": {}, "confidence": 0.0},
            "risks_and_opportunities": {},
            "sources": []
        }
        import json
        return json.dumps(minimal)
    
    @property
    def _llm_type(self) -> str:
        return "fake-openrouter"


def get_llm() -> LLM:
    """Factory for LLM: returns a FakeOpenRouterLLM when FORCE_FAKE_LLM is set,
    otherwise returns the real OpenRouterLLM wrapper.
    """
    # If FORCE_FAKE_LLM is set, always return the fake LLM (explicit test/CIs)
    if os.getenv("FORCE_FAKE_LLM", "0").lower() in ("1", "true", "yes"):
        logging.warning("FORCE_FAKE_LLM is set; using FakeOpenRouterLLM")
        return FakeOpenRouterLLM()

    # If ALLOW_AUTO_FAKE is set, return a resilient wrapper that will attempt
    # the real OpenRouterLLM and fall back to FakeOpenRouterLLM on errors.
    if os.getenv("ALLOW_AUTO_FAKE", "0").lower() in ("1", "true", "yes"):
        return ResilientOpenRouterLLM()

    # Default: use the real LLM
    return OpenRouterLLM()


class ResilientOpenRouterLLM(LLM):
    """Wrapper that tries the real OpenRouterLLM and, on failure, falls back
    to the FakeOpenRouterLLM when ALLOW_AUTO_FAKE is enabled. The fallback is
    opt-in via the ALLOW_AUTO_FAKE env var and the response JSON will be
    annotated with "__fake_output": true so callers can detect it.
    """
    def __init__(self):
        # instantiate both but use private attribute names to avoid pydantic/BaseModel
        # assignment validation errors (LLM is a pydantic model).
        object.__setattr__(self, "_real", OpenRouterLLM())
        object.__setattr__(self, "_fake", FakeOpenRouterLLM())

    @property
    def _llm_type(self) -> str:
        return "resilient-openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            return object.__getattribute__(self, "_real")._call(prompt, stop=stop)
        except Exception as e:
            # Only fall back if ALLOW_AUTO_FAKE is explicitly enabled
            if os.getenv("ALLOW_AUTO_FAKE", "0").lower() in ("1", "true", "yes"):
                logging.warning("OpenRouterLLM call failed; ALLOW_AUTO_FAKE is enabled — returning fake LLM response. Error: %s", str(e))
                fake_text = object.__getattribute__(self, "_fake")._call(prompt, stop=stop)
                # try to annotate JSON; if fake_text already JSON, inject flag
                try:
                    parsed = json.loads(fake_text)
                    if isinstance(parsed, dict):
                        parsed["__fake_output"] = True
                        return json.dumps(parsed)
                except Exception:
                    # not JSON, wrap into a JSON envelope
                    envelope = {"__fake_output": True, "content": fake_text}
                    return json.dumps(envelope)
                return fake_text
            # If not allowed, re-raise
            raise
