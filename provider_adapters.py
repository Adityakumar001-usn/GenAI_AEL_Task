import os
import time
import json
import logging
import asyncio
import requests
import aiohttp
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

# Suppress verbose logging from third party libs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set up standard logging
logger = logging.getLogger("provider_adapters")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)

def redact_sensitive_data(log_string: str) -> str:
    """Masks potential API keys in the log string."""
    # Simple heuristic to mask strings that look like API keys
    import re
    # Mask common API key patterns (alphanumeric strings longer than 20 chars often passed in headers or URLs)
    redacted = re.sub(r'(api_key=|Bearer |"api_key":\s*")([a-zA-Z0-9_-]{15,})', r'\1***REDACTED***', log_string)
    # Also catch groq/gemini keys specifically if logged
    redacted = re.sub(r'(gsk_[a-zA-Z0-9]{20,})', '***REDACTED_GROQ_KEY***', redacted)
    redacted = re.sub(r'(AIza[0-9A-Za-z-_]{35})', '***REDACTED_GEMINI_KEY***', redacted)
    return redacted

class BaseAdapter(ABC):
    def __init__(self, provider_name: str, model_name: str, max_retries: int = 3):
        self.provider_name = provider_name
        self.model_name = model_name
        self.max_retries = max_retries

    def log_interaction(self, prompt: str, response: str, latency: float, error: str = None):
        log_data = {
            "provider": self.provider_name,
            "model": self.model_name,
            "prompt_preview": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "latency": latency,
            "error": error
        }
        log_str = json.dumps(log_data)
        safe_log_str = redact_sensitive_data(log_str)
        if error:
            logger.error(f"Interaction Error: {safe_log_str}")
        else:
            logger.info(f"Interaction Success: {safe_log_str}")

    @abstractmethod
    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generates a response from the LLM.
        Returns a tuple of (response_text, metadata_dict)
        Metadata should include 'latency_ms', and ideally token counts if available.
        """
        pass

    async def generate_with_retry(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        retries = 0
        backoff = 2
        while retries <= self.max_retries:
            start_time = time.time()
            try:
                response_text, metadata = await self.generate_response(prompt)
                latency = time.time() - start_time
                metadata['latency_ms'] = latency * 1000
                self.log_interaction(prompt, response_text, latency)
                return response_text, metadata
            except Exception as e:
                latency = time.time() - start_time
                self.log_interaction(prompt, "", latency, error=str(e))
                if retries == self.max_retries:
                    logger.error(f"{self.provider_name} adapter failed after {self.max_retries} retries. Final error: {e}")
                    raise e
                logger.warning(f"{self.provider_name} request failed (attempt {retries+1}/{self.max_retries}). Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2
                retries += 1

class GeminiAdapter(BaseAdapter):
    def __init__(self, model_name="gemini-1.5-flash", max_retries=3):
        super().__init__("Gemini", model_name, max_retries)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Gemini generation will fail.")

        # We use aiohttp directly for async support and finer control over latency tracking
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("API Key missing")

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2} # Low temp for consistency
        }
        url_with_key = f"{self.api_url}?key={self.api_key}"

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.post(url_with_key, headers=headers, json=payload) as response:
                api_time = time.time() - start_time
                if response.status == 429:
                    raise Exception("Rate limit exceeded (429)")
                response.raise_for_status()
                data = await response.json()

                try:
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    # Gemini sometimes provides token usage in usageMetadata
                    usage = data.get('usageMetadata', {})
                    metadata = {
                        "api_response_time_ms": api_time * 1000,
                        "prompt_tokens": usage.get("promptTokenCount", 0),
                        "completion_tokens": usage.get("candidatesTokenCount", 0)
                    }
                    return text, metadata
                except KeyError as e:
                    raise Exception(f"Unexpected API response format: {data}") from e


class GroqAdapter(BaseAdapter):
    def __init__(self, model_name="llama3-8b-8192", max_retries=3):
        super().__init__("Groq", model_name, max_retries)
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY environment variable not set. Groq generation will fail.")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("API Key missing")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                api_time = time.time() - start_time
                if response.status == 429:
                    raise Exception("Rate limit exceeded (429)")
                response.raise_for_status()
                data = await response.json()

                text = data['choices'][0]['message']['content']
                usage = data.get('usage', {})
                metadata = {
                    "api_response_time_ms": api_time * 1000,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                }
                return text, metadata

class OllamaAdapter(BaseAdapter):
    def __init__(self, model_name="llama3", max_retries=3, host="http://localhost:11434"):
        super().__init__("Ollama", model_name, max_retries)
        self.api_url = f"{host}/api/generate"

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.post(self.api_url, json=payload) as response:
                    api_time = time.time() - start_time
                    response.raise_for_status()
                    data = await response.json()

                    text = data.get('response', '')
                    metadata = {
                        "api_response_time_ms": api_time * 1000,
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0)
                    }
                    return text, metadata
            except aiohttp.ClientConnectorError:
                raise Exception("Failed to connect to Ollama. Is the service running?")
