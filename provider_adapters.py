import os
import time
import json
import logging
import asyncio
import requests
import aiohttp
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

# Suppress verbose logging from third party HTTP libraries to keep our console clean
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set up standard logging for our application
logger = logging.getLogger("provider_adapters")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
# Define the format of the log messages (timestamp, logger name, severity, message)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)

def redact_sensitive_data(log_string: str) -> str:
    """
    Masks potential API keys in the log string to prevent security leaks.
    This fulfills the requirement to securely mask API keys during interaction logging.
    """
    import re
    # Mask common API key patterns (alphanumeric strings longer than 20 chars often passed in headers or URLs)
    redacted = re.sub(r'(api_key=|Bearer |"api_key":\s*")([a-zA-Z0-9_-]{15,})', r'\1***REDACTED***', log_string)
    # Mask Groq specific keys starting with 'gsk_'
    redacted = re.sub(r'(gsk_[a-zA-Z0-9]{20,})', '***REDACTED_GROQ_KEY***', redacted)
    # Mask Gemini specific keys starting with 'AIza'
    redacted = re.sub(r'(AIza[0-9A-Za-z-_]{35})', '***REDACTED_GEMINI_KEY***', redacted)
    return redacted

class BaseAdapter(ABC):
    """
    Abstract Base Class that defines the unified API interface for all LLM providers.
    It handles common logic like automatic retry handling and interaction logging.
    """
    def __init__(self, provider_name: str, model_name: str, max_retries: int = 3):
        self.provider_name = provider_name
        self.model_name = model_name
        self.max_retries = max_retries # How many times to retry on failure

    def log_interaction(self, prompt: str, response: str, latency: float, error: str = None):
        """
        Logs the interaction with the LLM provider, including latency and errors.
        Ensures all sensitive data is redacted before printing.
        """
        # Create a dictionary of the log data
        log_data = {
            "provider": self.provider_name,
            "model": self.model_name,
            # Only log the first 50 characters of the prompt to avoid huge logs
            "prompt_preview": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "latency": latency,
            "error": error
        }
        # Convert dictionary to JSON string
        log_str = json.dumps(log_data)
        # Scrub the JSON string for API keys
        safe_log_str = redact_sensitive_data(log_str)

        # Log as an error if an error occurred, otherwise log as info
        if error:
            logger.error(f"Interaction Error: {safe_log_str}")
        else:
            logger.info(f"Interaction Success: {safe_log_str}")

    @abstractmethod
    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Abstract method to be implemented by child classes.
        Must return the generated text and a metadata dictionary.
        """
        pass

    async def generate_with_retry(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Executes the generate_response method with an exponential backoff retry mechanism.
        This fulfills the 'automatic retry handling' requirement.
        """
        retries = 0
        backoff = 2 # Start by waiting 2 seconds before retrying

        # Loop until we exceed the maximum allowed retries
        while retries <= self.max_retries:
            start_time = time.time() # Record start time to measure total latency
            try:
                # Attempt to call the specific provider's API asynchronously
                response_text, metadata = await self.generate_response(prompt)

                # Calculate how long the call took
                latency = time.time() - start_time
                metadata['latency_ms'] = latency * 1000 # Convert to milliseconds

                # Log the successful interaction
                self.log_interaction(prompt, response_text, latency)
                return response_text, metadata

            except Exception as e:
                # If an error occurs, calculate latency up to the point of failure
                latency = time.time() - start_time
                # Log the failed interaction
                self.log_interaction(prompt, "", latency, error=str(e))

                # If we've hit the retry limit, raise the error to crash or skip
                if retries == self.max_retries:
                    logger.error(f"{self.provider_name} adapter failed after {self.max_retries} retries. Final error: {e}")
                    raise e

                # Otherwise, log a warning, wait, and try again
                logger.warning(f"{self.provider_name} request failed (attempt {retries+1}/{self.max_retries}). Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2 # Exponentially increase wait time (2s, 4s, 8s...)
                retries += 1

class GeminiAdapter(BaseAdapter):
    """Adapter specifically for Google's Gemini Flash model."""
    def __init__(self, model_name="gemini-1.5-flash", max_retries=3):
        super().__init__("Gemini", model_name, max_retries)
        # Fetch API key from environment variables
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Gemini generation will fail.")

        # Set the Google REST API endpoint
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Asynchronously calls the Gemini API."""
        if not self.api_key:
            raise ValueError("API Key missing")

        headers = {"Content-Type": "application/json"}
        # Construct the payload according to Gemini's API spec
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2} # Low temperature for more consistent, factual responses
        }
        url_with_key = f"{self.api_url}?key={self.api_key}"

        # Use aiohttp for async network requests
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.post(url_with_key, headers=headers, json=payload) as response:
                api_time = time.time() - start_time

                # Handle rate limits specifically (HTTP 429)
                if response.status == 429:
                    raise Exception("Rate limit exceeded (429)")

                # Throw an error for other bad HTTP statuses (e.g., 500 Server Error)
                response.raise_for_status()
                data = await response.json()

                try:
                    # Extract the text from the complex JSON response structure
                    text = data['candidates'][0]['content']['parts'][0]['text']

                    # Extract exact token usage from the API if available
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
    """Adapter specifically for Groq's Llama 3 API."""
    def __init__(self, model_name="llama3-8b-8192", max_retries=3):
        super().__init__("Groq", model_name, max_retries)
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY environment variable not set. Groq generation will fail.")
        # Groq uses an OpenAI-compatible endpoint structure
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Asynchronously calls the Groq API."""
        if not self.api_key:
            raise ValueError("API Key missing")

        # Groq requires the API key in the Authorization header
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Standard OpenAI payload format
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

                # Extract text
                text = data['choices'][0]['message']['content']
                # Extract token usage
                usage = data.get('usage', {})
                metadata = {
                    "api_response_time_ms": api_time * 1000,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                }
                return text, metadata

class OllamaAdapter(BaseAdapter):
    """Adapter for a locally running instance of Ollama (Llama 3)."""
    def __init__(self, model_name="llama3", max_retries=3, host="http://localhost:11434"):
        super().__init__("Ollama", model_name, max_retries)
        # Point to the local machine's port where Ollama runs
        self.api_url = f"{host}/api/generate"

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Asynchronously calls the local Ollama API."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False, # Wait for the full response, do not stream tokens
            "options": {
                "temperature": 0.2
            }
        }

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                # Call local endpoint
                async with session.post(self.api_url, json=payload) as response:
                    api_time = time.time() - start_time
                    response.raise_for_status()
                    data = await response.json()

                    text = data.get('response', '')
                    # Ollama provides specific keys for token evaluation counts
                    metadata = {
                        "api_response_time_ms": api_time * 1000,
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0)
                    }
                    return text, metadata
            except aiohttp.ClientConnectorError:
                # If connection fails entirely, it usually means the background server isn't running
                raise Exception("Failed to connect to Ollama. Is the service running?")
