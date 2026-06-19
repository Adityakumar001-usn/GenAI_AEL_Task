"""
File: evaluation_framework.py
Purpose: Core module for the Automotive LLM Benchmarking Framework.
This file has been comprehensively commented to ensure maximum readability and maintainability.
"""
import csv
import asyncio
import logging
import random
from typing import List, Dict, Any

from provider_adapters import GeminiAdapter, GroqAdapter, OllamaAdapter
from hallucination_detector import HallucinationDetector
from metrics_engine import MetricsEngine

# Set up logging for the orchestrator
logger = logging.getLogger("evaluation_framework")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class EvaluationFramework:
    """
    The Core Engine that orchestrates the entire benchmarking loop.
    It reads prompts, passes them to adapters, scores responses via metrics engines,
    and writes results to a CSV file.
    """
    def __init__(self, prompt_file: str = "prompt_dataset.csv", output_file: str = "benchmark_results.csv", delay_seconds: int = 0):
        self.prompt_file = prompt_file
        self.output_file = output_file
        self.delay_seconds = delay_seconds

        # Initialize the list of LLMs we want to test
        self.adapters = [
            GeminiAdapter(),
            GroqAdapter(),
            OllamaAdapter()
        ]

        # Tiered Traffic Control Layer 1: Concurrency Control (The Semaphore)
        # We explicitly restrict concurrency to 1 to ensure strictly sequential execution.
        # This prevents simultaneous bursts that instantly trigger HTTP 429 (Rate Limit Exceeded) bans.
        self.semaphore = asyncio.Semaphore(1)

        # Initialize evaluation modules
        self.hallucination_detector = HallucinationDetector()
        self.metrics_engine = MetricsEngine()

        # Load the dataset immediately upon initialization
        self.prompts = self._load_prompts()
        self.results = []

    def _load_prompts(self) -> List[Dict[str, str]]:
        """Reads the CSV dataset and loads the prompts into memory."""
        prompts = []
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompts.append(row) # Each row is a dict: {'Category': '...', 'Prompt': '...'}
            logger.info(f"Loaded {len(prompts)} prompts from {self.prompt_file}")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
        return prompts

    async def run_prompt_iterations(self, adapter, prompt_data: Dict[str, str], iterations: int = 5) -> Dict[str, Any]:
        """
        Executes a single prompt against a single provider multiple times (default 5).
        This loop is critical for measuring Consistency and Worst-case Latency.
        """
        category = prompt_data['Category']
        prompt_text = prompt_data['Prompt']

        # Arrays to hold data for all 5 iterations
        responses = []
        latencies = []

        prompt_tokens = 0
        total_completion_tokens = 0

        logger.info(f"Running '{adapter.provider_name}' on prompt: {prompt_text[:30]}... ({iterations} iterations)")

        # --- The 5-Iteration Loop ---
        for i in range(iterations):
            try:
                # Call the API asynchronously
                response_text, metadata = await adapter.generate_with_retry(prompt_text)

                # Store the results for later aggregation
                responses.append(response_text)
                latencies.append(metadata.get('latency_ms', 0))

                # Extract token counts
                pt = metadata.get('prompt_tokens', 0)
                ct = metadata.get('completion_tokens', 0)

                # Fallback: If API didn't provide token counts, calculate them manually using tiktoken
                if pt == 0:
                    pt = self.metrics_engine.calculate_token_count(prompt_text)
                if ct == 0:
                    ct = self.metrics_engine.calculate_token_count(response_text)

                prompt_tokens = max(prompt_tokens, pt) # Input size stays the same
                total_completion_tokens += ct # Sum output sizes for averaging

            except Exception as e:
                # If a run completely fails (even after Layer 3 retries), log it and append blank data
                logger.error(f"Iteration {i+1} failed for {adapter.provider_name}: {e}")
                responses.append("")
                latencies.append(0)

            # Tiered Traffic Control Layer 2 & 3: Configurable Throughput + Traffic Naturalization (Jitter)
            if self.delay_seconds > 0:
                # Add a random jitter (0 to 3 seconds) to the base delay to make traffic patterns
                # look organic and prevent triggering sophisticated bot/rate-limit firewalls.
                jitter = random.uniform(0, 3)
                await asyncio.sleep(self.delay_seconds + jitter)

        # --- Aggregate Phase ---
        # Filter out failed responses
        valid_responses = [r for r in responses if r]
        avg_completion_tokens = total_completion_tokens / max(1, len(valid_responses))

        # Calculate Latency Metrics (Average and Worst-case)
        valid_latencies = [l for l in latencies if l > 0]
        avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0
        worst_latency = max(valid_latencies) if valid_latencies else 0 # Worst-case is the maximum latency

        # Calculate overall Tokens Per Second (TPS)
        avg_tps = self.metrics_engine.calculate_tps(avg_completion_tokens, avg_latency)

        # Run the 5 iterations through the Consistency engine
        consistency_metrics = self.metrics_engine.evaluate_consistency(valid_responses)

        # For deeper semantic analysis (Length, Hallucination, Reasoning),
        # we evaluate the first successful response as the representative sample.
        rep_response = valid_responses[0] if valid_responses else ""

        # Calculate Length metrics
        word_count = self.metrics_engine.calculate_word_count(rep_response)
        compression_ratio = self.metrics_engine.calculate_compression_ratio(prompt_tokens, avg_completion_tokens)
        info_density = self.metrics_engine.calculate_information_density(rep_response)

        # Run Hallucination and Reasoning Checks
        print(f"[evaluation_framework.py] | INPUT: Response sample | PROCESS: Routing to Hallucination Detector | OUTPUT: Executing SAE/ISO validation...")
        hallucination_metrics = self.hallucination_detector.evaluate(rep_response)
        reasoning_metrics = self.metrics_engine.evaluate_reasoning_quality(rep_response)

        # Return a compiled dictionary mapping to the final CSV columns
        return {
            "Provider": adapter.provider_name,
            "Category": category,
            "Prompt": prompt_text,
            "Avg_Latency_ms": avg_latency,
            "Worst_Latency_ms": worst_latency,
            "Avg_Tokens_Per_Second": avg_tps,
            "Prompt_Tokens": prompt_tokens,
            "Avg_Completion_Tokens": avg_completion_tokens,
            "Word_Count": word_count,
            "Compression_Ratio": compression_ratio,
            "Information_Density": info_density,
            "Consistency_Score": consistency_metrics["overall_consistency_score"],
            "Numerical_Inconsistencies": consistency_metrics["numerical_inconsistencies"],
            "Hallucination_Flags": hallucination_metrics["hallucination_flags_count"],
            "Is_Hallucinating": hallucination_metrics["is_hallucinating"],
            "Reasoning_Score": reasoning_metrics["reasoning_score"]
        }

    async def run_benchmark(self):
        """
        The main public method. Loops through all adapters and prompts,
        executes the benchmark, and saves results dynamically.
        """
        if not self.prompts:
            logger.error("No prompts to process. Exiting.")
            return

        # Define the structure of the output CSV
        fieldnames = [
            "Provider", "Category", "Prompt", "Avg_Latency_ms", "Worst_Latency_ms",
            "Avg_Tokens_Per_Second", "Prompt_Tokens", "Avg_Completion_Tokens",
            "Word_Count", "Compression_Ratio", "Information_Density",
            "Consistency_Score", "Numerical_Inconsistencies", "Hallucination_Flags",
            "Is_Hallucinating", "Reasoning_Score"
        ]

        # Loop through each LLM provider sequentially
        for adapter in self.adapters:
            logger.info(f"--- Starting benchmark for {adapter.provider_name} ---")

            # Loop through all 56 generated prompts
            for prompt_data in self.prompts:
                # Enforce Concurrency Throttling: Wait until a semaphore slot is open.
                # If 2 requests are already processing, this block will pause and wait,
                # ensuring we never overwhelm the provider's API limits.
                async with self.semaphore:
                    # Await the execution of the 5-iteration loop
                    result = await self.run_prompt_iterations(adapter, prompt_data)

                    # Append to memory
                    self.results.append(result)

                    # Write to disk incrementally to prevent data loss if the script crashes
                    self._save_results(fieldnames)

        logger.info("Benchmarking complete!")

    def _save_results(self, fieldnames):
        """Helper function to write the current results array to the CSV file."""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    framework = EvaluationFramework()
    pass
