import csv
import asyncio
import logging
from typing import List, Dict, Any

from provider_adapters import GeminiAdapter, GroqAdapter, OllamaAdapter
from hallucination_detector import HallucinationDetector
from metrics_engine import MetricsEngine

logger = logging.getLogger("evaluation_framework")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class EvaluationFramework:
    def __init__(self, prompt_file: str = "prompt_dataset.csv", output_file: str = "benchmark_results.csv"):
        self.prompt_file = prompt_file
        self.output_file = output_file

        self.adapters = [
            GeminiAdapter(),
            GroqAdapter(),
            OllamaAdapter()
        ]

        self.hallucination_detector = HallucinationDetector()
        self.metrics_engine = MetricsEngine()
        self.prompts = self._load_prompts()
        self.results = []

    def _load_prompts(self) -> List[Dict[str, str]]:
        prompts = []
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompts.append(row)
            logger.info(f"Loaded {len(prompts)} prompts from {self.prompt_file}")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
        return prompts

    async def run_prompt_iterations(self, adapter, prompt_data: Dict[str, str], iterations: int = 5) -> Dict[str, Any]:
        category = prompt_data['Category']
        prompt_text = prompt_data['Prompt']

        responses = []
        latencies = []
        prompt_tokens = 0
        total_completion_tokens = 0

        logger.info(f"Running '{adapter.provider_name}' on prompt: {prompt_text[:30]}... ({iterations} iterations)")

        for i in range(iterations):
            try:
                response_text, metadata = await adapter.generate_with_retry(prompt_text)
                responses.append(response_text)
                latencies.append(metadata.get('latency_ms', 0))

                # Use provided tokens or estimate
                pt = metadata.get('prompt_tokens', 0)
                ct = metadata.get('completion_tokens', 0)

                if pt == 0:
                    pt = self.metrics_engine.calculate_token_count(prompt_text)
                if ct == 0:
                    ct = self.metrics_engine.calculate_token_count(response_text)

                prompt_tokens = max(prompt_tokens, pt) # Should be consistent across runs
                total_completion_tokens += ct

            except Exception as e:
                logger.error(f"Iteration {i+1} failed for {adapter.provider_name}: {e}")
                responses.append("")
                latencies.append(0)

        # Calculate Aggregate Metrics
        valid_responses = [r for r in responses if r]
        avg_completion_tokens = total_completion_tokens / max(1, len(valid_responses))

        # Latency Metrics
        valid_latencies = [l for l in latencies if l > 0]
        avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0
        worst_latency = max(valid_latencies) if valid_latencies else 0

        # TPS calculation
        avg_tps = self.metrics_engine.calculate_tps(avg_completion_tokens, avg_latency)

        # Consistency
        consistency_metrics = self.metrics_engine.evaluate_consistency(valid_responses)

        # For length, hallucination, and reasoning, we evaluate the best/longest response or an average.
        # Let's use the first successful response as the representative for deep analysis
        rep_response = valid_responses[0] if valid_responses else ""

        word_count = self.metrics_engine.calculate_word_count(rep_response)
        compression_ratio = self.metrics_engine.calculate_compression_ratio(prompt_tokens, avg_completion_tokens)
        info_density = self.metrics_engine.calculate_information_density(rep_response)

        hallucination_metrics = self.hallucination_detector.evaluate(rep_response)
        reasoning_metrics = self.metrics_engine.evaluate_reasoning_quality(rep_response)

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
        if not self.prompts:
            logger.error("No prompts to process. Exiting.")
            return

        # Prepare CSV output
        fieldnames = [
            "Provider", "Category", "Prompt", "Avg_Latency_ms", "Worst_Latency_ms",
            "Avg_Tokens_Per_Second", "Prompt_Tokens", "Avg_Completion_Tokens",
            "Word_Count", "Compression_Ratio", "Information_Density",
            "Consistency_Score", "Numerical_Inconsistencies", "Hallucination_Flags",
            "Is_Hallucinating", "Reasoning_Score"
        ]

        # Process sequentially to avoid absolutely overwhelming rate limits,
        # though production systems might use asyncio.gather with semaphores.
        for adapter in self.adapters:
            logger.info(f"--- Starting benchmark for {adapter.provider_name} ---")
            for prompt_data in self.prompts:
                result = await self.run_prompt_iterations(adapter, prompt_data)
                self.results.append(result)

                # Write incrementally to avoid data loss
                self._save_results(fieldnames)

        logger.info("Benchmarking complete!")

    def _save_results(self, fieldnames):
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    framework = EvaluationFramework()
    # To test locally quickly, we could slice the prompts, but for production we run all.
    # We will just verify syntax for now.
    pass
