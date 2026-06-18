"""
File: demo.py
Purpose: Glass-box demonstration script for the Automotive LLM Benchmarking Framework.
This script executes a real prompt through the actual engine logic.
"""
import asyncio
from provider_adapters import OllamaAdapter
from evaluation_framework import EvaluationFramework

async def run_glass_box_demo():
    print("=" * 60)
    print("AUTOMOTIVE LLM BENCHMARK: GLASS BOX DEMONSTRATION")
    print("=" * 60)

    print("\n[INITIATING CORE ORCHESTRATOR]")
    # Initialize framework but we'll override its dataset to just 1 prompt for speed
    framework = EvaluationFramework()

    # We will test using local Ollama to ensure it runs instantly without cloud API rate limits
    adapter = OllamaAdapter(max_retries=1)

    sample_prompt_data = {
        "Category": "DTC Diagnostics",
        "Prompt": "What does DTC P0420 indicate in a modern vehicle?"
    }

    print(f"\n[MODULE: evaluation_framework.py & provider_adapters.py]")
    print(f"-> Sending 5 concurrent iterations of prompt: '{sample_prompt_data['Prompt']}' to Ollama...")

    # Execute the actual logic loop
    result = await framework.run_prompt_iterations(adapter, sample_prompt_data, iterations=5)

    print("\n[MODULE: metrics_engine.py]")
    print(f"-> Average Latency Calculated: {result['Avg_Latency_ms']:.2f} ms")
    print(f"-> Tokens Per Second (TPS): {result['Avg_Tokens_Per_Second']:.2f}")
    print(f"-> Cosine Consistency Score (across 5 iterations): {result['Consistency_Score']:.2f}%")
    print(f"-> Detected Numeric Spec Drift (Inconsistencies): {result['Numerical_Inconsistencies']}")

    print("\n[MODULE: hallucination_detector.py]")
    print(f"-> Hallucination Flags Triggered: {result['Hallucination_Flags']}")
    print(f"-> Final Hallucination Status: {result['Is_Hallucinating']}")
    print(f"-> Reasoning Quality Score: {result['Reasoning_Score']}/100")

    print("\n" + "=" * 60)
    print("GLASS BOX DEMONSTRATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_glass_box_demo())
