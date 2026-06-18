# Automotive LLM Benchmark: Rubric-to-Code Mapping

This document provides a "glass box" visual transparency map for the final review, linking the task requirements directly to the underlying Python architecture.

| Rubric Criterion (100% Weight) | Handling Module | Specific Function / Logic Implementation |
| :--- | :--- | :--- |
| **Multi-Provider Interface (20%)** | `provider_adapters.py` | `GeminiAdapter`, `GroqAdapter`, `OllamaAdapter` wrap `aiohttp` and `google-genai`. Implements `generate_with_retry` for automatic rate-limit management (exponential backoff). |
| **Latency/TPS Metrics (15%)** | `metrics_engine.py` | `calculate_tps()`. Captures precise asynchronous API response times (`api_time = time.time() - start_time`) and calculates token generation speed. |
| **Consistency Evaluation (15%)** | `metrics_engine.py` & `evaluation_framework.py` | `run_prompt_iterations()` executes a concurrent 5-iteration loop. `evaluate_consistency()` calculates mathematical Cosine Similarity and strictly checks for numerical spec drift. |
| **Hallucination Detection (25%)** | `hallucination_detector.py` | `evaluate()`. Analyzes output using Regex to enforce the SAE J2012 standard format (`[PBCU][0-9A-Fa-f]{4}`) and cross-references against mock vehicle registries. |
| **Architecture & Dashboards (25%)** | `generate_notebook.py` & `visualization_dashboard.ipynb` | Dynamically generates a secure Google Colab environment. Uses `google.colab.userdata` for API key protection and renders interactive Plotly Radar/Violin charts. |

---

## Output-First Visual Transparency
To prove the execution flow of the modules without running the full 3-hour benchmark, execute:

```bash
python3 demo.py
```
This script acts as a "Visual Transparency Map", simulating a single prompt through all three logical modules and printing a human-readable trace of the exact variables, token counts, and Regex matches occurring inside the engine.
