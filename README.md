# GenAI_AEL_Task: Automotive LLM Benchmarking Framework

## Project Overview
This repository contains a comprehensive, production-ready evaluation framework designed to automatically benchmark multiple Large Language Models (LLMs) on complex automotive engineering tasks.

The framework currently supports the evaluation of **Gemini 1.5 Flash** (now natively utilizing the `google-genai` SDK for compatibility with the modern `AQ.` Auth Keys), **Groq (Llama 3)**, and **Ollama (Local Llama 3)** across 50+ specialized prompts in 8 distinct automotive categories.

It evaluates models not just on traditional metrics like Latency and Tokens Per Second (TPS), but strictly enforces domain-specific checks including formatting adherence to **SAE J2012 DTCs**, validation of **ISO/SAE standards**, numeric specification drift tracking, and advanced technical hallucination detection.

## End-to-End (E2E) Pipeline Workflow
1. **Prompt Ingestion:** The `evaluation_framework.py` orchestrator loads the `prompt_dataset.csv`.
2. **Provider Execution:** It passes each prompt to the unified `provider_adapters.py`. The adapters handle rate limits via exponential backoff and execute the prompt asynchronously.
3. **Consistency Loop:** Every prompt is run **5 times** per provider.
4. **Metrics Calculation:** Results are passed to the `metrics_engine.py` which calculates response length, TPS, compression ratios, and mathematical Cosine Similarity across the 5 iterations.
5. **Hallucination Checking:** Results are processed by the `hallucination_detector.py` which uses Regex to extract technical data and validates it against mock SAE/ISO registries.
6. **Data Output:** All calculated metrics are aggregated and saved to `benchmark_results.csv`.
7. **Visualization:** The `visualization_dashboard.ipynb` reads the CSV and generates interactive Plotly charts.

## Deliverable Structure
*   `evaluation_framework.py`: Core orchestrator engine.
*   `provider_adapters.py`: Connectors for Gemini, Groq, and Ollama (includes robust retry logic and data masking).
*   `prompt_dataset.csv`: 56 benchmark prompts covering diagnostics, predictive maintenance, CAN bus analysis, EV batteries, ADAS, cybersecurity, vehicle dynamics, and service documentation.
*   `hallucination_detector.py`: Fact-checking module for DTCs and standards.
*   `metrics_engine.py`: Computes length, latency, consistency, and reasoning scores.
*   `visualization_dashboard.ipynb`: Interactive data analysis dashboard.
*   `benchmark_results.csv`: Raw experimental results (generated at runtime).

## Requirements & Setup

### Running in Google Colab (Recommended)
This is the easiest method as it handles all dependency installation and local background server setup automatically.

1. Open `visualization_dashboard.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Ensure `prompt_dataset.csv`, `evaluation_framework.py`, `provider_adapters.py`, `metrics_engine.py`, and `hallucination_detector.py` are uploaded to the Colab session storage.
3. Open the 🔑 Secrets tab on the left sidebar in Colab and add two secrets: `GEMINI_API_KEY` and `GROQ_API_KEY`.
4. Run the newly added **Pre-Flight Verification Runner** (Cell 1.5) to test connectivity with both cloud APIs and the local Ollama instance before committing to the full benchmark loop.
5. Go to **Runtime > Run all**. The notebook will automatically download Ollama, start the local server, run the 5-iteration benchmark loop, and display the final Plotly charts.

### Running Locally (Terminal / VS Code)

1. **Install Python Dependencies:**
   ```bash
   pip install requests aiohttp tiktoken plotly pandas jupyterlab nbformat google-genai scikit-learn
   ```

2. **Install & Start Ollama:**
   - Download from [ollama.com](https://ollama.com/)
   - Start the server in the background: `ollama serve`
   - Pull the model: `ollama pull llama3`

3. **Set API Keys in Environment:**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   export GROQ_API_KEY="your_api_key_here"
   ```

4. **Run the Benchmark:**
   To generate the `benchmark_results.csv`, you can run the Colab Notebook locally or execute a simple python script importing the framework.

5. **View Results:**
   Open the `visualization_dashboard.ipynb` using Jupyter Lab or Jupyter Notebook to view the interactive data analysis.
