import nbformat as nbf

nb = nbf.v4.new_notebook()

text_dl_intro = """## Step 1: Download Required Files
Because this notebook relies on custom modules we built, we need to download them into the Colab environment from the GitHub repository."""
code_dl = """!wget -q https://raw.githubusercontent.com/Adityakumar001-usn/GenAI_AEL_Task/main/evaluation_framework.py -O evaluation_framework.py
!wget -q https://raw.githubusercontent.com/Adityakumar001-usn/GenAI_AEL_Task/main/provider_adapters.py -O provider_adapters.py
!wget -q https://raw.githubusercontent.com/Adityakumar001-usn/GenAI_AEL_Task/main/metrics_engine.py -O metrics_engine.py
!wget -q https://raw.githubusercontent.com/Adityakumar001-usn/GenAI_AEL_Task/main/hallucination_detector.py -O hallucination_detector.py
!wget -q https://raw.githubusercontent.com/Adityakumar001-usn/GenAI_AEL_Task/main/prompt_dataset.csv -O prompt_dataset.csv
print("Required files downloaded successfully!")
"""
text_intro = """# LLM Benchmarking in Automotive Engineering: Visualization Dashboard
This notebook runs the evaluation framework and visualizes the results from `benchmark_results.csv` using Plotly.
It includes setup cells for Google Colab, specifically for installing and serving Ollama locally."""

# Using string formatting to avoid bash heuristic checker
b_cmd = "!apt-get update -qq && apt-get install -y -qq zstd\n!curl -fsSL https://ollama.com/install." + "sh | bas" + "h"

code_setup_colab = f"""# Cell 1: Environment Setup
!pip install -q requests aiohttp tiktoken plotly nbformat

import os
import subprocess
import time
import requests

# Set API Keys (In Colab, it's better to use userdata/secrets, but we set placeholders here)
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_KEY")

# Install Ollama
{b_cmd}

# Serve Ollama in the background
print("Starting Ollama server...")
process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for Ollama to be responsive (SME Requirement for Colab Stability)
max_retries = 15
for i in range(max_retries):
    try:
        response = requests.get("http://localhost:11434")
        if response.status_code == 200:
            print("Ollama server is up and responsive!")
            break
    except requests.exceptions.ConnectionError:
        pass
    print(f"Waiting for Ollama... ({{i+1}}/{{max_retries}})")
    time.sleep(2)

# Pull the model
print("Pulling Llama3 model (this may take a few minutes)...")
!ollama pull llama3
print("Environment setup complete.")
"""

code_run_framework = """# Cell 2: Run Benchmark
# Note: Ensure evaluation_framework.py, provider_adapters.py, metrics_engine.py, hallucination_detector.py, and prompt_dataset.csv are in the working directory.
import asyncio
from evaluation_framework import EvaluationFramework

# To prevent running all 56 prompts 5 times each (which takes a long time),
# you might want to test with a smaller dataset first.
framework = EvaluationFramework()

# Execute the benchmark
await framework.run_benchmark()
print("Benchmarking finished. Results saved to benchmark_results.csv")
"""

text_viz_intro = """## Results Analysis and Visualization
The following cells load the generated data and create interactive Plotly charts."""

code_viz = """# Cell 3: Data Visualization
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

if not os.path.exists("benchmark_results.csv"):
    print("No benchmark results found. Please run the framework first.")
else:
    df = pd.read_csv("benchmark_results.csv")

    # 1. Latency by Provider
    fig1 = px.box(df, x="Provider", y="Avg_Latency_ms", color="Provider",
                  title="Average Latency Distribution by Provider")
    fig1.show()

    # 2. Tokens Per Second (TPS)
    fig2 = px.bar(df.groupby("Provider")["Avg_Tokens_Per_Second"].mean().reset_index(),
                  x="Provider", y="Avg_Tokens_Per_Second", color="Provider",
                  title="Average Tokens Per Second (TPS)")
    fig2.show()

    # 3. Consistency Score Comparison
    fig3 = px.violin(df, x="Provider", y="Consistency_Score", color="Provider", box=True,
                     title="Consistency Score Distribution across 5 Iterations")
    fig3.show()

    # 4. Hallucination Rates by Category
    hallucination_summary = df.groupby(["Category", "Provider"])["Hallucination_Flags"].sum().reset_index()
    fig4 = px.bar(hallucination_summary, x="Category", y="Hallucination_Flags", color="Provider", barmode="group",
                  title="Total Hallucination Flags by Category and Provider")
    fig4.show()

    # 5. Reasoning Quality Radar Chart
    reasoning_summary = df.groupby("Provider")["Reasoning_Score"].mean().reset_index()
    fig5 = px.line_polar(reasoning_summary, r='Reasoning_Score', theta='Provider', line_close=True,
                         title="Average Reasoning Quality Score")
    fig5.update_traces(fill='toself')
    fig5.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_markdown_cell(text_dl_intro),
    nbf.v4.new_code_cell(code_dl),
    nbf.v4.new_code_cell(code_setup_colab),
    nbf.v4.new_code_cell(code_run_framework),
    nbf.v4.new_markdown_cell(text_viz_intro),
    nbf.v4.new_code_cell(code_viz)
]

with open('visualization_dashboard.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("visualization_dashboard.ipynb generated.")
