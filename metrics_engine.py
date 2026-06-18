import re
import math
import tiktoken
from typing import List, Dict, Any
from collections import Counter

class MetricsEngine:
    def __init__(self):
        # We use a standard encoding for token estimation if the API doesn't provide exact counts
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # --- A. Response Length Metrics ---

    def calculate_word_count(self, text: str) -> int:
        return len(text.split())

    def calculate_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def calculate_compression_ratio(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Ratio of output size to input size. Higher means more verbose elaboration."""
        if prompt_tokens == 0:
            return 0.0
        return completion_tokens / prompt_tokens

    def calculate_information_density(self, text: str) -> float:
        """
        A heuristic for information density.
        Calculated as the ratio of unique 'significant' words to total words.
        We approximate by removing short stop-word-like lengths and taking unique count.
        """
        words = [w.lower() for w in re.findall(r'\b\w+\b', text) if len(w) > 3]
        if not words:
            return 0.0
        unique_words = set(words)
        return len(unique_words) / len(words)

    # --- B. Latency Metrics ---
    # These are calculated dynamically during the run in evaluation_framework based on adapter metadata

    def calculate_tps(self, tokens: int, latency_ms: float) -> float:
        if latency_ms == 0:
            return 0.0
        return tokens / (latency_ms / 1000.0)

    # --- C. Consistency Metrics ---

    def _cosine_similarity(self, vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x]**2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        return float(numerator) / denominator

    def _text_to_vector(self, text: str) -> Dict[str, int]:
        words = re.findall(r'\w+', text.lower())
        return Counter(words)

    def extract_numerical_data(self, text: str) -> set:
        """Extracts numbers with their immediate preceding/succeeding context to track technical data points."""
        # e.g., "400V", "1.5 Amps", "95%"
        pattern = re.compile(r'\b(\d+(?:\.\d+)?\s?[a-zA-Z%]+)\b')
        return set(pattern.findall(text))

    def evaluate_consistency(self, responses: List[str]) -> Dict[str, Any]:
        """
        Calculates consistency across 5 iterations of a prompt.
        Uses pairwise cosine similarity and numerical data variance.
        """
        if len(responses) < 2:
            return {"cosine_similarity_score": 1.0, "numerical_inconsistencies": 0}

        # 1. Cosine Similarity (TF-IDF approximation via word counts)
        vectors = [self._text_to_vector(r) for r in responses]
        similarities = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarities.append(self._cosine_similarity(vectors[i], vectors[j]))

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # 2. Numerical Consistency (Strict SME requirement)
        # We extract all numerical quantities from each run. If a run introduces a quantity
        # that conflicts with another run (e.g. one says 400V, another says 800V), we flag it.
        # This is a basic implementation: we collect all unique numbers. If the variance is high, flag.
        extracted_data_per_run = [self.extract_numerical_data(r) for r in responses]

        # We find the union of all numerical data
        all_data = set()
        for data_set in extracted_data_per_run:
            all_data.update(data_set)

        # If a run has completely disjoint numerical data from the consensus, that's an inconsistency
        inconsistencies = 0
        if all_data:
            # Check how many data points are shared vs unique per run
            for data_set in extracted_data_per_run:
                # If a run misses more than 50% of the aggregate data points, or introduces completely new ones
                # It's a rough heuristic for technical drift.
                drift = len(all_data - data_set)
                if drift > len(all_data) * 0.7:  # Highly divergent numericals
                    inconsistencies += 1

        return {
            "cosine_similarity_score": avg_similarity,
            "numerical_inconsistencies": inconsistencies,
            "overall_consistency_score": (avg_similarity * 100) - (inconsistencies * 5) # Scale to 100
        }

    # --- E. Reasoning Quality Metrics ---

    def evaluate_reasoning_quality(self, text: str) -> Dict[str, Any]:
        """
        Heuristic-based reasoning quality assessment.
        Looks for chain-of-thought markers and structural completeness.
        """
        lower_text = text.lower()

        # Chain of thought markers
        cot_markers = ["first", "second", "therefore", "because", "resulting in", "step", "then", "consequently"]
        cot_score = sum(1 for marker in cot_markers if marker in lower_text)

        # Coverage of key concepts (generic check if it's not a single short sentence)
        is_comprehensive = len(text.split()) > 50 and "." in text

        return {
            "cot_markers_found": cot_score,
            "is_comprehensive": is_comprehensive,
            "reasoning_score": min(100, cot_score * 10 + (20 if is_comprehensive else 0))
        }

if __name__ == "__main__":
    engine = MetricsEngine()
    print("Testing length metrics:", engine.calculate_token_count("Hello world this is a test."))

    r1 = "The battery is 400V and the current is 1.5A. First, check the fuse. Therefore it is broken."
    r2 = "The battery is 800V. First, inspect the wiring."
    print("Testing consistency:", engine.evaluate_consistency([r1, r2, r1, r1, r1]))
