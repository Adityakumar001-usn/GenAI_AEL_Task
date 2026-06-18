import re
import math
import tiktoken
from typing import List, Dict, Any
from collections import Counter

class MetricsEngine:
    """
    Calculates quantitative length, latency, consistency, and reasoning metrics.
    Fulfills 'Evaluation Metrics Engine' requirement.
    """
    def __init__(self):
        # Initialize tiktoken for general token counting when providers do not supply exact counts.
        # "cl100k_base" is a standard OpenAI tokenizer.
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # --- A. Response Length Metrics ---

    def calculate_word_count(self, text: str) -> int:
        """Counts total words by splitting on whitespace."""
        return len(text.split())

    def calculate_token_count(self, text: str) -> int:
        """Counts tokens by encoding the text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def calculate_compression_ratio(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Ratio of output size to input size.
        A higher ratio indicates the model elaborated extensively on a short prompt.
        """
        if prompt_tokens == 0:
            return 0.0
        return completion_tokens / prompt_tokens

    def calculate_information_density(self, text: str) -> float:
        """
        A heuristic for how dense the information is.
        Calculated as the ratio of *unique* significant words (length > 3) to total significant words.
        Filters out common stop words (like 'the', 'a', 'is') by length.
        """
        # Find all words, make lowercase, keep only those longer than 3 characters
        words = [w.lower() for w in re.findall(r'\b\w+\b', text) if len(w) > 3]
        if not words:
            return 0.0
        # Determine how many are unique
        unique_words = set(words)
        return len(unique_words) / len(words)

    # --- B. Latency Metrics ---

    def calculate_tps(self, tokens: int, latency_ms: float) -> float:
        """
        Calculates Tokens Per Second (TPS).
        Formula: (Total Tokens Generated) / (Total Latency in Seconds)
        """
        if latency_ms == 0:
            return 0.0
        return tokens / (latency_ms / 1000.0)

    # --- C. Consistency Metrics ---

    def _cosine_similarity(self, vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        """
        Mathematical function to compute the cosine similarity between two word frequency vectors.
        Returns a float between 0.0 (completely dissimilar) and 1.0 (identical).
        """
        # Find the words common to both vectors
        intersection = set(vec1.keys()) & set(vec2.keys())
        # Calculate dot product (numerator)
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        # Calculate magnitudes (denominator)
        sum1 = sum([vec1[x]**2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x]**2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        return float(numerator) / denominator

    def _text_to_vector(self, text: str) -> Dict[str, int]:
        """Converts text into a dictionary of word frequencies (TF mapping)."""
        words = re.findall(r'\w+', text.lower())
        return Counter(words)

    def extract_numerical_data(self, text: str) -> set:
        """
        Uses regex to find numbers attached to units (e.g., "400V", "1.5 Amps", "95%").
        This is critical for tracking if technical specifications drift between runs.
        """
        # Regex explained: \d+ (numbers) optionally followed by a decimal \.\d+, optionally a space, then alphabets/percent
        pattern = re.compile(r'\b(\d+(?:\.\d+)?\s?[a-zA-Z%]+)\b')
        return set(pattern.findall(text))

    def evaluate_consistency(self, responses: List[str]) -> Dict[str, Any]:
        """
        Calculates consistency across the 5 iterations of a single prompt.
        Combines structural similarity (Cosine) with factual consistency (Numerical Drift).
        Fulfills 'Consistency Evaluation (15%)' requirement.
        """
        if len(responses) < 2:
            return {"cosine_similarity_score": 1.0, "numerical_inconsistencies": 0, "overall_consistency_score": 100.0}

        # 1. Cosine Similarity Analysis
        # Convert all responses to frequency vectors
        vectors = [self._text_to_vector(r) for r in responses]
        similarities = []
        # Calculate pairwise similarity for every combination of runs
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarities.append(self._cosine_similarity(vectors[i], vectors[j]))

        # Average the similarity scores
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # 2. Numerical Consistency Analysis (Strict SME requirement)
        # Extract specs from every run
        extracted_data_per_run = [self.extract_numerical_data(r) for r in responses]

        # Create a "master pool" of all numerical data mentioned across all runs
        all_data = set()
        for data_set in extracted_data_per_run:
            all_data.update(data_set)

        inconsistencies = 0
        if all_data:
            # Check each run against the master pool
            for data_set in extracted_data_per_run:
                # Calculate 'drift' - how many data points are missing from this specific run
                drift = len(all_data - data_set)
                # If a single run misses more than 70% of the aggregate data points,
                # it means the model is outputting wildly different technical specs each time.
                if drift > len(all_data) * 0.7:
                    inconsistencies += 1

        # Calculate a final score out of 100
        # We start with the cosine similarity (e.g., 0.85 -> 85%) and penalize for numerical contradictions
        final_score = (avg_similarity * 100) - (inconsistencies * 5)

        return {
            "cosine_similarity_score": avg_similarity,
            "numerical_inconsistencies": inconsistencies,
            "overall_consistency_score": final_score
        }

    # --- E. Reasoning Quality Metrics ---

    def evaluate_reasoning_quality(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the response for logical 'Chain-of-Thought' structures.
        Fulfills the 'Reasoning Quality Metrics' requirement.
        """
        lower_text = text.lower()

        # Look for explicit markers that indicate step-by-step reasoning
        cot_markers = ["first", "second", "therefore", "because", "resulting in", "step", "then", "consequently"]
        cot_score = sum(1 for marker in cot_markers if marker in lower_text)

        # Basic check to ensure the model didn't just output a single, terse sentence
        is_comprehensive = len(text.split()) > 50 and "." in text

        # Calculate a reasoning score out of 100
        reasoning_score = min(100, cot_score * 10 + (20 if is_comprehensive else 0))

        return {
            "cot_markers_found": cot_score,
            "is_comprehensive": is_comprehensive,
            "reasoning_score": reasoning_score
        }

if __name__ == "__main__":
    engine = MetricsEngine()
    print("Testing length metrics:", engine.calculate_token_count("Hello world this is a test."))
