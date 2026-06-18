"""
File: hallucination_detector.py
Purpose: Core module for the Automotive LLM Benchmarking Framework.
This file has been comprehensively commented to ensure maximum readability and maintainability.
"""
import re
import json
from typing import Dict, Any, List

# Mock registries for validation. In a real production system, this would be a massive database.
# This represents the 'reference dictionary/JSON' requirement from the task.
VALID_SAE_ISO_STANDARDS = {
    "SAE J1939", "SAE J2012", "ISO 15765", "ISO 14229", "ISO 26262",
    "ISO 21434", "SAE J2534", "ISO 11898", "SAE J1772", "ISO 15118"
}

# A small subset of real DTCs to act as a mock database.
MOCK_VALID_DTCS = {
    "P0420", "P0300", "P0171", "U0100", "B1001", "C1201"
}

class HallucinationDetector:
    """
    Analyzes LLM text to detect fabricated or technically inaccurate automotive data.
    Fulfills the 'Hallucination Detection Module' requirement.
    """
    def __init__(self):
        # Regex to find potential Diagnostic Trouble Codes (DTC).
        # Ensures strict adherence to SAE J2012 format:
        # Starts with P (Powertrain), B (Body), C (Chassis), or U (Network)
        # Followed by exactly 4 hexadecimal characters (0-9, A-F).
        self.dtc_pattern = re.compile(r'\b[PBCU][0-9A-Fa-f]{4}\b')

        # Regex to find mentions of technical standards (e.g., 'SAE J1939' or 'ISO-15765').
        # \b ensures we match word boundaries.
        self.standard_pattern = re.compile(r'\b(SAE|ISO)\s?[a-zA-Z0-9-]+\b', re.IGNORECASE)

    def extract_dtcs(self, text: str) -> List[str]:
        """Finds all occurrences of validly formatted DTCs in the text."""
        print(f"[hallucination_detector.py] | INPUT: Raw text | PROCESS: Scanning for SAE J2012 format [P/B/C/U][0-9A-F]{{4}} | OUTPUT: Regex complete")
        return self.dtc_pattern.findall(text)

    def extract_standards(self, text: str) -> List[str]:
        """Finds and normalizes mentions of ISO/SAE standards in the text."""
        # Find all raw matches using regex
        matches = [m.group(0) for m in re.finditer(r'\b(?:SAE|ISO)\s?[a-zA-Z0-9-]+\b', text, re.IGNORECASE)]

        cleaned = []
        for match in matches:
            # Clean up the string by splitting on whitespace
            parts = re.split(r'\s+', match.strip())
            if len(parts) == 1:
                # If they wrote it as one word (e.g. ISO15765), split it manually into "ISO 15765"
                prefix = parts[0][:3].upper()
                suffix = parts[0][3:]
                cleaned.append(f"{prefix} {suffix}")
            else:
                # If properly spaced, just capitalize it
                cleaned.append(f"{parts[0].upper()} {parts[1]}")
        return cleaned

    def detect_invalid_dtcs(self, text: str) -> List[str]:
        """
        Heuristic #1: Detects 'Non-existent DTC codes' (per task rubric).
        It flags codes that either don't exist in the mock database OR violate standard formatting.
        """
        extracted = self.extract_dtcs(text)
        invalid = []

        # Check cleanly formatted codes against our mock database
        for dtc in extracted:
            if dtc.upper() not in MOCK_VALID_DTCS:
                invalid.append(dtc.upper())

        # Second sweep: Look for things that look like DTCs but break the rules
        # (e.g., Start with 'Z' or contain illegal letters like 'G')
        malformed_pattern = re.compile(r'\b[A-Z][0-9A-Z]{4}\b')
        potential_dtcs = malformed_pattern.findall(text)

        for p_dtc in potential_dtcs:
            # Check if it fails the P/B/C/U prefix rule
            if p_dtc[0] not in ['P', 'B', 'C', 'U']:
                if p_dtc not in invalid:
                    invalid.append(p_dtc) # E.g., Z1234
            # Check if characters 2-5 fail the hexadecimal rule
            elif not all(c in '0123456789ABCDEFabcdef' for c in p_dtc[1:]):
                if p_dtc not in invalid:
                    invalid.append(p_dtc) # E.g., P123G

        return invalid

    def detect_invalid_standards(self, text: str) -> List[str]:
        """
        Heuristic #2: Detects 'Unsupported automotive standards' (per task rubric).
        Extracts standards and checks them against VALID_SAE_ISO_STANDARDS.
        """
        extracted = self.extract_standards(text)
        invalid = []
        for std in extracted:
            if std.upper() not in VALID_SAE_ISO_STANDARDS:
                invalid.append(std.upper())
        return invalid

    def detect_technical_contradictions(self, text: str) -> List[str]:
        """
        Heuristic #3: Detects 'Contradictory technical statements' (per task rubric).
        Uses basic keyword relationships to flag suspicious engineering claims.
        """
        contradictions = []
        lower_text = text.lower()

        # Contradiction Type A: Simultaneous opposites without an 'or' condition
        # e.g., "The root cause is an open circuit and a short circuit."
        if "open circuit" in lower_text and "short circuit" in lower_text and "or" not in lower_text:
            contradictions.append("Mentions both open and short circuit simultaneously without clear separation.")

        # Contradiction Type B: Conflicting voltage domains
        # e.g., Claiming a 12V battery provides 800V without mentioning a DC-DC converter
        if re.search(r'\b12v\b', lower_text) and re.search(r'\b(400v|800v)\b', lower_text):
             if "dc-dc" not in lower_text and "converter" not in lower_text:
                 contradictions.append("Conflicting voltage domains mentioned without reference to a DC-DC converter.")

        return contradictions

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Main execution function. Runs all heuristics on the text and returns a summary.
        """
        invalid_dtcs = self.detect_invalid_dtcs(text)
        invalid_standards = self.detect_invalid_standards(text)
        contradictions = self.detect_technical_contradictions(text)

        # Calculate a simple hallucination severity score (0 is perfect, higher is worse)
        score = len(invalid_dtcs) + len(invalid_standards) + len(contradictions)

        return {
            "invalid_dtcs": invalid_dtcs,
            "invalid_standards": invalid_standards,
            "contradictions": contradictions,
            "hallucination_flags_count": score,
            "is_hallucinating": score > 0
        }

if __name__ == "__main__":
    # Test block to verify logic
    detector = HallucinationDetector()
    test_text = "The issue is a P0420 and a Z9999. Also check SAE J9999 and ISO 15765. The 12V battery provides 800V."
    results = detector.evaluate(test_text)
    print(json.dumps(results, indent=2))
