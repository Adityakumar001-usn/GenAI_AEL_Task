import json
import re
from typing import Dict, Any, List

# Mock registries for validation
VALID_SAE_ISO_STANDARDS = {
    "SAE J1939", "SAE J2012", "ISO 15765", "ISO 14229", "ISO 26262",
    "ISO 21434", "SAE J2534", "ISO 11898", "SAE J1772", "ISO 15118"
}

# Real DTCs have thousands of variations, but we'll include common prefixes and some specific codes for our mock logic
MOCK_VALID_DTCS = {
    "P0420", "P0300", "P0171", "U0100", "B1001", "C1201"
}

class HallucinationDetector:
    def __init__(self):
        # Regex to find potential DTC codes (e.g., P0420, B1234, C5678, U9012)
        # Standard SAE J2012 formatting: 1 letter (P, B, C, U) + 4 hex digits
        self.dtc_pattern = re.compile(r'\b[PBCU][0-9A-Fa-f]{4}\b')

        # Regex for potential SAE/ISO standard mentions
        self.standard_pattern = re.compile(r'\b(SAE|ISO)\s?[a-zA-Z0-9-]+\b', re.IGNORECASE)

    def extract_dtcs(self, text: str) -> List[str]:
        return self.dtc_pattern.findall(text)

    def extract_standards(self, text: str) -> List[str]:
        # Clean up spacing (e.g., "SAE J1939" instead of "SAE J1939" or "SAEJ1939")
        raw_standards = self.standard_pattern.findall(text)
        # The regex actually extracts the group if we use parentheses incorrectly, let's just find iterative matches
        matches = [m.group(0) for m in re.finditer(r'\b(?:SAE|ISO)\s?[a-zA-Z0-9-]+\b', text, re.IGNORECASE)]

        cleaned = []
        for match in matches:
            # Normalize to "PREFIX SUFFIX"
            parts = re.split(r'\s+', match.strip())
            if len(parts) == 1:
                # E.g. ISO15765 -> ISO 15765
                prefix = parts[0][:3].upper()
                suffix = parts[0][3:]
                cleaned.append(f"{prefix} {suffix}")
            else:
                cleaned.append(f"{parts[0].upper()} {parts[1]}")
        return cleaned

    def detect_invalid_dtcs(self, text: str) -> List[str]:
        """
        Extracts DTCs from text and flags any that do not strictly match the
        mock valid DTC list or violate the fundamental SAE format.
        """
        extracted = self.extract_dtcs(text)
        invalid = []
        for dtc in extracted:
            # Strictly, we check if it is in our mock list to simulate a full database check.
            # However, for a generalized heuristic, any P, B, C, U + 4 hex digits IS technically a valid format.
            # To meet the rubric for "Non-existent DTC codes", we assume anything not in MOCK_VALID_DTCS
            # but matching the format MIGHT be hallucinated, but realistically we also check for malformed ones
            # that slipped through, or we rely on the strict mock list.
            if dtc.upper() not in MOCK_VALID_DTCS:
                invalid.append(dtc.upper())

        # Additionally, let's find things that look like DTCs but violate the format (e.g., Z0420, P123G)
        malformed_pattern = re.compile(r'\b[A-Z][0-9A-Z]{4}\b')
        potential_dtcs = malformed_pattern.findall(text)
        for p_dtc in potential_dtcs:
            if p_dtc[0] not in ['P', 'B', 'C', 'U']:
                # Example: Z1234
                if p_dtc not in invalid:
                    invalid.append(p_dtc)
            elif not all(c in '0123456789ABCDEFabcdef' for c in p_dtc[1:]):
                # Example: P123G
                if p_dtc not in invalid:
                    invalid.append(p_dtc)

        return invalid

    def detect_invalid_standards(self, text: str) -> List[str]:
        """Flags mentioned standards that don't exist in our reference."""
        extracted = self.extract_standards(text)
        invalid = []
        for std in extracted:
            if std.upper() not in VALID_SAE_ISO_STANDARDS:
                invalid.append(std.upper())
        return invalid

    def detect_technical_contradictions(self, text: str) -> List[str]:
        """
        Basic heuristics to detect contradictions in a single text.
        (Advanced logic would use an LLM or complex NLP, but we use simple heuristics for the benchmark).
        """
        contradictions = []
        lower_text = text.lower()

        # Example 1: Claiming both open circuit and short circuit as the root cause without qualification
        if "open circuit" in lower_text and "short circuit" in lower_text and "or" not in lower_text:
            # Very naive check
            contradictions.append("Mentions both open and short circuit simultaneously without clear separation.")

        # Example 2: Voltage contradictions (e.g., saying a 12V system is 400V)
        # This is hard to do perfectly with regex, but we can flag if conflicting voltage classes are stated adjacently
        if re.search(r'\b12v\b', lower_text) and re.search(r'\b(400v|800v)\b', lower_text):
             # Just a flag, might be legitimate in EVs, but for the sake of the heuristic:
             if "dc-dc" not in lower_text and "converter" not in lower_text:
                 contradictions.append("Conflicting voltage domains mentioned without reference to a DC-DC converter.")

        return contradictions

    def evaluate(self, text: str) -> Dict[str, Any]:
        """Runs all heuristics and returns a summary dictionary."""
        invalid_dtcs = self.detect_invalid_dtcs(text)
        invalid_standards = self.detect_invalid_standards(text)
        contradictions = self.detect_technical_contradictions(text)

        # Hallucination Score based on counts (lower is better, 0 is perfect)
        score = len(invalid_dtcs) + len(invalid_standards) + len(contradictions)

        return {
            "invalid_dtcs": invalid_dtcs,
            "invalid_standards": invalid_standards,
            "contradictions": contradictions,
            "hallucination_flags_count": score,
            "is_hallucinating": score > 0
        }

if __name__ == "__main__":
    # Test the module
    detector = HallucinationDetector()
    test_text = "The issue is a P0420 and a Z9999. Also check SAE J9999 and ISO 15765. The 12V battery provides 800V."
    results = detector.evaluate(test_text)
    print(json.dumps(results, indent=2))
