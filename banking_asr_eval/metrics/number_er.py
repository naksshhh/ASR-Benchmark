"""
Number Error Rate (NER) — Layer 2, banking-critical.

Extracts numeric tokens from reference and hypothesis,
then computes WER only over those tokens.
"""

import re
from typing import Dict, List

import jiwer

from .normalize import normalize

# Hindi number words (Devanagari)
HINDI_NUMBER_WORDS = {
    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "पाँच", "छह", "छः",
    "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह",
    "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    "सौ", "हज़ार", "हजार", "लाख", "करोड़",
}

# Transliterated Hindi number words
TRANSLITERATED_NUMBER_WORDS = {
    "ek", "do", "teen", "char", "paanch", "chhe", "saat", "aath",
    "nau", "das", "gyarah", "barah", "sau", "hazaar", "lakh", "crore",
}

# English number words
ENGLISH_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "hundred", "thousand", "lakh", "million", "crore",
}

ALL_NUMBER_WORDS = HINDI_NUMBER_WORDS | TRANSLITERATED_NUMBER_WORDS | ENGLISH_NUMBER_WORDS
DIGIT_PATTERN = re.compile(r"\b\d[\d,\.]*\b")


def extract_number_tokens(text: str) -> List[str]:
    """Extract all numeric tokens (digits and number words) from text."""
    tokens = []
    for word in text.lower().split():
        if DIGIT_PATTERN.match(word):
            tokens.append(word.replace(",", ""))
        elif word in ALL_NUMBER_WORDS:
            tokens.append(word)
    return tokens


def number_error_rate(reference: str, hypothesis: str, pre_normalize: bool = True) -> Dict:
    """Compute NER: WER restricted to numeric tokens only."""
    if pre_normalize:
        reference = normalize(reference, do_normalize_numbers=False)
        hypothesis = normalize(hypothesis, do_normalize_numbers=False)

    ref_numbers = extract_number_tokens(reference)
    hyp_numbers = extract_number_tokens(hypothesis)

    if not ref_numbers:
        return {"ner": 0.0, "ref_count": 0, "hyp_count": len(hyp_numbers),
                "ref_numbers": ref_numbers, "hyp_numbers": hyp_numbers, "has_numbers": False}

    ref_str = " ".join(ref_numbers)
    hyp_str = " ".join(hyp_numbers) if hyp_numbers else ""
    ner = jiwer.wer(ref_str, hyp_str)

    return {"ner": ner, "ref_count": len(ref_numbers), "hyp_count": len(hyp_numbers),
            "ref_numbers": ref_numbers, "hyp_numbers": hyp_numbers, "has_numbers": True}


def analyze_number_errors(reference: str, hypothesis: str) -> List[Dict]:
    """Detailed analysis: which numbers were wrong and how."""
    ref_norm = normalize(reference, do_normalize_numbers=False)
    hyp_norm = normalize(hypothesis, do_normalize_numbers=False)
    ref_numbers = extract_number_tokens(ref_norm)
    hyp_numbers = extract_number_tokens(hyp_norm)

    if not ref_numbers:
        return []

    output = jiwer.process_words(" ".join(ref_numbers), " ".join(hyp_numbers) if hyp_numbers else "")
    errors = []
    for chunk in output.alignments[0]:
        if chunk.type != "equal":
            errors.append({
                "error_type": chunk.type,
                "reference": output.references[0][chunk.ref_start_idx:chunk.ref_end_idx],
                "hypothesis": output.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx],
            })
    return errors
