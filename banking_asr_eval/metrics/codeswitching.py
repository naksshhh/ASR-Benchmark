"""
Code-switching metrics — Layer 2.

Measures WER separately on Hindi-only, English-only, and mixed segments.
Mixed segments will always be worst. Quantifying how much worse tells you
whether you need a multilingual model or separate models with language detection.
"""

import re
from typing import Dict, List, Optional, Tuple

import jiwer

from .normalize import normalize


# Unicode ranges for script detection
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
LATIN_RANGE = re.compile(r"[a-zA-Z]")


def detect_word_language(word: str) -> str:
    """
    Detect language of a single word based on script.

    Returns: 'hindi', 'english', 'numeric', or 'unknown'
    """
    has_devanagari = bool(DEVANAGARI_RANGE.search(word))
    has_latin = bool(LATIN_RANGE.search(word))
    has_digit = bool(re.search(r"\d", word))

    if has_devanagari and not has_latin:
        return "hindi"
    elif has_latin and not has_devanagari:
        return "english"
    elif has_digit and not has_devanagari and not has_latin:
        return "numeric"
    elif has_devanagari and has_latin:
        return "mixed"
    else:
        return "unknown"


def classify_segment_language(text: str) -> str:
    """
    Classify the dominant language of a text segment.

    Returns: 'hindi', 'english', 'mixed', or 'numeric'
    """
    words = text.split()
    if not words:
        return "unknown"

    lang_counts = {"hindi": 0, "english": 0, "numeric": 0, "mixed": 0, "unknown": 0}
    for word in words:
        lang = detect_word_language(word)
        lang_counts[lang] += 1

    content_words = lang_counts["hindi"] + lang_counts["english"] + lang_counts["mixed"]
    if content_words == 0:
        return "numeric"

    hindi_ratio = lang_counts["hindi"] / content_words if content_words > 0 else 0
    english_ratio = lang_counts["english"] / content_words if content_words > 0 else 0

    if hindi_ratio > 0.8:
        return "hindi"
    elif english_ratio > 0.8:
        return "english"
    else:
        return "mixed"


def codeswitching_wer(
    reference: str,
    hypothesis: str,
    language_segments: Optional[List[Dict]] = None,
) -> Dict:
    """
    Compute per-language WER for code-switched text.

    If language_segments is provided (from manifest annotation), uses those.
    Otherwise, auto-detects language per word.

    Returns:
        Dict with overall_wer, hindi_wer, english_wer, mixed_wer,
        and word counts per language.
    """
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    # Overall WER
    overall_wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm.strip() else 0.0

    # Per-word language classification on reference
    ref_words = ref_norm.split()
    hindi_words = []
    english_words = []
    mixed_words = []

    for word in ref_words:
        lang = detect_word_language(word)
        if lang == "hindi":
            hindi_words.append(word)
        elif lang == "english":
            english_words.append(word)
        else:
            mixed_words.append(word)

    # Compute per-language WER by filtering hypothesis similarly
    hyp_words = hyp_norm.split()
    hindi_hyp = [w for w in hyp_words if detect_word_language(w) == "hindi"]
    english_hyp = [w for w in hyp_words if detect_word_language(w) == "english"]

    result = {
        "overall_wer": overall_wer,
        "dominant_language": classify_segment_language(ref_norm),
        "hindi_word_count": len(hindi_words),
        "english_word_count": len(english_words),
        "mixed_word_count": len(mixed_words),
    }

    # Per-language WER (only if enough words)
    if hindi_words:
        ref_hi = " ".join(hindi_words)
        hyp_hi = " ".join(hindi_hyp) if hindi_hyp else ""
        result["hindi_wer"] = jiwer.wer(ref_hi, hyp_hi)
    else:
        result["hindi_wer"] = None

    if english_words:
        ref_en = " ".join(english_words)
        hyp_en = " ".join(english_hyp) if english_hyp else ""
        result["english_wer"] = jiwer.wer(ref_en, hyp_en)
    else:
        result["english_wer"] = None

    return result
