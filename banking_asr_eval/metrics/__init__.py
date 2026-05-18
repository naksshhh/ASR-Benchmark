"""
Banking ASR Evaluation — Metrics Package.

Exposes a clean API for all metric layers:
- Layer 1: WER, CER, MER, WIL (core transcription)
- Layer 2: NER, Entity Accuracy, Code-switching WER (banking-domain)
- Layer 3: RTF (latency-quality tradeoff, computed in benchmark.py)
"""

from .core import compute_core_metrics, compute_wer, compute_cer, get_word_alignments
from .normalize import normalize, BankingNormalize
from .number_er import number_error_rate, extract_number_tokens, analyze_number_errors
from .entity_accuracy import entity_accuracy, extract_entities
from .codeswitching import codeswitching_wer, classify_segment_language

__all__ = [
    # Layer 1
    "compute_core_metrics",
    "compute_wer",
    "compute_cer",
    "get_word_alignments",
    # Normalization
    "normalize",
    "BankingNormalize",
    # Layer 2
    "number_error_rate",
    "extract_number_tokens",
    "analyze_number_errors",
    "entity_accuracy",
    "extract_entities",
    "codeswitching_wer",
    "classify_segment_language",
]
