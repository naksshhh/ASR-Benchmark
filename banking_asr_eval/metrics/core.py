"""
Core transcription metrics — Layer 1.

WER, CER, MER, WIL computed via jiwer.
These are your baseline metrics required for comparison against published numbers.

NOTE: We normalize text BEFORE passing to jiwer (not as a jiwer transform).
jiwer v4 transforms must return List[List[str]] which is incompatible with
our banking normalization pipeline. Instead, we pre-normalize and pass
clean strings to jiwer directly.
"""

import jiwer
from typing import Dict, List, Optional, Union

from .normalize import normalize


def compute_core_metrics(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    pre_normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute all Layer 1 metrics in one pass.

    Args:
        reference: Reference transcript(s)
        hypothesis: ASR hypothesis transcript(s)
        pre_normalize: Whether to apply banking normalization before scoring

    Returns:
        Dict with keys: wer, cer, mer, wil, wip, substitutions, deletions,
        insertions, hits
    """
    if pre_normalize:
        if isinstance(reference, str):
            reference = normalize(reference)
            hypothesis = normalize(hypothesis)
        else:
            reference = [normalize(r) for r in reference]
            hypothesis = [normalize(h) for h in hypothesis]

    # Compute word-level metrics
    word_output = jiwer.process_words(reference, hypothesis)

    # Compute character-level metrics
    char_output = jiwer.process_characters(reference, hypothesis)

    return {
        "wer": word_output.wer,
        "mer": word_output.mer,
        "wil": word_output.wil,
        "wip": word_output.wip,
        "cer": char_output.cer,
        "substitutions": word_output.substitutions,
        "deletions": word_output.deletions,
        "insertions": word_output.insertions,
        "hits": word_output.hits,
        # Raw counts for aggregation
        "num_ref_words": sum(
            len(ref) for ref in word_output.references
        ),
        "num_hyp_words": sum(
            len(hyp) for hyp in word_output.hypotheses
        ),
    }


def compute_wer(
    reference: str,
    hypothesis: str,
    pre_normalize: bool = True,
) -> float:
    """Compute Word Error Rate only."""
    if pre_normalize:
        reference = normalize(reference)
        hypothesis = normalize(hypothesis)
    return jiwer.wer(reference, hypothesis)


def compute_cer(
    reference: str,
    hypothesis: str,
    pre_normalize: bool = True,
) -> float:
    """Compute Character Error Rate only."""
    if pre_normalize:
        reference = normalize(reference)
        hypothesis = normalize(hypothesis)
    return jiwer.cer(reference, hypothesis)


def get_word_alignments(
    reference: str,
    hypothesis: str,
    pre_normalize: bool = True,
) -> list:
    """
    Get word-level alignments between reference and hypothesis.

    Returns list of alignment chunks with operation type (hit, substitution,
    deletion, insertion) and the corresponding reference/hypothesis words.
    Useful for error analysis.
    """
    if pre_normalize:
        reference = normalize(reference)
        hypothesis = normalize(hypothesis)

    output = jiwer.process_words(reference, hypothesis)

    alignments = []
    for chunk in output.alignments[0]:
        ref_words = output.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
        hyp_words = output.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]
        alignments.append({
            "type": chunk.type,  # 'equal', 'substitute', 'delete', 'insert'
            "ref_words": ref_words,
            "hyp_words": hyp_words,
        })

    return alignments
