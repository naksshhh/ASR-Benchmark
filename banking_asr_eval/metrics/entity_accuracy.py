"""
Named Entity Accuracy — Layer 2, banking-critical.

Measures accuracy on banking-specific entities: bank names, product names,
identity document references, and financial terms that ASR frequently mangles.
"""

import re
from typing import Dict, List, Set

import jiwer

from .normalize import normalize


# Banking-domain entities that must be recognized correctly
BANKING_ENTITIES: Dict[str, Set[str]] = {
    "BANK": {
        "sbi", "hdfc", "icici", "axis", "kotak", "pnb", "bob", "canara",
        "union bank", "idbi", "yes bank", "indusind", "bandhan", "rbl",
        "federal bank", "south indian bank", "karur vysya",
    },
    "IDENTITY": {
        "aadhar", "aadhaar", "pan", "pan card", "voter id", "passport",
        "driving license", "ration card",
    },
    "PRODUCT": {
        "emi", "fd", "rd", "sip", "mutual fund", "credit card", "debit card",
        "home loan", "personal loan", "car loan", "education loan",
        "savings account", "current account", "ppf", "nps",
    },
    "SYSTEM": {
        "upi", "neft", "rtgs", "imps", "nach", "cibil", "ifsc",
        "otp", "cvv", "pin", "atm", "pos", "ecs",
    },
}

# Flatten for quick lookup
ALL_ENTITIES = set()
for entities in BANKING_ENTITIES.values():
    ALL_ENTITIES.update(entities)


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract banking entities from text with their types (word-boundary matching)."""
    text_lower = text.lower()
    found = []
    for entity_type, entities in BANKING_ENTITIES.items():
        for entity in sorted(entities, key=len, reverse=True):
            # Use word boundaries to avoid "rd" matching inside "card"
            pattern = r"\b" + re.escape(entity) + r"\b"
            if re.search(pattern, text_lower):
                found.append({"text": entity, "type": entity_type})
    return found


def entity_accuracy(reference: str, hypothesis: str) -> Dict:
    """
    Compute entity-level accuracy between reference and hypothesis.

    Returns:
        Dict with overall accuracy and per-type breakdown
    """
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    ref_entities = extract_entities(ref_norm)
    hyp_entities = extract_entities(hyp_norm)

    if not ref_entities:
        return {"entity_accuracy": 1.0, "ref_count": 0, "matched": 0,
                "missed": [], "spurious": [], "has_entities": False}

    ref_set = {e["text"] for e in ref_entities}
    hyp_set = {e["text"] for e in hyp_entities}

    matched = ref_set & hyp_set
    missed = ref_set - hyp_set
    spurious = hyp_set - ref_set

    accuracy = len(matched) / len(ref_set) if ref_set else 1.0

    return {
        "entity_accuracy": accuracy,
        "ref_count": len(ref_set),
        "matched": list(matched),
        "missed": list(missed),
        "spurious": list(spurious),
        "has_entities": True,
    }
