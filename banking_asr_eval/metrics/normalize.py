"""
Text normalization for ASR evaluation.

This is THE most critical module in the pipeline. Two papers reporting WER on
the same dataset with different normalization produce incomparable numbers.

Handles:
- Hindi (Devanagari) text normalization
- English text normalization
- Number expansion (both languages)
- Currency normalization (₹, Rs, rupees → standard form)
- Banking-specific abbreviations (EMI, KYC, CIBIL, NACH, etc.)
- Code-switched text (Hindi-English mix)
"""

import re
import unicodedata
from typing import Optional

# ─── Hindi number words ──────────────────────────────────────────────────────
HINDI_NUMBER_WORDS = {
    "शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4",
    "पांच": "5", "पाँच": "5", "छह": "6", "छः": "6", "सात": "7",
    "आठ": "8", "नौ": "9", "दस": "10", "ग्यारह": "11", "बारह": "12",
    "तेरह": "13", "चौदह": "14", "पंद्रह": "15", "सोलह": "16",
    "सत्रह": "17", "अठारह": "18", "उन्नीस": "19", "बीस": "20",
    "तीस": "30", "चालीस": "40", "पचास": "50", "साठ": "60",
    "सत्तर": "70", "अस्सी": "80", "नब्बे": "90",
    "सौ": "100", "हज़ार": "1000", "हजार": "1000",
    "लाख": "100000", "करोड़": "10000000",
}

# Transliterated Hindi number words (as they appear in ASR output)
TRANSLITERATED_HINDI_NUMBERS = {
    "ek": "1", "do": "2", "teen": "3", "char": "4", "paanch": "5",
    "chhe": "6", "saat": "7", "aath": "8", "nau": "9", "das": "10",
    "gyarah": "11", "barah": "12", "tera": "13", "chaudah": "14",
    "pandrah": "15", "solah": "16", "satrah": "17", "atharah": "18",
    "unnees": "19", "bees": "20", "tees": "30", "chaalees": "40",
    "pachaas": "50", "saath": "60", "sattar": "70", "assi": "80",
    "nabbe": "90", "sau": "100", "hazaar": "1000", "hazar": "1000",
    "lakh": "100000", "crore": "10000000",
}

# English number words
ENGLISH_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "lakh": "100000", "lac": "100000", "million": "1000000",
    "crore": "10000000",
}

# ─── Banking abbreviations ───────────────────────────────────────────────────
# Normalize common banking terms to canonical forms
BANKING_ABBREVIATIONS = {
    # These map variant spellings/pronunciations to canonical form
    "e.m.i.": "emi", "e m i": "emi", "equated monthly installment": "emi",
    "k.y.c.": "kyc", "k y c": "kyc", "know your customer": "kyc",
    "c.i.b.i.l.": "cibil", "c i b i l": "cibil",
    "n.a.c.h.": "nach", "n a c h": "nach",
    "i.f.s.c.": "ifsc", "i f s c": "ifsc",
    "u.p.i.": "upi", "u p i": "upi",
    "a.t.m.": "atm", "a t m": "atm",
    "p.a.n.": "pan", "p a n": "pan",
    "g.s.t.": "gst", "g s t": "gst",
    "o.t.p.": "otp", "o t p": "otp",
    "n.e.f.t.": "neft", "n e f t": "neft",
    "r.t.g.s.": "rtgs", "r t g s": "rtgs",
    "i.m.p.s.": "imps", "i m p s": "imps",
    "f.d.": "fd", "f d": "fd", "fixed deposit": "fd",
    "r.d.": "rd", "r d": "rd", "recurring deposit": "rd",
    "aadhaar": "aadhar", "aadhar": "aadhar", "aadhaar card": "aadhar",
}

# ─── Currency patterns ───────────────────────────────────────────────────────
CURRENCY_PATTERNS = [
    (re.compile(r"₹\s*"), "rupees "),
    (re.compile(r"rs\.?\s*", re.IGNORECASE), "rupees "),
    (re.compile(r"rupaya\s*", re.IGNORECASE), "rupees "),
    (re.compile(r"rupaye\s*", re.IGNORECASE), "rupees "),
    (re.compile(r"rupaiye\s*", re.IGNORECASE), "rupees "),
]

# ─── Punctuation to remove ───────────────────────────────────────────────────
PUNCTUATION_PATTERN = re.compile(r"[।,\.\?\!;:\-\—\–\"\'\(\)\[\]\{\}\/\\…\"\"\'\'«»]")

# Devanagari punctuation
DEVANAGARI_PUNCTUATION = re.compile(r"[।॥,]")


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form (important for Devanagari)."""
    return unicodedata.normalize("NFC", text)


def lowercase(text: str) -> str:
    """Lowercase text. Safe for Devanagari (no case)."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove punctuation marks (both English and Devanagari)."""
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = DEVANAGARI_PUNCTUATION.sub(" ", text)
    return text


def normalize_currency(text: str) -> str:
    """Normalize currency symbols and words to 'rupees'."""
    for pattern, replacement in CURRENCY_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def normalize_banking_terms(text: str) -> str:
    """Normalize banking abbreviations to canonical forms."""
    # Sort by length descending so longer patterns match first
    for variant, canonical in sorted(
        BANKING_ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        text = text.replace(variant, canonical)
    return text


def normalize_numbers_in_text(text: str) -> str:
    """
    Normalize written number words to digits.

    This is a simplified version — handles isolated number words.
    Does NOT handle compound numbers like "twenty five thousand"
    (that requires a full number parser, added in v2).
    """
    words = text.split()
    result = []
    for word in words:
        # Check all number word dictionaries
        normalized = (
            HINDI_NUMBER_WORDS.get(word)
            or TRANSLITERATED_HINDI_NUMBERS.get(word)
            or ENGLISH_NUMBER_WORDS.get(word)
        )
        result.append(normalized if normalized else word)
    return " ".join(result)


def remove_filler_words(text: str) -> str:
    """Remove common filler words that ASR may insert/miss."""
    fillers = {
        # English fillers
        "um", "uh", "umm", "uhh", "hmm", "hm", "ah", "oh",
        # Hindi fillers
        "अं", "हां", "हम्म", "उम्म",
    }
    words = text.split()
    return " ".join(w for w in words if w not in fillers)


def collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces into single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def remove_commas_from_numbers(text: str) -> str:
    """Remove commas from numbers: 50,000 → 50000."""
    return re.sub(r"(\d),(\d)", r"\1\2", text)


def normalize(
    text: str,
    do_lowercase: bool = True,
    do_remove_punctuation: bool = True,
    do_normalize_currency: bool = True,
    do_normalize_banking: bool = True,
    do_normalize_numbers: bool = True,
    do_remove_fillers: bool = True,
) -> str:
    """
    Full normalization pipeline for ASR evaluation.

    Order matters:
    1. Unicode normalization (before anything else)
    2. Lowercase
    3. Currency normalization (before punctuation removal, since ₹ is punctuation)
    4. Remove commas from numbers (before punctuation removal eats them)
    5. Remove punctuation
    6. Banking term normalization
    7. Number word normalization
    8. Filler word removal
    9. Whitespace collapse

    Args:
        text: Raw transcript text
        do_lowercase: Whether to lowercase
        do_remove_punctuation: Whether to strip punctuation
        do_normalize_currency: Whether to normalize currency symbols
        do_normalize_banking: Whether to normalize banking terms
        do_normalize_numbers: Whether to expand number words to digits
        do_remove_fillers: Whether to remove filler words

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    text = normalize_unicode(text)

    if do_lowercase:
        text = lowercase(text)

    if do_normalize_currency:
        text = normalize_currency(text)

    text = remove_commas_from_numbers(text)

    if do_remove_punctuation:
        text = remove_punctuation(text)

    if do_normalize_banking:
        text = normalize_banking_terms(text)

    if do_normalize_numbers:
        text = normalize_numbers_in_text(text)

    if do_remove_fillers:
        text = remove_filler_words(text)

    text = collapse_whitespace(text)
    return text


# ─── Convenience: jiwer-compatible transform ─────────────────────────────────

class BankingNormalize:
    """
    A callable class compatible with jiwer's transform pipeline.

    Usage:
        transforms = jiwer.Compose([BankingNormalize()])
        measures = jiwer.compute_measures(ref, hyp,
            truth_transform=transforms, hypothesis_transform=transforms)
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, sentences):
        if isinstance(sentences, str):
            return normalize(sentences, **self.kwargs)
        return [normalize(s, **self.kwargs) for s in sentences]
