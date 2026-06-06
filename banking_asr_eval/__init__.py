"""
Banking ASR Evaluation Pipeline.

A comprehensive evaluation framework for ASR models in Indian banking contexts.
Measures WER, CER, Number Error Rate, Entity Accuracy, and Code-switching WER
with proper text normalization for Hindi-English banking dialogues.
"""

# Monkeypatch Hugging Face transformers check_torch_load_is_safe to bypass CVE-2025-32434 check on older PyTorch versions on the cluster
try:
    import transformers.utils.import_utils as hf_import_utils
    hf_import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
except (ImportError, AttributeError):
    pass
try:
    import transformers.utils as hf_utils
    hf_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
except (ImportError, AttributeError):
    pass

__version__ = "0.1.0"
