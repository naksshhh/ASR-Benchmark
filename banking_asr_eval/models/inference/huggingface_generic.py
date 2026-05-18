"""
Generic HuggingFace ASR inference wrapper.

For models like AI4Bharat's IndicWav2Vec, IndicConformer, etc.
that follow the standard HuggingFace ASR pipeline interface.
"""

import torch
from transformers import pipeline
from typing import Callable


def create_hf_model(
    model_id: str,
    device: str = None,
) -> Callable[[str], str]:
    """
    Create a HuggingFace ASR inference function.

    Args:
        model_id: HuggingFace model ID
        device: Force device. Auto-detected if None.

    Returns:
        Callable that takes audio_path and returns transcript string.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
    )

    def transcribe(audio_path: str) -> str:
        """Transcribe a single audio file."""
        result = pipe(audio_path)
        return result["text"].strip()

    return transcribe
