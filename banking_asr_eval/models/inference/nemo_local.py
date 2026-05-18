"""
NeMo inference wrapper (Param Rudra only).

Handles Parakeet, Canary, and other NeMo-based models.
Requires nemo_toolkit[asr] — not installed on Mac Mini.
"""

from typing import Callable


def create_nemo_model(model_id: str) -> Callable[[str], str]:
    """
    Create a NeMo ASR inference function.

    Args:
        model_id: NeMo model identifier (e.g., nvidia/parakeet-tdt-0.6b-v3)

    Returns:
        Callable that takes audio_path and returns transcript string.
    """
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        raise ImportError(
            "NeMo is not installed. Install with: pip install nemo_toolkit[asr]\n"
            "This is only supported on Param Rudra (requires GPU + heavy deps)."
        )

    # Load the model
    if "parakeet" in model_id.lower():
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    elif "canary" in model_id.lower():
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)

    model.eval()

    def transcribe(audio_path: str) -> str:
        """Transcribe a single audio file using NeMo."""
        transcriptions = model.transcribe([audio_path])
        if isinstance(transcriptions, list):
            return transcriptions[0].strip() if transcriptions else ""
        return str(transcriptions).strip()

    return transcribe
