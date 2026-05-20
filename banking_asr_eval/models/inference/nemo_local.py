"""
NeMo inference wrapper (Param Rudra only).

Handles Parakeet, Canary, IndicConformer and other NeMo-based models.
Requires nemo_toolkit[asr] — not installed on Mac Mini.

Note: IndicConformer requires AI4Bharat's fork of NeMo:
  git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo && git checkout nemo-v2 && bash reinstall.sh
"""

import torch
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
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.freeze()  # inference mode
    model = model.to(device)
    model.eval()

    # Detect if this is an IndicConformer hybrid model
    is_indicconformer = "indicconformer" in model_id.lower()
    if is_indicconformer and hasattr(model, "cur_decoder"):
        model.cur_decoder = "ctc"
        print(f"[NeMo] Set cur_decoder='ctc' for IndicConformer")

    def transcribe(audio_path: str) -> str:
        """Transcribe a single audio file using NeMo."""
        # IndicConformer requires language_id='hi'
        if is_indicconformer:
            transcriptions = model.transcribe(
                [audio_path], batch_size=1, logprobs=False, language_id="hi"
            )
        else:
            transcriptions = model.transcribe([audio_path])

        # Helper to extract text from whatever NeMo returns
        def extract_text(val):
            if isinstance(val, (list, tuple)):
                if not val:
                    return ""
                return extract_text(val[0])
            if hasattr(val, "text"):
                return str(val.text)
            return str(val)

        return extract_text(transcriptions).strip()

    return transcribe

