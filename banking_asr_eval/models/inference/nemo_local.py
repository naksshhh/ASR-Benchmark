"""
NeMo inference wrapper (Param Rudra only).

Handles Parakeet, Canary, IndicConformer and other NeMo-based models.
Requires nemo_toolkit[asr] — not installed on Mac Mini.

Note: IndicConformer requires AI4Bharat's fork of NeMo:
  git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo && git checkout nemo-v2 && bash reinstall.sh
"""

import os
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

    def _ensure_wav_16k(audio_path: str) -> str:
        """
        IndicConformer requires 16kHz mono WAV input.
        If the input is not WAV or not 16kHz, convert it in-place to a temp file.
        """
        import soundfile as sf
        try:
            info = sf.info(audio_path)
            if info.samplerate == 16000 and info.channels == 1:
                return audio_path
        except Exception:
            pass

        # Need to convert
        import tempfile
        import numpy as np
        try:
            import librosa
            audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception:
            audio_array, sr = sf.read(audio_path, dtype="float32")
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_array, 16000)
        return tmp.name

    def _extract_text(val) -> str:
        """
        Robustly extract transcript text from NeMo transcribe output.
        NeMo models can return:
        - A list of strings: ['transcript']
        - A list of Hypothesis objects with .text attribute
        - A list of lists (batch dimension)
        - A tuple of (texts, other_info)
        - A dict with 'text' key
        """
        if val is None:
            return ""

        # If it's a string, return directly
        if isinstance(val, str):
            return val.strip()

        # If it's a dict, try 'text' key
        if isinstance(val, dict):
            if "text" in val:
                return str(val["text"]).strip()
            return ""

        # If it has a .text attribute (Hypothesis object)
        if hasattr(val, "text"):
            t = val.text
            if isinstance(t, str):
                return t.strip()
            return str(t).strip() if t is not None else ""

        # If it's a tuple, take the first element
        if isinstance(val, tuple):
            return _extract_text(val[0])

        # If it's a list
        if isinstance(val, (list,)):
            if not val:
                return ""
            first = val[0]
            return _extract_text(first)

        # Fallback: stringify
        s = str(val)
        if s in ("nan", "None", ""):
            return ""
        return s.strip()

    def transcribe(audio_path: str) -> str:
        """Transcribe a single audio file using NeMo."""
        # Ensure proper format
        converted_path = _ensure_wav_16k(audio_path)
        cleanup = converted_path != audio_path

        try:
            if is_indicconformer:
                transcriptions = model.transcribe(
                    [converted_path], batch_size=1, logprobs=False, language_id="hi"
                )
            else:
                transcriptions = model.transcribe([converted_path])

            result = _extract_text(transcriptions)

            # If we got nothing, try RNNT decoder as fallback for hybrid models
            if not result and is_indicconformer and hasattr(model, "cur_decoder"):
                prev = model.cur_decoder
                model.cur_decoder = "rnnt"
                try:
                    transcriptions = model.transcribe(
                        [converted_path], batch_size=1, language_id="hi"
                    )
                    result = _extract_text(transcriptions)
                finally:
                    model.cur_decoder = prev

            return result
        finally:
            if cleanup and os.path.exists(converted_path):
                try:
                    os.remove(converted_path)
                except Exception:
                    pass

    return transcribe


