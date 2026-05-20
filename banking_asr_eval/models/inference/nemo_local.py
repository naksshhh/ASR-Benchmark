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

def _load_nemo_manually(nemo_asr, model_id: str):
    """
    Fix NeMo 2.x extraction issue and reload.

    NeMo downloads the .nemo archive into a cache directory but doesn't
    always extract model_config.yaml into it. This function:
    1. Finds the .nemo file in the NeMo cache
    2. Extracts it INTO the same cache directory NeMo expects
    3. Retries from_pretrained (which now finds model_config.yaml)
    """
    import tarfile
    import glob
    import shutil

    # Step 1: Find the .nemo file in the NeMo cache
    nemo_path = None
    nemo_cache = os.path.expanduser("~/.cache/torch/NeMo")
    candidates = glob.glob(
        os.path.join(nemo_cache, "**", "*.nemo"), recursive=True
    )
    model_short = model_id.split("/")[-1]
    matching = [c for c in candidates if model_short in c]
    if matching:
        nemo_path = matching[0]

    if not nemo_path:
        # Try downloading via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            nemo_path = hf_hub_download(
                repo_id=model_id,
                filename=model_short + ".nemo",
            )
        except Exception:
            pass

    if not nemo_path:
        raise FileNotFoundError(
            f"Could not find .nemo file for {model_id}. "
            f"Try: huggingface-cli download {model_id}"
        )

    print(f"[NeMo] Found .nemo archive: {nemo_path}")
    cache_dir = os.path.dirname(nemo_path)

    # Step 2: Extract the .nemo archive into the SAME cache directory
    # This places model_config.yaml where NeMo expects it
    if tarfile.is_tarfile(nemo_path):
        print(f"[NeMo] Extracting .nemo archive into cache dir: {cache_dir}")
        with tarfile.open(nemo_path, "r:*") as tar:
            tar.extractall(cache_dir)

        # Verify extraction
        config_path = os.path.join(cache_dir, "model_config.yaml")
        if os.path.exists(config_path):
            print(f"[NeMo] Successfully extracted model_config.yaml")
        else:
            # Check if extracted into a subdirectory
            found = glob.glob(os.path.join(cache_dir, "**", "model_config.yaml"), recursive=True)
            if found:
                # Move files up to cache_dir
                subdir = os.path.dirname(found[0])
                for item in os.listdir(subdir):
                    src = os.path.join(subdir, item)
                    dst = os.path.join(cache_dir, item)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                print(f"[NeMo] Moved extracted files from subdirectory to cache dir")
    else:
        print(f"[NeMo] .nemo file is not a tar archive, skipping extraction")

    # Step 3: Retry from_pretrained — NeMo should now find model_config.yaml
    print(f"[NeMo] Retrying from_pretrained after extraction...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    return model


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

    # Load the model — handle NeMo 2.x extraction issues for IndicConformer
    is_indicconformer = "indicconformer" in model_id.lower()

    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    except FileNotFoundError as e:
        if "model_config.yaml" in str(e) and is_indicconformer:
            print(f"[NeMo] from_pretrained failed (missing model_config.yaml). "
                  f"Attempting manual .nemo archive extraction...")
            model = _load_nemo_manually(nemo_asr, model_id)
        else:
            raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.freeze()  # inference mode
    model = model.to(device)
    model.eval()

    # Set decoder for hybrid models
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


