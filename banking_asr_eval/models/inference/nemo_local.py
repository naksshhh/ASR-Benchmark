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
    Fix NeMo 2.x loading for IndicConformer.

    NeMo's from_pretrained fails because:
    1. It doesn't extract model_config.yaml from the .nemo archive
    2. Even with restore_from, the tokenizer.dir config key is missing

    The fix: manually extract the .nemo archive, patch the tokenizer config
    to point to the extracted tokenizer files, then restore_from with the
    patched config.
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

    # Step 2: Copy .nemo to /tmp (NeMo deletes its own cache dir on retry)
    tmp_nemo = os.path.join("/tmp", os.path.basename(nemo_path))
    if not os.path.exists(tmp_nemo):
        print(f"[NeMo] Copying .nemo to {tmp_nemo}...")
        shutil.copy2(nemo_path, tmp_nemo)

    # Step 3: Extract the .nemo archive to inspect and patch config
    extract_dir = os.path.join("/tmp", "nemo_indicconf_extract")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    print(f"[NeMo] Extracting .nemo to {extract_dir}...")
    with tarfile.open(tmp_nemo, "r:*") as tar:
        tar.extractall(extract_dir)

    # List extracted contents for debugging
    extracted_files = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), extract_dir)
            extracted_files.append(rel)
    print(f"[NeMo] Extracted files: {extracted_files}")

    # Step 4: Load config and patch tokenizer.dir
    config_path = os.path.join(extract_dir, "model_config.yaml")
    if not os.path.exists(config_path):
        # Search subdirectories
        found = glob.glob(os.path.join(extract_dir, "**", "model_config.yaml"), recursive=True)
        if found:
            config_path = found[0]

    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)

    # Find tokenizer files (.model files used by SentencePiece)
    tokenizer_files = glob.glob(os.path.join(extract_dir, "**", "*.model"), recursive=True)
    if tokenizer_files:
        tokenizer_dir = os.path.dirname(tokenizer_files[0])
    else:
        # Fallback: use extract_dir itself
        tokenizer_dir = extract_dir

    print(f"[NeMo] Tokenizer files found: {tokenizer_files}")
    print(f"[NeMo] Setting tokenizer.dir = {tokenizer_dir}")

    # Patch the tokenizer config
    if hasattr(config, "tokenizer"):
        config.tokenizer.dir = tokenizer_dir
    else:
        config.tokenizer = OmegaConf.create({"dir": tokenizer_dir})

    # Save patched config
    patched_config_path = os.path.join(extract_dir, "model_config_patched.yaml")
    OmegaConf.save(config, patched_config_path)
    print(f"[NeMo] Saved patched config to {patched_config_path}")

    # Step 5: Load using the specific model class with patched config
    from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
    print(f"[NeMo] Loading EncDecHybridRNNTCTCBPEModel with patched config...")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(
        restore_path=tmp_nemo,
        override_config_path=patched_config_path,
    )
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


