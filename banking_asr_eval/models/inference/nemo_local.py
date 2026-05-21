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
    Fully manual loading for IndicConformer on NeMo 2.x.

    NeMo 2.7.3's save_restore_connector is fundamentally broken for this model:
    - from_pretrained doesn't extract model_config.yaml
    - restore_from ignores override_config_path for tokenizer patching
    - The AI4Bharat config format is incompatible with standard NeMo

    The fix: bypass NeMo's loader entirely. Extract the .nemo archive,
    patch the config, instantiate the model directly, and load weights.
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

    # Step 3: Extract the .nemo archive
    extract_dir = os.path.join("/tmp", "nemo_indicconf_extract")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    print(f"[NeMo] Extracting .nemo to {extract_dir}...")
    with tarfile.open(tmp_nemo, "r:*") as tar:
        tar.extractall(extract_dir)

    # List extracted contents
    extracted_files = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), extract_dir)
            extracted_files.append(rel)
    print(f"[NeMo] Extracted files: {extracted_files}")

    # Step 4: Load config
    config_path = os.path.join(extract_dir, "model_config.yaml")
    if not os.path.exists(config_path):
        found = glob.glob(os.path.join(extract_dir, "**", "model_config.yaml"), recursive=True)
        if found:
            config_path = found[0]

    from omegaconf import OmegaConf, open_dict
    config = OmegaConf.load(config_path)

    # Debug: print tokenizer config before patching
    if hasattr(config, "tokenizer"):
        print(f"[NeMo] Original tokenizer config: {OmegaConf.to_yaml(config.tokenizer)}")
    else:
        print(f"[NeMo] No tokenizer section in config!")

    # Step 5: Find tokenizer files and patch config
    tokenizer_files = glob.glob(os.path.join(extract_dir, "**", "*.model"), recursive=True)
    if tokenizer_files:
        tokenizer_dir = os.path.dirname(tokenizer_files[0])
    else:
        tokenizer_dir = extract_dir

    print(f"[NeMo] Tokenizer files: {tokenizer_files}")

    # Force-patch tokenizer config (open_dict allows adding new keys)
    with open_dict(config):
        if not hasattr(config, "tokenizer"):
            config.tokenizer = {}
        config.tokenizer.dir = tokenizer_dir
        config.tokenizer.type = "bpe"

    print(f"[NeMo] Patched tokenizer config: {OmegaConf.to_yaml(config.tokenizer)}")

    # Step 6: Find weights file
    weights_path = None
    for candidate in ["model_weights.ckpt", "model_weights.bin"]:
        wp = os.path.join(extract_dir, candidate)
        if os.path.exists(wp):
            weights_path = wp
            break
    if not weights_path:
        found = glob.glob(os.path.join(extract_dir, "**", "*.ckpt"), recursive=True)
        if found:
            weights_path = found[0]

    if not weights_path:
        raise FileNotFoundError(f"No model weights found in {extract_dir}")
    print(f"[NeMo] Weights file: {weights_path}")

    # Step 7: Instantiate model from config and load weights
    # This bypasses save_restore_connector entirely
    from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
    print(f"[NeMo] Instantiating EncDecHybridRNNTCTCBPEModel from patched config...")

    model = EncDecHybridRNNTCTCBPEModel(cfg=config)

    print(f"[NeMo] Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

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


