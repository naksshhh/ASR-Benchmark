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

    # Step 5: Simplify multilingual tokenizer config to Hindi-only
    # The model has per-language tokenizers under config.tokenizer.langs.{lang}
    # Standard NeMo expects a flat tokenizer config with dir, type, model_path
    hi_tok = config.tokenizer.langs.hi
    print(f"[NeMo] Hindi tokenizer config: {OmegaConf.to_yaml(hi_tok)}")

    # Replace the entire tokenizer config with flat Hindi-only version
    # Keep nemo: prefixed paths — restore_from resolves them to extracted files
    with open_dict(config):
        config.tokenizer = OmegaConf.create({
            "dir": extract_dir,  # Will be overwritten by restore_from
            "type": str(hi_tok.type),
            "model_path": str(hi_tok.model_path),
            "vocab_path": str(hi_tok.vocab_path),
            "spe_tokenizer_vocab": str(hi_tok.spe_tokenizer_vocab),
        })

    print(f"[NeMo] Patched tokenizer config: {OmegaConf.to_yaml(config.tokenizer)}")

    # Step 6: Strip AI4Bharat-specific config keys incompatible with NeMo 2.7.3
    # Instead of hardcoding keys, introspect actual NeMo classes to find unsupported params
    import inspect

    classes_to_check = {
        "decoder": "nemo.collections.asr.modules.rnnt.RNNTDecoder",
        "joint": "nemo.collections.asr.modules.rnnt.RNNTJoint",
    }
    for section, class_path in classes_to_check.items():
        if not hasattr(config, section):
            continue
        try:
            parts = class_path.rsplit(".", 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            cls = getattr(mod, parts[1])
            sig = inspect.signature(cls.__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not has_kwargs:
                # Only strip if class doesn't accept **kwargs
                section_keys = set(OmegaConf.to_container(config[section]).keys())
                # Keep _target_ and other hydra keys
                hydra_keys = {k for k in section_keys if k.startswith("_")}
                unsupported = section_keys - valid_params - hydra_keys
                if unsupported:
                    print(f"[NeMo] Removing unsupported keys from {section}: {unsupported}")
                    with open_dict(config):
                        for key in unsupported:
                            del config[section][key]
        except Exception as e:
            print(f"[NeMo] Warning: could not introspect {class_path}: {e}")
            # Fallback: remove known AI4Bharat keys
            for key in ["multisoftmax", "multilingual", "num_langs", "lang_ids"]:
                if key in config.get(section, {}):
                    print(f"[NeMo] Removing known AI4Bharat key: {section}.{key}")
                    with open_dict(config):
                        del config[section][key]

    # Step 7: Save modified config back into the extracted dir and repack
    OmegaConf.save(config, config_path)
    print(f"[NeMo] Saved patched config to {config_path}")

    # Repack the .nemo archive with the modified config
    repacked_nemo = os.path.join("/tmp", "indicconformer_hi_repacked.nemo")
    print(f"[NeMo] Repacking .nemo archive to {repacked_nemo}...")
    with tarfile.open(repacked_nemo, "w:gz") as tar:
        for item in os.listdir(extract_dir):
            tar.add(os.path.join(extract_dir, item), arcname=item)

    # Step 7: Load using restore_from on the repacked archive
    # restore_from extracts to its own temp dir and resolves nemo: paths
    from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
    print(f"[NeMo] Loading EncDecHybridRNNTCTCBPEModel via restore_from...")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=repacked_nemo)
    return model


def create_nemo_model(model_id: str, target_lang: str = "hi-IN") -> Callable[[str], str]:
    """
    Create a NeMo ASR inference function.

    Args:
        model_id: NeMo model identifier (e.g., nvidia/parakeet-tdt-0.6b-v3)
        target_lang: Target language tag for multilingual/conditioned models (e.g., hi-IN, auto)

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

    # Define a helper to patch load_state_dict on any class/module
    def make_patched_load_state_dict(original_fn):
        def patched(self, state_dict, strict=True):
            class_name = self.__class__.__name__
            if any(p in class_name for p in ["EncDec", "RNNT", "CTC", "Joint", "Model", "Prompt"]):
                strict = False
            return original_fn(self, state_dict, strict=strict)
        return patched

    # Apply to torch.nn.Module first
    torch.nn.Module.load_state_dict = make_patched_load_state_dict(torch.nn.Module.load_state_dict)
    print("[NeMo] Installed torch.nn.Module.load_state_dict monkeypatch (strict=False)", flush=True)

    # Register module alias and map missing prompt module to existing hybrid prompt class
    try:
        import sys
        import types
        import nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt as hybrid_prompt
        EncDecHybridRNNTCTCBPEModelWithPrompt = hybrid_prompt.EncDecHybridRNNTCTCBPEModelWithPrompt
        
        module_name = 'nemo.collections.asr.models.rnnt_bpe_models_prompt'
        m = types.ModuleType(module_name)
        m.EncDecRNNTBPEModelWithPrompt = EncDecHybridRNNTCTCBPEModelWithPrompt
        sys.modules[module_name] = m
        
        # Also bind to the parent models module
        import nemo.collections.asr.models as models
        models.rnnt_bpe_models_prompt = m
        print("[NeMo] Successfully mapped EncDecRNNTBPEModelWithPrompt -> EncDecHybridRNNTCTCBPEModelWithPrompt", flush=True)
    except Exception as e:
        print(f"[NeMo] Warning: Could not set up module mapping: {e}", flush=True)

    # Now import and patch other class load_state_dict methods if they override it
    try:
        import pytorch_lightning as pl
        if hasattr(pl.LightningModule, "load_state_dict"):
            pl.LightningModule.load_state_dict = make_patched_load_state_dict(pl.LightningModule.load_state_dict)
            print("[NeMo] Installed pytorch_lightning.LightningModule.load_state_dict monkeypatch", flush=True)
    except Exception:
        pass

    try:
        import nemo.core.classes as nemo_classes
        if hasattr(nemo_classes.ModelPT, "load_state_dict"):
            nemo_classes.ModelPT.load_state_dict = make_patched_load_state_dict(nemo_classes.ModelPT.load_state_dict)
            print("[NeMo] Installed nemo.core.classes.ModelPT.load_state_dict monkeypatch", flush=True)
    except Exception:
        pass

    # Load the model — handle NeMo 2.x extraction issues for IndicConformer
    is_indicconformer = "indicconformer" in model_id.lower()
    is_nemotron = "nemotron" in model_id.lower()

    model = None
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    except TypeError as e:
        if "abstract" in str(e).lower():
            print(f"[NeMo] ASRModel.from_pretrained failed with abstract class error: {e}. "
                  f"Trying specific concrete classes...")
            # Try loading with concrete classes
            try:
                from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
                print(f"[NeMo] Attempting to load using EncDecHybridRNNTCTCBPEModel...")
                model = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_id)
                print(f"[NeMo] Loaded successfully using EncDecHybridRNNTCTCBPEModel")
            except Exception as e_hybrid:
                print(f"[NeMo] EncDecHybridRNNTCTCBPEModel failed: {e_hybrid}")
                try:
                    from nemo.collections.asr.models import EncDecRNNTBPEModel
                    print(f"[NeMo] Attempting to load using EncDecRNNTBPEModel...")
                    model = EncDecRNNTBPEModel.from_pretrained(model_id)
                    print(f"[NeMo] Loaded successfully using EncDecRNNTBPEModel")
                except Exception as e_rnnt:
                    print(f"[NeMo] EncDecRNNTBPEModel failed: {e_rnnt}")
                    raise e
        else:
            raise
    except FileNotFoundError as e:
        if "model_config.yaml" in str(e) and is_indicconformer:
            print(f"[NeMo] from_pretrained failed (missing model_config.yaml). "
                  f"Attempting manual .nemo archive extraction...")
            model = _load_nemo_manually(nemo_asr, model_id)
        elif "stt_hi_conformer_ctc_large" in model_id:
            print(f"[NeMo] Warning: Model '{model_id}' was not found in the local NeMo registry: {e}")
            
            # Check for local downloaded files
            import glob
            search_paths = [
                "./stt_hi_conformer_ctc_large.nemo",
                "stt_hi_conformer_ctc_large.nemo",
                "/scratch/*/stt_hi_conformer_ctc_large.nemo",
                "/scratch/*/*/stt_hi_conformer_ctc_large.nemo",
                os.path.expanduser("~/stt_hi_conformer_ctc_large.nemo"),
                os.path.expanduser("~/.cache/torch/NeMo/stt_hi_conformer_ctc_large.nemo"),
            ]
            local_file = None
            for p in search_paths:
                matches = glob.glob(p)
                if matches and os.path.exists(matches[0]):
                    local_file = matches[0]
                    break
            
            if local_file:
                print(f"[NeMo] Found local checkpoint at {local_file}. Restoring...")
                from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
                model = EncDecCTCModelBPE.restore_from(local_file)
            else:
                fallback_id = "stt_hi_conformer_ctc_medium"
                print(f"[NeMo] No local checkpoint found for '{model_id}'. Falling back to registered model: '{fallback_id}'")
                from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
                model = EncDecCTCModelBPE.from_pretrained(fallback_id)
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
    elif is_nemotron and hasattr(model, "cur_decoder"):
        model.cur_decoder = "rnnt"
        print(f"[NeMo] Set cur_decoder='rnnt' for Nemotron-3.5")

    # Monkeypatch model.forward to temporarily replace torch.cat to fix off-by-one prompt shape mismatch
    if is_nemotron:
        _orig_model_forward = model.forward
        def _patched_model_forward(self, *args, **kwargs):
            import torch
            import torch.nn.functional as F
            _orig_cat = torch.cat
            
            def _patched_cat(tensors, dim=-1, *cat_args, **cat_kwargs):
                if len(tensors) == 2 and isinstance(tensors[0], torch.Tensor) and isinstance(tensors[1], torch.Tensor):
                    t0, t1 = tensors
                    if t0.dim() == 3 and t1.dim() == 3:
                        if (dim == -1 or dim == 2) and t0.shape[1] != t1.shape[1]:
                            diff = t0.shape[1] - t1.shape[1]
                            if diff > 0:
                                t1 = F.pad(t1, (0, 0, 0, diff))
                            else:
                                t1 = t1[:, :t0.shape[1], :]
                            tensors = [t0, t1]
                return _orig_cat(tensors, dim, *cat_args, **cat_kwargs)
                
            torch.cat = _patched_cat
            try:
                return _orig_model_forward(*args, **kwargs)
            finally:
                torch.cat = _orig_cat

        import types
        model.forward = types.MethodType(_patched_model_forward, model)
        print("[NeMo] Patched model.forward to resolve prompt shape mismatch", flush=True)

    # Monkeypatch Lhotse prompt dataset to handle None language and map manifest tags during transcription
    # NeMo's Lhotse dataset reads cut.supervisions[0].language which resolves to tags in the manifest
    # (like "hindi", "mixed", "english", "auto") that are invalid in Nemotron's vocabulary.
    if is_nemotron:
        try:
            import nemo.collections.asr.data.audio_to_text_lhotse_prompt as lhotse_prompt_mod
            _default_lang = target_lang

            for _cls_name in ['PromptedAudioToTextLhotseDataset', 'LhotseSpeechToTextBpeDatasetWithPrompt']:
                _cls = getattr(lhotse_prompt_mod, _cls_name, None)
                if _cls and hasattr(_cls, '_get_prompt_index'):
                    _orig = _cls._get_prompt_index
                    def _make_patch(orig):
                        def _patched(self, lang):
                            print(f"[NeMo DEBUG] _patched_get_prompt_index called with: lang={repr(lang)}, _default_lang={repr(_default_lang)}", flush=True)
                            if lang is None or str(lang) == 'None' or str(lang).strip() == '':
                                lang = _default_lang
                            
                            # Map language/dialect tags to valid Nemotron prompt locales
                            lang_str = str(lang).lower().strip()
                            if lang_str in ['hi', 'hindi', 'hi-in', 'mixed', 'auto']:
                                lang = 'hi-IN'
                            elif lang_str in ['en', 'english', 'en-us']:
                                lang = 'en-US'
                            else:
                                # Fallback for other regional Indian languages in Lahaja/Vaani
                                lang = 'hi-IN'
                            print(f"[NeMo DEBUG] _patched_get_prompt_index resolved to: lang={repr(lang)}", flush=True)
                            return orig(self, lang)
                        return _patched
                    _cls._get_prompt_index = _make_patch(_orig)
                    print(f"[NeMo] Monkeypatched {_cls_name}._get_prompt_index to default None and map manifest tags", flush=True)
        except Exception as e:
            print(f"[NeMo] Warning: Could not monkeypatch _get_prompt_index: {e}")

    def _ensure_wav_16k(audio_path: str) -> str:
        """
        IndicConformer and Nemotron require 16kHz mono WAV input.
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
        tmp.close()
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
            elif is_nemotron:
                from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import HybridRNNTCTCPromptTranscribeConfig
                _tcfg = HybridRNNTCTCPromptTranscribeConfig(target_lang=target_lang, num_workers=0, batch_size=1)
                print(f"[NeMo DEBUG] Transcribing file: {converted_path} with target_lang={target_lang}", flush=True)
                transcriptions = model.transcribe(
                    [converted_path], override_config=_tcfg
                )
                print(f"[NeMo DEBUG] Raw transcriptions: {repr(transcriptions)}", flush=True)
            else:
                transcriptions = model.transcribe([converted_path])

            result = _extract_text(transcriptions)
            print(f"[NeMo DEBUG] Extracted transcript: {repr(result)}", flush=True)

            # Strip language tags if model is Nemotron (e.g. <hi-IN> or <en-US> at end of text)
            if is_nemotron and result:
                import re
                result = re.sub(r'<[a-zA-Z]{2,3}(-[a-zA-Z]{2,4})?>', '', result)
                result = re.sub(r'\s+', ' ', result).strip()

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


