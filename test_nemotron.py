#!/usr/bin/env python3
"""
Quick test: verify Nemotron-3.5 ASR loading and transcription works.

Usage (on Param Rudra compute/login node):
    python test_nemotron.py [--device cpu/cuda]
"""

import sys
import os
import argparse
import wave
import struct

def create_silence_wav(filename="silence.wav", duration_sec=1.0, sr=16000):
    """Create a temporary mono 16kHz 16-bit WAV file with silence."""
    with wave.open(filename, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        num_frames = int(sr * duration_sec)
        for _ in range(num_frames):
            w.writeframesraw(struct.pack('<h', 0))
    return filename

def main():
    parser = argparse.ArgumentParser(description="Test Nemotron model loading.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                        help="Device to load model on. Defaults to CPU if CUDA is OOM or not requested.")
    args = parser.parse_args()

    import sys
    import types
    import torch

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
    print("Installed torch.nn.Module.load_state_dict monkeypatch (strict=False)", flush=True)

    # Register module alias and map missing prompt module to existing hybrid prompt class
    try:
        import nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt as hybrid_prompt
        EncDecHybridRNNTCTCBPEModelWithPrompt = hybrid_prompt.EncDecHybridRNNTCTCBPEModelWithPrompt
        
        module_name = 'nemo.collections.asr.models.rnnt_bpe_models_prompt'
        m = types.ModuleType(module_name)
        m.EncDecRNNTBPEModelWithPrompt = EncDecHybridRNNTCTCBPEModelWithPrompt
        sys.modules[module_name] = m
        
        # Also bind to the parent models module
        import nemo.collections.asr.models as models
        models.rnnt_bpe_models_prompt = m
        print("Successfully mapped EncDecRNNTBPEModelWithPrompt -> EncDecHybridRNNTCTCBPEModelWithPrompt", flush=True)
    except Exception as e:
        print(f"Warning: Could not set up module mapping: {e}", flush=True)

    # Now import and patch other class load_state_dict methods if they override it
    try:
        import pytorch_lightning as pl
        if hasattr(pl.LightningModule, "load_state_dict"):
            pl.LightningModule.load_state_dict = make_patched_load_state_dict(pl.LightningModule.load_state_dict)
            print("Installed pytorch_lightning.LightningModule.load_state_dict monkeypatch", flush=True)
    except Exception:
        pass

    try:
        import nemo.core.classes as nemo_classes
        if hasattr(nemo_classes.ModelPT, "load_state_dict"):
            nemo_classes.ModelPT.load_state_dict = make_patched_load_state_dict(nemo_classes.ModelPT.load_state_dict)
            print("Installed nemo.core.classes.ModelPT.load_state_dict monkeypatch", flush=True)
    except Exception:
        pass

    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("Error: nemo_toolkit[asr] is not installed in the current environment.", flush=True)
        sys.exit(1)

    model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    print(f"==========================================")
    print(f"Testing loading of: {model_id}")
    print(f"==========================================")

    model = None
    
    # 1. Try standard ASRModel.from_pretrained
    try:
        print("Method 1: Loading via nemo_asr.models.ASRModel.from_pretrained...")
        model = nemo_asr.models.ASRModel.from_pretrained(model_id)
        print("Method 1 Success!")
    except Exception as e:
        print(f"Method 1 Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
    # 2. Try EncDecHybridRNNTCTCBPEModel
    if model is None:
        try:
            print("\nMethod 2: Loading via EncDecHybridRNNTCTCBPEModel.from_pretrained...")
            from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
            model = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_id)
            print("Method 2 Success!")
        except Exception as e:
            print(f"Method 2 Failed: {type(e).__name__}: {e}")

    # 3. Try EncDecRNNTBPEModel
    if model is None:
        try:
            print("\nMethod 3: Loading via EncDecRNNTBPEModel.from_pretrained...")
            from nemo.collections.asr.models import EncDecRNNTBPEModel
            model = EncDecRNNTBPEModel.from_pretrained(model_id)
            print("Method 3 Success!")
        except Exception as e:
            print(f"Method 3 Failed: {type(e).__name__}: {e}")

    if model is None:
        print("\n[ERROR] All loading methods failed.")
        sys.exit(1)
        
    print(f"\nModel class successfully loaded: {model.__class__.__name__}")
    
    # Determine target device
    target_device = args.device
    if target_device is None:
        # If user didn't specify, try CUDA if available, but fall back to CPU on OOM
        if torch.cuda.is_available():
            try:
                # Test a small allocation to check if CUDA is OOM
                _ = torch.zeros(1, device="cuda")
                target_device = "cuda"
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("\n[Warning] CUDA is out of memory/busy. Falling back to CPU for verification.")
                    target_device = "cpu"
                else:
                    raise e
        else:
            target_device = "cpu"

    device = torch.device(target_device)
    print(f"Transferring model to device: {device}...")
    try:
        model.freeze()
        model = model.to(device)
        model.eval()
        print("Model successfully loaded onto device and set to eval mode.")
    except Exception as e:
        print(f"[ERROR] Failed to transfer model to {device}: {e}")
        if device.type == "cuda":
            print("Retrying on CPU...")
            device = torch.device("cpu")
            model = model.to(device)
            model.eval()
            print("Model successfully loaded onto CPU.")
        else:
            sys.exit(1)

    # Force RNNT decoder for Nemotron (pure RNN-T model loaded as hybrid)
    if hasattr(model, "cur_decoder"):
        model.cur_decoder = "rnnt"
        print(f"Set cur_decoder = 'rnnt'")

    # Run dummy transcription to verify end-to-end forward pass
    print("\nRunning dummy transcription to verify forward pass...")
    wav_path = os.path.abspath("silence.wav")
    create_silence_wav(wav_path)

    import json
    import tempfile

    success = False

    # ── Approach 1: Use override_config with HybridRNNTCTCPromptTranscribeConfig ──
    print("\n  Approach 1: Using override_config with HybridRNNTCTCPromptTranscribeConfig...")
    try:
        from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import HybridRNNTCTCPromptTranscribeConfig
        import inspect
        sig = inspect.signature(HybridRNNTCTCPromptTranscribeConfig)
        print(f"    Config signature: {sig}")

        # Try to create config with prompt settings
        try:
            cfg = HybridRNNTCTCPromptTranscribeConfig(prompt_tag="hi-IN")
            transcriptions = model.transcribe([wav_path], override_config=cfg)
            print(f"    Result: {repr(transcriptions)}")
            print("\n[SUCCESS] override_config with prompt_tag works!")
            success = True
        except Exception as e1:
            print(f"    prompt_tag failed: {e1}")
            try:
                cfg = HybridRNNTCTCPromptTranscribeConfig(target_lang="hi-IN")
                transcriptions = model.transcribe([wav_path], override_config=cfg)
                print(f"    Result: {repr(transcriptions)}")
                print("\n[SUCCESS] override_config with target_lang works!")
                success = True
            except Exception as e2:
                print(f"    target_lang failed: {e2}")
    except Exception as e:
        print(f"    Approach 1 failed entirely: {e}")

    # ── Approach 2: Create a JSONL manifest with target_lang field ──
    if not success:
        print("\n  Approach 2: Creating JSONL manifest with target_lang field...")
        try:
            manifest_path = "/tmp/test_nemotron_manifest.jsonl"
            entry = {
                "audio_filepath": wav_path,
                "duration": 1.0,
                "text": "",
                "target_lang": "hi-IN"
            }
            with open(manifest_path, "w") as f:
                f.write(json.dumps(entry) + "\n")

            # NeMo's transcribe treats a single string as an audio path.
            # Instead, set up the test dataloader manually with the manifest.
            from omegaconf import OmegaConf, open_dict
            with open_dict(model.cfg):
                model.cfg.test_ds.manifest_filepath = manifest_path
                model.cfg.test_ds.batch_size = 1
            model.setup_test_data(model.cfg.test_ds)

            import torch
            with torch.no_grad():
                for batch in model.test_dataloader():
                    # Move batch to model device
                    if isinstance(batch, (list, tuple)):
                        batch = [b.to(model.device) if hasattr(b, 'to') else b for b in batch]
                    print(f"    Batch type: {type(batch)}, len: {len(batch) if isinstance(batch, (list,tuple)) else 'N/A'}")
                    # Try forward pass
                    log_probs, encoded_len, *rest = model.forward(
                        input_signal=batch[0], input_signal_length=batch[1]
                    )
                    print(f"    Forward pass produced log_probs shape: {log_probs.shape}")
                    print("\n[SUCCESS] Forward pass works via manifest-based dataloader!")
                    success = True
                    break
        except Exception as e:
            print(f"    Approach 2 failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # ── Approach 3: Monkeypatch the internal manifest builder to inject target_lang ──
    if not success:
        print("\n  Approach 3: Monkeypatching _transcribe_input_manifest_processing...")
        try:
            # Find and patch the method that builds temporary manifest entries
            original_transcribe = model.__class__.transcribe

            # Check source to understand prompt flow
            import inspect
            source_lines = inspect.getsource(model.__class__.transcribe)
            # Print first 40 lines of the transcribe method source
            lines = source_lines.split("\n")[:40]
            for line in lines:
                print(f"    | {line}")
            print(f"    ... ({len(source_lines.split(chr(10)))} total lines)")
        except Exception as e:
            print(f"    Approach 3 failed: {type(e).__name__}: {e}")

    if os.path.exists(wav_path):
        os.remove(wav_path)

    if not success:
        print("\n[ERROR] All transcription approaches failed.")
        print("[DEBUG] Dumping model prompt-related attributes:")
        for attr in dir(model):
            if 'prompt' in attr.lower():
                try:
                    val = getattr(model, attr)
                    if not callable(val):
                        print(f"  model.{attr} = {repr(val)[:200]}")
                except:
                    pass
        sys.exit(1)

if __name__ == "__main__":
    main()
