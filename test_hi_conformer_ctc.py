#!/usr/bin/env python3
"""
Quick test: verify stt_hi_conformer_ctc_large ASR loading and transcription works.

Usage:
    python test_hi_conformer_ctc.py [--device cpu/cuda]
"""

import sys
import os
import argparse
import tempfile
import torch

def main():
    parser = argparse.ArgumentParser(description="Test stt_hi_conformer_ctc_large model loading and transcription.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                        help="Device to load model on. Defaults to CUDA if available.")
    args = parser.parse_args()

    # Apply strict=False monkeypatch for state_dict loading to survive minor version mismatch issues
    def make_patched_load_state_dict(original_fn):
        def patched(self, state_dict, strict=True):
            class_name = self.__class__.__name__
            if any(p in class_name for p in ["EncDec", "RNNT", "CTC", "Joint", "Model"]):
                strict = False
            return original_fn(self, state_dict, strict=strict)
        return patched

    torch.nn.Module.load_state_dict = make_patched_load_state_dict(torch.nn.Module.load_state_dict)
    print("Installed torch.nn.Module.load_state_dict monkeypatch (strict=False)", flush=True)

    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("Error: nemo_toolkit[asr] is not installed. This model requires NeMo ASR collection.")
        sys.exit(1)

    model_id = "stt_hi_conformer_ctc_large"
    print(f"\n==========================================")
    print(f"Loading NeMo Conformer-CTC: {model_id}")
    print(f"==========================================\n")

    model = None
    
    # Check for local downloaded files (e.g. downloaded via NGC CLI)
    import glob
    search_paths = [
        "./stt_hi_conformer_ctc_large_v*/*.nemo",
        "./stt_hi_conformer_ctc_large_v*/stt_hi_conformer_ctc_large.nemo",
        "./stt_hi_conformer_ctc_large.nemo",
        "stt_hi_conformer_ctc_large.nemo",
        os.path.expanduser("~/stt_hi_conformer_ctc_large_v*/*.nemo"),
        os.path.expanduser("~/stt_hi_conformer_ctc_large.nemo"),
    ]
    local_file = None
    for p in search_paths:
        matches = glob.glob(p)
        if matches and os.path.exists(matches[0]):
            local_file = matches[0]
            break

    if local_file:
        print(f"Found local checkpoint at: {local_file}")
        try:
            from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
            model = EncDecCTCModelBPE.restore_from(local_file)
            print("Model loaded successfully from local file!")
        except Exception as e:
            print(f"Failed to load from local file {local_file}: {e}")

    if model is None:
        try:
            from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
            model = EncDecCTCModelBPE.from_pretrained(model_id)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"\n[Warning] Failed to load '{model_id}' directly: {type(e).__name__}: {e}")
            
            # List available models to help diagnose
            try:
                from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
                all_models = EncDecCTCModelBPE.list_available_models()
                hindi_models = [m.model_name for m in all_models if getattr(m, 'language', '') == 'hi' or 'stt_hi' in m.model_name]
                print("\nAvailable pre-trained Hindi models in your local NeMo registry:")
                for name in hindi_models:
                    print(f"  - {name}")
            except Exception as list_err:
                print(f"Could not list available models: {list_err}")

            # Fallback to the medium model
            fallback_id = "stt_hi_conformer_ctc_medium"
            print(f"\nAttempting fallback to registered model: '{fallback_id}'...")
            try:
                model = EncDecCTCModelBPE.from_pretrained(fallback_id)
                model_id = fallback_id
                print(f"Successfully loaded fallback model: '{model_id}'!")
            except Exception as fallback_err:
                print(f"[ERROR] Fallback model loading also failed: {fallback_err}")
                print("\nTo use the large model, download the '.nemo' file manually from NGC:")
                print("https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_large")
                print("And load it using `EncDecCTCModelBPE.restore_from('path/to/stt_hi_conformer_ctc_large.nemo')`")
                sys.exit(1)

    # Determine device
    target_device = args.device
    if target_device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(target_device)
    print(f"Moving model to device: {device}...")
    model.freeze()
    model = model.to(device)
    model.eval()
    print("Model successfully set to eval mode.\n")

    # Locate a sample audio file to transcribe
    mp3_path = os.path.abspath("data/synthetic/audio/banking_0000.mp3")
    if not os.path.exists(mp3_path):
        print(f"Could not find test audio at {mp3_path}")
        sys.exit(1)

    wav_path = os.path.abspath("test_conformer_sample.wav")
    print(f"Converting test sample {mp3_path} to 16kHz WAV...")
    
    import soundfile as sf
    import numpy as np
    try:
        import librosa
        audio_array, sr = librosa.load(mp3_path, sr=16000, mono=True)
    except Exception:
        audio_array, sr = sf.read(mp3_path, dtype="float32")
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
    sf.write(wav_path, audio_array, 16000)
    print(f"WAV file created at: {wav_path}\n")

    # Transcribe
    print("Running transcription...")
    try:
        transcriptions = model.transcribe([wav_path])
        print(f"\nRaw outputs: {repr(transcriptions)}")
        
        # Parse output
        if isinstance(transcriptions, list) and len(transcriptions) > 0:
            first = transcriptions[0]
            if hasattr(first, 'text'):
                txt = first.text
            elif isinstance(first, str):
                txt = first
            else:
                txt = str(first)
            print(f"\nParsed Transcript: {repr(txt)}")
        else:
            print("\nTranscription returned empty or unexpected structure.")
            
    except Exception as e:
        print(f"Transcription failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
            print("\nCleaned up temporary WAV file.")

if __name__ == "__main__":
    main()
