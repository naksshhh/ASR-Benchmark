#!/usr/bin/env python3
"""
Quick test: verify IndicConformer transcription works on a single audio file.

Usage (on Param Rudra compute node):
    python test_indicconformer.py /path/to/any/audio.wav
    
    # or test with first sample from kathbath manifest:
    python test_indicconformer.py --manifest ./data/manifests/kathbath_hindi.json
"""

import sys
import json
import os

def main():
    import nemo.collections.asr as nemo_asr
    import torch

    model_id = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"
    print(f"Loading {model_id}...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.freeze()
    model = model.to(device)
    model.eval()

    # Get audio path
    if len(sys.argv) > 1 and sys.argv[1] == "--manifest":
        manifest_path = sys.argv[2] if len(sys.argv) > 2 else "./data/manifests/kathbath_hindi.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        audio_path = manifest[0]["audio_path"]
        ref = manifest[0].get("reference_transcript", "")
        print(f"Reference: {ref}")
    elif len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        print("Usage: python test_indicconformer.py <audio_path>")
        print("   or: python test_indicconformer.py --manifest ./data/manifests/kathbath_hindi.json")
        sys.exit(1)

    print(f"Audio: {audio_path}")
    print(f"Exists: {os.path.exists(audio_path)}")

    # Test CTC decoder
    if hasattr(model, "cur_decoder"):
        model.cur_decoder = "ctc"
        print(f"\ncur_decoder = 'ctc'")

    result = model.transcribe([audio_path], batch_size=1, logprobs=False, language_id="hi")
    
    print(f"\nRaw output type: {type(result)}")
    print(f"Raw output repr: {repr(result)[:500]}")

    # Try to extract text
    if isinstance(result, (list, tuple)) and len(result) > 0:
        first = result[0]
        print(f"First element type: {type(first)}")
        print(f"First element repr: {repr(first)[:300]}")
        if hasattr(first, "text"):
            print(f"first.text = {repr(first.text)}")
        if isinstance(first, str):
            print(f"Transcript: {first}")
        if isinstance(first, dict):
            print(f"Dict keys: {first.keys()}")
    
    # Test RNNT decoder
    if hasattr(model, "cur_decoder"):
        model.cur_decoder = "rnnt"
        print(f"\ncur_decoder = 'rnnt'")
        result_rnnt = model.transcribe([audio_path], batch_size=1, language_id="hi")
        print(f"RNNT output type: {type(result_rnnt)}")
        print(f"RNNT output repr: {repr(result_rnnt)[:500]}")
        if isinstance(result_rnnt, (list, tuple)) and len(result_rnnt) > 0:
            first = result_rnnt[0]
            print(f"First element type: {type(first)}")
            if hasattr(first, "text"):
                print(f"first.text = {repr(first.text)}")
            if isinstance(first, str):
                print(f"Transcript: {first}")

    print("\nDone!")


if __name__ == "__main__":
    main()
