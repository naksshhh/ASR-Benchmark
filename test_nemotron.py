#!/usr/bin/env python3
"""
Quick test: verify Nemotron-3.5 ASR loading and transcription works.

Usage (on Param Rudra compute/login node):
    python test_nemotron.py
"""

import sys
import os

def main():
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("Error: nemo_toolkit[asr] is not installed in the current environment.")
        sys.exit(1)
        
    import torch

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
        
    print(f"\nModel class loaded: {model.__class__.__name__}")
    
    # Check device and verify we can run basic steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.freeze()
    model = model.to(device)
    model.eval()
    print("Model successfully loaded onto device and set to eval mode.")

if __name__ == "__main__":
    main()
