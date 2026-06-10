import sys
import os
import json
import torch
import numpy as np
import transformers

# 1. Monkeypatch WhisperForConditionalGeneration.__init__ to remove the 'dtype' argument if passed.
# This fixes the ctranslate2 converter compatibility bug in newer transformers versions (>=4.41.0).
_orig_init = transformers.WhisperForConditionalGeneration.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("dtype", None)
    _orig_init(self, *args, **kwargs)
transformers.WhisperForConditionalGeneration.__init__ = _patched_init

print("[Patch] Successfully monkeypatched WhisperForConditionalGeneration.__init__ to strip 'dtype'", flush=True)

# 2. Monkeypatch json.JSONEncoder.default to serialize torch and numpy dtype objects to strings.
# This prevents: TypeError: Object of type dtype is not JSON serializable.
_orig_default = json.JSONEncoder.default
def _patched_default(self, o):
    if isinstance(o, torch.dtype):
        return str(o).replace("torch.", "")
    if isinstance(o, np.dtype) or o.__class__.__name__ == "dtype":
        return str(o)
    try:
        return _orig_default(self, o)
    except TypeError:
        # Fallback for other non-serializable dtype-like objects
        if "dtype" in str(type(o)).lower():
            return str(o)
        raise

json.JSONEncoder.default = _patched_default
print("[Patch] Successfully monkeypatched json.JSONEncoder.default to serialize dtypes", flush=True)

import ctranslate2.converters.transformers as transformers_converter

def main():
    model_path = "/scratch/nakshatrak_iitp/checkpoints/whisper-medium-banking-configD/final"
    output_path = "./ct2_whisper_medium_banking_configD"
    quantization = "int8_float16"
    
    print(f"Starting CTranslate2 conversion of {model_path} to {output_path} with {quantization}...")
    
    # Configure arguments for ctranslate2 converter
    sys.argv = [
        "ct2-transformers-converter",
        "--model", model_path,
        "--output_dir", output_path,
        "--quantization", quantization,
        "--force"
    ]
    
    try:
        transformers_converter.main()
        print("\n[SUCCESS] Model converted to CTranslate2 successfully!")
        print(f"Location: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
