import sys
import os
import transformers

# Monkeypatch WhisperForConditionalGeneration.__init__ to remove the 'dtype' argument if passed.
# This fixes the ctranslate2 converter compatibility bug in newer transformers versions (>=4.41.0).
_orig_init = transformers.WhisperForConditionalGeneration.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("dtype", None)
    _orig_init(self, *args, **kwargs)
transformers.WhisperForConditionalGeneration.__init__ = _patched_init

print("[Patch] Successfully monkeypatched WhisperForConditionalGeneration.__init__ to strip 'dtype'", flush=True)

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
