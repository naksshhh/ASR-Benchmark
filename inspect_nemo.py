#!/usr/bin/env python3
import sys
import os
import inspect

def main():
    try:
        import nemo
        print(f"Installed NeMo version: {nemo.__version__}")
    except ImportError:
        print("NeMo is not installed.")
        return

    # Check which prompt/multitask classes exist under nemo.collections.asr.models
    import nemo.collections.asr as nemo_asr
    print("\n--- Inspecting nemo.collections.asr.models for classes ---")
    for name, obj in inspect.getmembers(nemo_asr.models):
        if inspect.isclass(obj):
            if "prompt" in name.lower() or "multitask" in name.lower():
                print(f"  {name} (module: {obj.__module__})")

    # Let's inspect submodules
    submodules = [
        "nemo.collections.asr.models.rnnt_bpe_models",
        "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models",
        "nemo.collections.asr.models.rnnt_bpe_models_prompt",
        "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt"
    ]
    print("\n--- Testing submodule imports ---")
    for sub in submodules:
        try:
            mod = __import__(sub, fromlist=["*"])
            print(f"  [SUCCESS] Imported {sub}")
            # print classes in the module
            classes = [name for name, obj in inspect.getmembers(mod) if inspect.isclass(obj) and obj.__module__ == sub]
            print(f"    Classes: {classes}")
        except ImportError as e:
            print(f"  [FAILED] Imported {sub}: {e}")

    # Test our alias trick!
    print("\n--- Testing alias loading trick ---")
    try:
        import nemo.collections.asr.models.rnnt_bpe_models as rnnt_bpe_models
        sys.modules['nemo.collections.asr.models.rnnt_bpe_models_prompt'] = rnnt_bpe_models
        print("  Registered sys.modules['nemo.collections.asr.models.rnnt_bpe_models_prompt'] = rnnt_bpe_models")
        
        # Check if we can import from the alias
        from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
        print(f"  [SUCCESS] Imported EncDecRNNTBPEModelWithPrompt from aliased module: {EncDecRNNTBPEModelWithPrompt}")
    except Exception as e:
        print(f"  [FAILED] Alias import failed: {e}")

if __name__ == "__main__":
    main()
