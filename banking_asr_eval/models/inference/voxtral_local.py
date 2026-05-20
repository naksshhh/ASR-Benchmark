"""
Voxtral inference wrapper.

Handles Voxtral Mini-3B and Voxtral Small-24B models.
Requires transformers >= 4.54.0 and mistral-common[audio].
"""

import torch
from typing import Callable


def create_voxtral_model(
    model_id: str,
    device: str = None,
) -> Callable[[str], str]:
    """
    Create a Voxtral inference function.

    Args:
        model_id: Hugging Face model repository (e.g., mistralai/Voxtral-Mini-3B-2507)
        device: Device to load model onto. Auto-detected if None.

    Returns:
        Callable that takes audio_path and returns transcript string.
    """
    try:
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
    except ImportError:
        raise ImportError(
            "Voxtral requires transformers >= 4.54.0. Please run: pip install transformers"
        )

    try:
        import mistral_common
    except ImportError:
        raise ImportError(
            "mistral-common with audio dependencies is required for Voxtral. "
            "Please run: pip install 'mistral-common[audio]'"
        )

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    original_model_id = model_id
    load_path = model_id
    local_only = False
    import os

    # Resolve local cached snapshot path when running offline
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        local_only = True
        try:
            from huggingface_hub import HF_HUB_CACHE
            repo_folder_name = f"models--{model_id.replace('/', '--')}"
            repo_path = os.path.join(HF_HUB_CACHE, repo_folder_name)
            if os.path.exists(repo_path):
                snapshots_path = os.path.join(repo_path, "snapshots")
                if os.path.exists(snapshots_path) and os.path.isdir(snapshots_path):
                    snapshots = os.listdir(snapshots_path)
                    if snapshots:
                        # Find the first snapshot directory
                        load_path = os.path.join(snapshots_path, snapshots[0])
                        print(f"[Voxtral] Offline mode: loading from local cache snapshot: {load_path}")
        except Exception as e:
            print(f"[Voxtral] Warning: failed to resolve local snapshot path: {e}")

    print(f"[Voxtral] Loading {original_model_id} onto {device}...")
    processor = AutoProcessor.from_pretrained(load_path, local_files_only=local_only)

    # Voxtral requires bfloat16 (or float16) on GPU for memory/numerical stability
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = VoxtralForConditionalGeneration.from_pretrained(
        load_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    ).to(device)

    model.eval()

    def transcribe(audio_path: str) -> str:
        # Request transcription in Hindi ("hi")
        inputs = processor.apply_transcription_request(
            language="hi",
            audio=audio_path,
            model_id=original_model_id,
        )
        # Move inputs to device and cast appropriate fields
        inputs = inputs.to(device, dtype=dtype)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.0)

        # Extract generated tokens (excluding prompt)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = processor.batch_decode(
            outputs[:, prompt_len:],
            skip_special_tokens=True
        )
        return decoded[0].strip()

    return transcribe
