"""
Generic HuggingFace ASR inference wrapper.

For models like AI4Bharat's IndicWav2Vec, IndicConformer, etc.
that follow the standard HuggingFace ASR pipeline interface.
"""

import torch
from transformers import pipeline
from typing import Callable


def create_hf_model(
    model_id: str,
    device: str = None,
) -> Callable[[str], str]:
    """
    Create a HuggingFace ASR inference function.
    Prefers manual CTC loading to avoid GPU pipeline dictionary errors,
    and falls back to pipeline for non-CTC models.

    Args:
        model_id: HuggingFace model ID
        device: Force device. Auto-detected if None.

    Returns:
        Callable that takes audio_path and returns transcript string.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    from transformers import AutoModelForCTC, AutoFeatureExtractor, AutoTokenizer
    import librosa

    print(f"[HF Generic] Loading {model_id} onto {device}...")

    try:
        # Load model, feature extractor, and tokenizer manually
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True).to(device)
        model.eval()

        def transcribe_ctc(audio_path: str) -> str:
            speech_array, sr = librosa.load(audio_path, sr=16000)
            inputs = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt")

            # Safely cast dict values to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

            predicted_ids = torch.argmax(logits, dim=-1)
            # Convert to flat Python list of ints for custom tokenizers
            ids_list = predicted_ids[0].cpu().tolist()
            transcription = tokenizer.decode(ids_list, skip_special_tokens=True)
            return transcription.strip()

        return transcribe_ctc

    except Exception as e:
        print(f"[HF Generic] CTC loading failed: {e}. Falling back to standard pipeline...")
        from transformers import pipeline
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            trust_remote_code=True,
        )

        def transcribe_pipeline(audio_path: str) -> str:
            result = pipe(audio_path)
            return result["text"].strip()

        return transcribe_pipeline
