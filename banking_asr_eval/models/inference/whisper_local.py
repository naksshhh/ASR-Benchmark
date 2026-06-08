"""
Whisper inference wrapper.

Works locally on Mac Mini (whisper-tiny) and Param Rudra (larger models).
Uses HuggingFace transformers pipeline for consistent interface.
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Callable


def create_whisper_model(
    model_id: str = "openai/whisper-tiny",
    language: str = "hi",
    task: str = "transcribe",
    device: str = None,
) -> Callable[[str], str]:
    """
    Create a Whisper inference function.

    Args:
        model_id: HuggingFace model ID
        language: Target language code
        task: "transcribe" or "translate"
        device: Force device ("cpu", "cuda"). Auto-detected if None.

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

    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    def transcribe(audio_path: str) -> str:
        """Transcribe a single audio file."""
        import soundfile as sf
        import numpy as np

        try:
            # sf.read is extremely fast and avoids launching ffmpeg subprocesses
            audio_array, samplerate = sf.read(audio_path, dtype="float32")
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # HuggingFace Whisper pipeline requires 16000Hz samplerate
            if samplerate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=samplerate, target_sr=16000)
            
            audio_input = audio_array
        except Exception:
            # Fallback to passing path directly if soundfile fails
            audio_input = audio_path

        result = pipe(
            audio_input,
            generate_kwargs={
                "language": language,
                "task": task,
                "condition_on_prev_tokens": False,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 4,
                "compression_ratio_threshold": 1.35,
            },
            return_timestamps=False,
        )
        return result["text"].strip()

    return transcribe
