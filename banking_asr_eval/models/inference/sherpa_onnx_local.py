"""
Sherpa-ONNX inference wrapper.

Handles streaming Zipformer and other sherpa-onnx-based models.
Requires sherpa-onnx — install with: pip install sherpa-onnx
"""

import os
import glob
from typing import Callable
from huggingface_hub import snapshot_download


def create_sherpa_onnx_model(model_id: str) -> Callable[[str], str]:
    """
    Create a Sherpa-ONNX inference function.

    Args:
        model_id: Hugging Face model repository (e.g., csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21)

    Returns:
        Callable that takes audio_path and returns transcript string.
    """
    try:
        import sherpa_onnx
    except ImportError:
        raise ImportError(
            "sherpa-onnx is not installed. Install with: pip install sherpa-onnx"
        )

    # Download model from HF
    print(f"[Sherpa-ONNX] Downloading/checking model {model_id} from Hugging Face...")
    model_dir = snapshot_download(repo_id=model_id)

    # Find model files
    encoders = glob.glob(os.path.join(model_dir, "encoder*.onnx"))
    decoders = glob.glob(os.path.join(model_dir, "decoder*.onnx"))
    joiners = glob.glob(os.path.join(model_dir, "joiner*.onnx"))
    tokens_files = glob.glob(os.path.join(model_dir, "tokens.txt"))

    def choose_file(files, name):
        if not files:
            raise FileNotFoundError(f"Could not find {name} (.onnx) file in {model_dir}")
        # Prefer non-int8 version if available, otherwise take the first
        fp32_files = [f for f in files if "int8" not in os.path.basename(f)]
        return fp32_files[0] if fp32_files else files[0]

    encoder_path = choose_file(encoders, "encoder")
    decoder_path = choose_file(decoders, "decoder")
    joiner_path = choose_file(joiners, "joiner")

    if not tokens_files:
        raise FileNotFoundError(f"Could not find tokens.txt in {model_dir}")
    tokens_path = tokens_files[0]

    # Instantiate the streaming (online) transducer recognizer
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        tokens=tokens_path,
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
    )

    import librosa

    def transcribe(audio_path: str) -> str:
        # Load audio at 16000Hz mono
        samples, sr = librosa.load(audio_path, sr=16000)
        
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        stream.input_finished()

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = recognizer.get_result(stream)
        # Some sherpa-onnx versions return a string, others return an object with .text
        text = result.text if hasattr(result, 'text') else str(result)
        return text.strip()

    return transcribe
