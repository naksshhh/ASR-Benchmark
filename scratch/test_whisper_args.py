import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import sys

sys.stdout.reconfigure(encoding='utf-8')

model_id = "openai/whisper-tiny"
try:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )
    
    # Let's create a dummy input (1 second of silence)
    import numpy as np
    dummy_input = np.zeros(16000, dtype=np.float32)
    
    # Try different generation arguments
    kwargs_list = [
        {"condition_on_prev_tokens": False},
        {"repetition_penalty": 1.1},
        {"no_repeat_ngram_size": 4},
        {"temperature": 0.0},
        {"compression_ratio_threshold": 1.35},
        # Combined without no_speech_threshold
        {
            "condition_on_prev_tokens": False,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 4,
            "compression_ratio_threshold": 1.35,
        }
    ]
    
    for kwargs in kwargs_list:
        print(f"Testing kwargs: {kwargs}")
        try:
            res = pipe(dummy_input, generate_kwargs=kwargs)
            print(f"  Success! Result: {res}")
        except Exception as e:
            print(f"  Error: {e}")
except Exception as e:
    print(f"Model loading or pipeline setup failed: {e}")
