import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration

def download():
    os.environ["HF_HOME"] = f"/scratch/{os.environ.get('USER')}/hf_cache"
    
    print("Downloading ai4bharat/indicwav2vec-hindi...")
    Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi", trust_remote_code=True)
    Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi", trust_remote_code=True)
    
    print("Downloading openai/whisper-medium...")
    WhisperProcessor.from_pretrained("openai/whisper-medium")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    
    print("Done! Models are now in your scratch HF cache.")

if __name__ == "__main__":
    download()
