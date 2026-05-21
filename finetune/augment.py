import numpy as np
from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise, TimeStretch
import librosa
import soundfile as sf
from pydub import AudioSegment
import random
import os
import json
from datasets import Dataset, Audio

# Telephony codec simulation (8kHz bandwidth)
def simulate_telephony(audio_array, sr=16000):
    # downsample to 8kHz -> upsample back (simulates codec bandwidth limiting)
    downsampled = librosa.resample(audio_array, orig_sr=sr, target_sr=8000)
    return librosa.resample(downsampled, orig_sr=8000, target_sr=sr)

# Pause injection for natural speech rhythm
def inject_pauses(audio_path, pause_probability=0.3):
    """Add 150-600ms pauses at natural break points. Returns numpy array."""
    audio = AudioSegment.from_wav(audio_path)
    
    # Simple approach: add silence at end of segments
    pause_ms = random.randint(150, 600)
    silence = AudioSegment.silent(duration=pause_ms)
    
    if random.random() < pause_probability:
        audio = audio + silence
        
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples = samples / (2**15)
    return samples

def get_augmenter():
    # Make sure to handle missing musan noise path gracefully
    transforms = [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.4),
    ]
    
    if os.path.exists("./data/noise/musan/noise/"):
        transforms.append(
            AddBackgroundNoise(
                sounds_path="./data/noise/musan/noise/",
                min_snr_db=10, max_snr_db=25, p=0.6
            )
        )
    if os.path.exists("./data/noise/musan/speech/"):
        transforms.append(
            AddBackgroundNoise(
                sounds_path="./data/noise/musan/speech/",
                min_snr_db=15, max_snr_db=30, p=0.3
            )
        )
        
    return Compose(transforms)

def augment_sample(audio_array, sr=16000):
    """Apply augmentation. Returns augmented array."""
    augmenter = get_augmenter()
    return augmenter(audio_array, sample_rate=sr)

def augment_dataset(dataset, copies=2):
    """Create N augmented copies of each sample in-memory."""
    augmenter = get_augmenter()
    augmented = []
    for sample in dataset:
        audio = sample["audio"]["array"]
        augmented.append(sample) # original
        for _ in range(copies):
            aug_audio = augmenter(audio.copy(), sample_rate=16000)
            aug_audio = simulate_telephony(aug_audio)
            augmented.append({**sample, "audio": {"array": aug_audio, "sampling_rate": 16000}})
    return augmented

if __name__ == "__main__":
    print("Augmentation library ready. Use augment_dataset(hf_dataset) in training script.")
