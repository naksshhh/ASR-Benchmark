import json
import os
import time
import asyncio
import librosa
import soundfile as sf

# Note: For Hinglish code-switching, edge-tts (Microsoft Azure TTS) is highly recommended
# as it naturally blends Hindi and English within the same sentence without needing to 
# chunk and stitch audio between two different models (like Kokoro + IndicTTS).
# We'll use edge-tts here for the best synthetic banking dataset.
# pip install edge-tts
import edge_tts

async def generate_audio_for_turn(text, speaker, output_path):
    # Agent: Female voice, Customer: Male voice (or vice versa)
    voice = "hi-IN-SwaraNeural" if speaker == "agent" else "hi-IN-MadhurNeural"
    
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

async def process_all():
    input_file = "data/synthetic/transcripts.json"
    output_dir = "data/synthetic/audio"
    manifest_file = "data/synthetic/manifest.json"
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, "r") as f:
        transcripts = json.load(f)
        
    manifest = []
    
    print(f"Generating audio for {len(transcripts)} utterances...")
    
    for i, turn in enumerate(transcripts):
        text = turn["text"]
        speaker = turn["speaker"]
        
        # Unique filename
        audio_filename = f"synth_{i:04d}_{speaker}.wav"
        audio_filepath = os.path.join(output_dir, audio_filename)
        
        # Edge-TTS outputs mp3 by default, but we can save as wav or convert
        temp_mp3 = audio_filepath.replace(".wav", ".mp3")
        
        try:
            await generate_audio_for_turn(text, speaker, temp_mp3)
            
            # Convert mp3 to 16kHz WAV for ASR training
            y, sr = librosa.load(temp_mp3, sr=16000)
            sf.write(audio_filepath, y, 16000)
            os.remove(temp_mp3)
            
            duration = librosa.get_duration(y=y, sr=16000)
            
            manifest.append({
                "audio_filepath": audio_filepath,
                "text": text,
                "duration": round(duration, 2),
                "speaker": speaker,
                "domain": turn.get("domain", "banking")
            })
            
            if (i+1) % 20 == 0:
                print(f"Processed {i+1}/{len(transcripts)} audio files...")
                
        except Exception as e:
            print(f"Error processing {i}: {e}")

    with open(manifest_file, "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")
            
    print(f"\nDone! Generated {len(manifest)} WAV files.")
    print(f"Manifest saved to {manifest_file}")

if __name__ == "__main__":
    asyncio.run(process_all())
