#!/usr/bin/env python3
"""
Whisper ASR Fine-tuning Script.
Designed for Week 3 of the ASR-Benchmark roadmap.

Finetunes openai/whisper-medium (or any other HuggingFace Whisper checkpoint)
on Kathbath and synthetic banking datasets.

Usage:
    python finetune_whisper.py \
      --train-manifest ./data/manifests/kathbath_hindi.json,./data/synthetic/manifest.json \
      --val-manifest ./data/synthetic/manifest.json \
      --model-id openai/whisper-medium \
      --output-dir ./checkpoints/whisper-medium-finetuned \
      --epochs 3
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import Dataset, Audio
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import jiwer


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 to ignore loss in PyTorch
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


def load_manifest_as_dataset(manifest_paths: str) -> Dataset:
    """Load and combine multiple JSON manifests into a single Hugging Face Dataset."""
    records = []
    
    # Support comma-separated list of manifests
    paths = [p.strip() for p in manifest_paths.split(",") if p.strip()]
    
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Manifest path not found: {path}")
            continue
            
        print(f"Loading manifest: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Resolve relative paths relative to manifest folder
        manifest_dir = os.path.dirname(os.path.abspath(path))
        
        for item in data:
            audio_path = item.get("audio_path", "")
            if audio_path and not os.path.isabs(audio_path):
                audio_path = os.path.join(manifest_dir, audio_path)
                
            if os.path.exists(audio_path):
                records.append({
                    "audio": audio_path,
                    "sentence": item.get("reference_transcript", "")
                })
            else:
                print(f"Warning: Audio file not found: {audio_path}")

    if not records:
        raise ValueError(f"No valid audio samples loaded from manifests: {manifest_paths}")

    print(f"Loaded {len(records)} total samples.")
    dataset = Dataset.from_list(records)
    
    # Cast audio column to HF Audio structure (automatically handles resampling)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper models on ASR datasets")
    parser.add_argument(
        "--train-manifest",
        required=True,
        help="Comma-separated paths to training manifest JSONs",
    )
    parser.add_argument(
        "--val-manifest",
        required=True,
        help="Comma-separated paths to validation manifest JSONs",
    )
    parser.add_argument(
        "--model-id",
        default="openai/whisper-medium",
        help="HuggingFace Whisper model ID (default: openai/whisper-medium)",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/whisper-finetuned",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--epochs", type=float, default=3.0, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size per GPU device"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before backward pass",
    )
    parser.add_argument(
        "--language", default="hindi", help="Target language (default: hindi)"
    )

    args = parser.parse_args()

    # 1. Setup multi-socket optimization for PyTorch CPU threads
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load processor and tokenizer
    print(f"Loading processor for {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id, language=args.language, task="transcribe")

    # 3. Load datasets
    print("\n--- Loading Training Dataset ---")
    train_dataset = load_manifest_as_dataset(args.train_manifest)
    print("\n--- Loading Validation Dataset ---")
    val_dataset = load_manifest_as_dataset(args.val_manifest)

    # 4. Preprocess datasets
    def prepare_dataset(batch):
        audio = batch["audio"]
        # Extract Log-Mel spectrogram features
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        # Tokenize labels
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("\nPreprocessing training dataset...")
    train_dataset = train_dataset.map(
        prepare_dataset, remove_columns=train_dataset.column_names, num_proc=2
    )

    print("Preprocessing validation dataset...")
    val_dataset = val_dataset.map(
        prepare_dataset, remove_columns=val_dataset.column_names, num_proc=2
    )

    # 5. Load model
    print(f"\nLoading model {args.model_id}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    # Whisper generation config
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Freeze encoder weights to accelerate training and prevent overfitting
    # (Optional, but highly recommended for limited domain datasets)
    model.freeze_encoder()

    # 6. Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 7. Metrics evaluation function
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    # 8. Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=False if torch_dtype == torch.bfloat16 else True,
        bf16=True if torch_dtype == torch.bfloat16 else False,
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to="none", # Disable external logging weights and biases
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
    )

    # 9. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # 10. Start Training
    print("\nStarting fine-tuning...")
    trainer.train()
    print(f"Training complete! Model saved to {args.output_dir}")

    # Save processor too
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
