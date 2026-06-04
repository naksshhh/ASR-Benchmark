import argparse
import json
import os
import torch
import jiwer
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import Dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

from augment import augment_dataset

def load_manifest(manifest_path):
    import json
    import re
    try:
        with open(manifest_path) as f:
            try:
                content = f.read()
                samples = json.loads(content)
            except json.JSONDecodeError:
                f.seek(0)
                samples = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return None
    
    audio_paths = []
    sentences = []
    durations = []
    
    for s in samples:
        path = s.get("audio_filepath") or s.get("audio_path")
        sentence = s.get("text") or s.get("reference_transcript")
        duration = s.get("duration") or s.get("duration_seconds") or 0
        
        # Filter out Latin/English characters for Devanagari-only indicwav2vec
        if sentence and re.search(r'[a-zA-Z]', sentence):
            continue
            
        # Filter out extremely long audios (> 15 seconds) to prevent CUDA OOM and speed up training
        if duration > 15.0:
            continue
            
        if path and sentence:
            audio_paths.append(path)
            sentences.append(sentence)
            durations.append(duration)
            
    return Dataset.from_dict({
        "audio": audio_paths,
        "sentence": sentences,
        "duration": durations,
    }).cast_column("audio", Audio(sampling_rate=16000))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B", "C", "D"], help="Ablation config (A, B, C, or D)")
    parser.add_argument("--train-manifests", nargs="+", help="Explicit list of training manifest files")
    parser.add_argument("--output", help="Explicit output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()

    if not args.config and not args.train_manifests:
        parser.error("Either --config or --train-manifests must be specified.")

    manifest_paths = []
    config_name = "custom"
    if args.config:
        config_name = args.config
        manifest_paths = [f"data/manifests/finetune_config{args.config}.json"]
    if args.train_manifests:
        manifest_paths = args.train_manifests

    from datasets import load_from_disk, DatasetDict
    user = os.environ.get("USER", "nakshatrak_iitp")
    prep_dir = f"/scratch/{user}/preprocessed_datasets/indicwav2vec_config{config_name}"
    
    # Dynamically detect allocated CPU cores in SLURM to scale multiprocessing memory safely
    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    num_proc = num_cpus if num_cpus > 1 else None
    print(f"Allocated CPUs: {num_cpus}. Using num_proc={num_proc} for preprocessing.")

    MODEL_ID = "ai4bharat/indicwav2vec-hindi"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, trust_remote_code=True)

    if os.path.exists(prep_dir):
        print(f"Loading preprocessed dataset from {prep_dir}...")
        processed_dataset = load_from_disk(prep_dir)
        train_dataset = processed_dataset["train"]
        eval_dataset = processed_dataset["eval"]
        
        # Filter out audios longer than 15 seconds (240,000 samples at 16kHz) to prevent CUDA OOM
        # This allows us to reuse the existing cache instantly without rebuilding it!
        print("Filtering out sequences longer than 15 seconds from loaded cache...")
        train_dataset = train_dataset.filter(lambda x: [l <= 240000 for l in x], batched=True, input_columns=["input_length"])
        eval_dataset = eval_dataset.filter(lambda x: [l <= 240000 for l in x], batched=True, input_columns=["input_length"])
    else:
        datasets = []
        for path in manifest_paths:
            if not os.path.exists(path):
                print(f"Manifest {path} not found. Did you run prepare_data.py?")
                return
            ds = load_manifest(path)
            if ds is not None:
                datasets.append(ds)

        if not datasets:
            print("No valid datasets loaded.")
            return

        from datasets import concatenate_datasets
        train_dataset_full = concatenate_datasets(datasets)

        print("Splitting 10% of train set for validation.")
        split = train_dataset_full.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        if len(eval_dataset) > 1000:
            print(f"Capping validation dataset from {len(eval_dataset)} to 1000 samples to prevent evaluation bottlenecks.")
            eval_dataset = eval_dataset.select(range(1000))

        def prepare_batch(batch):
            audio = batch["audio"]
            inputs = processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
                padding=True
            )
            batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
            batch["input_values"] = inputs.input_values[0]
            batch["input_length"] = len(batch["input_values"])
            return batch

        print("Extracting features and tokenizing datasets...")
        train_dataset = train_dataset.map(prepare_batch, num_proc=num_proc, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(prepare_batch, num_proc=num_proc, remove_columns=eval_dataset.column_names)

        # Filter out empty labels or labels that exceed the downsampled audio frames (Wav2Vec2 downsamples by 320)
        # This prevents CTC loss from exploding to infinity/nan
        print("Filtering invalid CTC sequences...")
        train_dataset = train_dataset.filter(lambda x: len(x["labels"]) > 0 and len(x["labels"]) <= x["input_length"] // 320, num_proc=num_proc)
        eval_dataset = eval_dataset.filter(lambda x: len(x["labels"]) > 0 and len(x["labels"]) <= x["input_length"] // 320, num_proc=num_proc)

        # Save to disk
        processed_dataset = DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset
        })
        print(f"Saving preprocessed dataset to {prep_dir}...")
        try:
            os.makedirs(os.path.dirname(prep_dir), exist_ok=True)
            processed_dataset.save_to_disk(prep_dir)
        except Exception as e:
            print(f"Warning: Could not save preprocessed dataset to disk: {e}")

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        trust_remote_code=True
    )

    model.freeze_feature_encoder()

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": f["input_values"]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]
            
            batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
            labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
            
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions.argmax(-1)
        pred_str = processor.batch_decode(pred_ids)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    if args.output:
        out_dir = args.output
    else:
        out_dir = f"/scratch/{os.environ.get('USER', 'default')}/checkpoints/indicwav2vec-banking-config{config_name}"
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=32,          # Increased from 16 to 32 to maximize VRAM utilization on A100 80GB
        gradient_accumulation_steps=1,           # Kept at 1 (effective batch size 32)
        eval_strategy="steps",
        per_device_eval_batch_size=32,          # Added to speed up evaluation runs
        num_train_epochs=args.epochs,
        fp16=True,
        save_steps=1000,                         # Increased from 500 to evaluate/save less frequently
        eval_steps=1000,                         # Increased from 500 to evaluate/save less frequently
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        disable_tqdm=True,  # Keeps SLURM logs clean by removing progress bar spam
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    # Check for existing checkpoints to resume automatically
    import glob
    checkpoints = glob.glob(f"{out_dir}/checkpoint-*")
    if checkpoints:
        print(f"Resuming training from latest checkpoint in {out_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    trainer.save_model(f"{out_dir}/final")

if __name__ == "__main__":
    main()
