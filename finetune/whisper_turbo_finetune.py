import argparse
import json
import os
import torch
import jiwer
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)

def load_manifest(manifest_path):
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
    telephony_flags = []
    
    for s in samples:
        path = s.get("audio_filepath") or s.get("audio_path")
        sentence = s.get("text") or s.get("reference_transcript")
        duration = s.get("duration") or s.get("duration_seconds") or 0
        telephony = s.get("telephony_codec_simulation", False)
        
        if duration > 15.0:
            continue
            
        if path and sentence:
            audio_paths.append(path)
            sentences.append(sentence)
            durations.append(duration)
            telephony_flags.append(telephony)
            
    return Dataset.from_dict({
        "audio": audio_paths,
        "sentence": sentences,
        "duration": durations,
        "telephony": telephony_flags
    }).cast_column("audio", Audio(sampling_rate=16000))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E", help="Config name (default: E)")
    parser.add_argument("--train-manifests", nargs="+", help="Explicit list of training manifest files")
    parser.add_argument("--output", help="Explicit output directory for checkpoints")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (default overrides with epochs)")
    args = parser.parse_args()

    manifest_paths = []
    config_name = args.config
    if args.config:
        manifest_paths = [f"data/manifests/finetune_config{args.config}.json"]
    if args.train_manifests:
        manifest_paths = args.train_manifests

    MODEL_ID = "openai/whisper-large-v3-turbo"
    LANGUAGE = "hi"
    TASK = "transcribe"

    print(f"Loading processor and model: {MODEL_ID}...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # Force Hindi transcription — critical for mixed scripts & prevents Romanized transcription
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []

    # Get CPU allocation from SLURM environment
    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    num_proc = num_cpus if num_cpus > 1 else None
    print(f"Allocated CPUs: {num_cpus}. Using num_proc={num_proc} for preprocessing.")

    datasets = []
    for path in manifest_paths:
        if not os.path.exists(path):
            print(f"Manifest {path} not found. Running prepare_configE.py...")
            return
        ds = load_manifest(path)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        print("No valid datasets loaded.")
        return

    from datasets import concatenate_datasets
    train_dataset_full = concatenate_datasets(datasets)
    
    # Split 10% for validation tracking
    print("Splitting 10% of train set for validation.")
    split = train_dataset_full.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    if len(eval_dataset) > 1000:
        print(f"Capping validation dataset from {len(eval_dataset)} to 1000 samples.")
        eval_dataset = eval_dataset.select(range(1000))

    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("hi")

    # Filter out sentences that are too long for Whisper
    def is_labels_short_batched(sentences):
        res = []
        for s in sentences:
            if s is None:
                res.append(False)
                continue
            normalized = normalizer.normalize(s)
            labels = processor.tokenizer(normalized).input_ids
            res.append(len(labels) < 440)
        return res

    print("Filtering datasets by token length...")
    train_dataset = train_dataset.filter(is_labels_short_batched, batched=True, input_columns=["sentence"], num_proc=num_proc)
    eval_dataset = eval_dataset.filter(is_labels_short_batched, batched=True, input_columns=["sentence"], num_proc=num_proc)

    def prepare_dataset(batch):
        normalized = normalizer.normalize(batch["sentence"])
        batch["labels"] = processor.tokenizer(normalized).input_ids
        return batch

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(prepare_dataset, num_proc=num_proc, remove_columns=["sentence", "duration"])
    eval_dataset = eval_dataset.map(prepare_dataset, num_proc=num_proc, remove_columns=["sentence", "duration"])

    @dataclass  
    class TelephonyCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_features = []
            for f in features:
                audio = f["audio"]
                array = audio["array"]
                
                # Apply 8kHz Telephony Codec Bandlimiting on-the-fly
                if f.get("telephony", False):
                    downsampled = librosa.resample(array, orig_sr=audio["sampling_rate"], target_sr=8000)
                    array = librosa.resample(downsampled, orig_sr=8000, target_sr=audio["sampling_rate"])
                
                inputs = self.processor(
                    array,
                    sampling_rate=audio["sampling_rate"],
                    return_tensors="pt"
                )
                input_features.append({"input_features": inputs.input_features[0]})
                
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = TelephonyCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        pred_str = [normalizer.normalize(s) for s in pred_str]
        label_str = [normalizer.normalize(s) for s in label_str]
        
        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    if args.output:
        out_dir = args.output
    else:
        out_dir = f"/scratch/{os.environ.get('USER', 'default')}/checkpoints/whisper-turbo-banking-config{config_name}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=8,           # Safe batch size to prevent OOM on L4 / A100 for larger model
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=300,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,            # Enable for Large model to save VRAM
        fp16=True,
        eval_strategy="epoch",                   # Evaluate once per epoch to save training time
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="epoch",
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        disable_tqdm=True,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    # Scans checkpoints to auto-resume on interruption
    import glob
    checkpoints = [d for d in glob.glob(f"{out_dir}/checkpoint-*") if d.split("-")[-1].isdigit()]
    if checkpoints:
        print(f"Resuming training from latest checkpoint in {out_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    trainer.save_model(f"{out_dir}/final")
    print("Fine-tuning completed successfully. Model saved to final directory.")

if __name__ == "__main__":
    main()
