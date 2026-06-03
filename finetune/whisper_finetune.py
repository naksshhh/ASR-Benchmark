import argparse
import json
import os
import torch
import jiwer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)

def load_manifest(manifest_path):
    import json
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
    
    # Always split 10% for validation (Trainer uses this for eval_loss / early stopping)
    # The 100 sample test set is strictly reserved for the final evaluate.py script
    print("Splitting 10% of train set for validation.")
    split = train_dataset_full.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    MODEL_ID = "openai/whisper-medium"
    LANGUAGE = "hi"
    TASK = "transcribe"

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # Force Hindi transcription — critical, prevents romanization
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        
        # Use IndicNLP normalization for Hindi text
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer("hi")
        normalized = normalizer.normalize(batch["sentence"])
        
        batch["labels"] = processor.tokenizer(normalized).input_ids
        return batch

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)

    # Filter out sequences that are too long for Whisper (max length 448)
    def is_in_length_range(length):
        return length < 448

    train_dataset = train_dataset.filter(lambda x: is_in_length_range(len(x["labels"])))
    eval_dataset = eval_dataset.filter(lambda x: is_in_length_range(len(x["labels"])))

    @dataclass  
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
            input_features = [{"input_features": f["input_features"]} for f in features]
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

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    if args.output:
        out_dir = args.output
    else:
        out_dir = f"/scratch/{os.environ.get('USER', 'default')}/checkpoints/whisper-medium-banking-config{config_name}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        disable_tqdm=True,  # Keeps SLURM logs clean by removing progress bar spam
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
