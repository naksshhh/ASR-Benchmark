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
    
    return Dataset.from_dict({
        "audio": [s["audio_filepath"] for s in samples],
        "sentence": [s["text"] for s in samples],
        "duration": [s.get("duration", 0) for s in samples],
    }).cast_column("audio", Audio(sampling_rate=16000))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B", "C"], required=True, help="Ablation config (A, B, or C)")
    args = parser.parse_args()

    manifest_path = f"data/manifests/finetune_config{args.config}.json"
    if not os.path.exists(manifest_path):
        print(f"Manifest {manifest_path} not found. Did you run prepare_data.py?")
        return

    train_dataset_full = load_manifest(manifest_path)
    # Optional: apply augmentation
    # train_dataset_full = Dataset.from_list(augment_dataset(train_dataset_full, copies=1))

    # Always split 10% for validation (Trainer uses this for eval_loss / early stopping)
    # The 100 sample test set is strictly reserved for the final evaluate.py script
    print("Splitting 10% of train set for validation.")
    split = train_dataset_full.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    MODEL_ID = "ai4bharat/indicwav2vec-hindi"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, trust_remote_code=True)

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

    train_dataset = train_dataset.map(prepare_batch, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(prepare_batch, remove_columns=eval_dataset.column_names)

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

    out_dir = f"/scratch/{os.environ.get('USER', 'default')}/checkpoints/indicwav2vec-banking-config{args.config}"
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=500,
        eval_steps=500,
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
