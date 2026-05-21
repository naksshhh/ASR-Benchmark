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
    with open(manifest_path) as f:
        samples = [json.loads(l) for l in f]
    
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

    train_dataset = load_manifest(manifest_path)
    # Optional: apply augmentation
    # train_dataset = Dataset.from_list(augment_dataset(train_dataset, copies=1))

    # Eval on our banking test set
    eval_dataset = load_manifest("data/manifests/banking_100_test.json")

    MODEL_ID = "ai4bharat/indicwav2vec2-hindi"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    def prepare_batch(batch):
        audio = batch["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
            padding=True
        )
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])
        return batch

    train_dataset = train_dataset.map(prepare_batch, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(prepare_batch, remove_columns=eval_dataset.column_names)

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    model.freeze_feature_extractor()

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": f["input_values"]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]
            
            batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
            
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
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
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
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(f"{out_dir}/final")

if __name__ == "__main__":
    main()
