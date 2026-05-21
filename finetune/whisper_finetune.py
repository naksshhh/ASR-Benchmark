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
        print(f"Manifest {manifest_path} not found.")
        return

    train_dataset = load_manifest(manifest_path)
    eval_dataset = load_manifest("data/manifests/banking_100_test.json")

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

    out_dir = f"/scratch/{os.environ.get('USER', 'default')}/checkpoints/whisper-medium-banking-config{args.config}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
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
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(f"{out_dir}/final")

if __name__ == "__main__":
    main()
