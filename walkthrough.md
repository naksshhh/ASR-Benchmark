# Banking ASR Fine-Tuning: End-to-End Walkthrough

This document outlines the complete journey of preparing, fine-tuning, and evaluating the Banking ASR models (`indicwav2vec` and `whisper`) on the Param Rudra HPC cluster.

---

## 1. Dataset Preparation & Slicing
Our first major hurdle was processing the **MUCS Hindi-English finance dataset** (90 hours). The data was provided in a strict Kaldi format (`wav.scp`, `text`, `segments`). 
*   **The Problem:** The raw `.wav` files were massive, multi-minute lecture recordings, but the model requires utterance-level segments.
*   **The Solution:** We wrote a highly parallelized audio slicer (`build_mucs_manifest.py`) using `soundfile` and `ProcessPoolExecutor`. This script read the Kaldi `segments` file, mathematically sliced the audio in memory, saved thousands of utterance-level `.wav` chunks into `segments_wav/`, and generated a ready-to-use JSON Lines manifest.
*   **Data Formats:** We then upgraded `prepare_data.py` to seamlessly handle both the JSON Lines format (from MUCS) and standard JSON Arrays (from our Gemini-generated synthetic datasets).

## 2. Optimizing for SLURM (Param Rudra)
Running jobs on an HPC cluster requires specific environment tuning:
*   **Log Spam:** By default, HuggingFace's `tqdm` progress bars spam thousands of carriage return (`\r`) characters into SLURM `.out` files. We injected `disable_tqdm=True` and `logging_steps=100` into `TrainingArguments` to ensure the SLURM logs printed clean, single-line updates.
*   **Offline Compute Nodes:** Compute nodes on Param Rudra lack internet access. We configured `export HF_HUB_OFFLINE=1` and routed `HF_HOME` to the `/scratch` directory.
*   **Forced Pre-Caching:** Because disabled models were skipped by the standard initialization, we wrote `download_hf_models.py` to forcibly download `ai4bharat/indicwav2vec-hindi` and `openai/whisper-medium` directly to the `/scratch` cache from the internet-enabled login node.

## 3. Fixing HuggingFace 4.41+ Breaking Changes
Because the environment was using the absolute latest version of the `transformers` library, we encountered and patched several breaking API changes:
*   **indicwav2vec:** Deprecated `processor.as_target_processor()` was replaced with a direct `processor.tokenizer()` call.
*   **indicwav2vec:** Replaced the deprecated `model.freeze_feature_extractor()` with the mathematically correct `model.freeze_feature_encoder()`.
*   **Whisper & indicwav2vec:** Renamed `evaluation_strategy` to `eval_strategy` in `TrainingArguments`.
*   **Whisper & indicwav2vec:** Updated the `Trainer` class initialization, replacing the deprecated `tokenizer` kwarg with the new `processing_class` kwarg to properly support audio feature extractors.

## 4. Training Resiliency & Validation
*   **Validation Data Split:** To prevent data leakage and allow for real-time validation WER tracking, we patched the training scripts to automatically split 10% of the active training dataset if the purely test dataset (`banking_100_test.json`) was omitted from the training loop.
*   **Auto-Resume:** To survive SLURM job time limits, we implemented automatic checkpoint detection. Before starting `trainer.train()`, the scripts scan the `out_dir` for existing checkpoints. If found, they automatically trigger `trainer.train(resume_from_checkpoint=True)`, allowing interrupted jobs to resume seamlessly without losing progress.

## 5. Execution and Analysis
We submitted the training jobs using **Config B** (the 639-sample Gemini synthetic dataset):
1.  **IndicWav2Vec:** The job timed out once at step 500, but thanks to the auto-resume logic, resubmitting the job allowed it to finish the full 30 epochs flawlessly, generating the `final` model checkpoint.
2.  **Whisper:** The Whisper job reached `checkpoint-1000` (Epoch 55). At this point, the training loss had plummeted to `0.0002` while validation loss slightly increased, signaling the model had perfectly memorized the dataset and was on the verge of overfitting. We concluded that training to 4000 steps was unnecessary and harmful, electing to use `checkpoint-1000` as the perfect stopping point.

## 6. Final Evaluation
With both models successfully fine-tuned, we updated `config.yaml` to point to our successful artifacts:
*   `indicwav2vec-banking` → `.../indicwav2vec-banking-configB/final`
*   `whisper-medium-banking` → `.../whisper-medium-banking-configB/checkpoint-1000`

Both models are now fully enabled and primed for evaluation!

## 7. Accent-Aware Evaluation & Config D Fine-Tuning Setup

To study regional accents and optimize performance on dialect-rich call center audio, we extended the repository to incorporate LAHAJA, RESPIN-S1.0, and Project Vaani:

1. **Obsolete Job Deletion**: Cleaned up the repository by removing obsolete SLURM scripts (`job4_ablation.sh`, `job5_eval.sh`, and `job6_whisper_ablations.sh`).
2. **Dynamic Evaluation Options**: Modified `banking_asr_eval/evaluate.py` to:
   - Accept a `--models` argument (comma-separated) to select specific models dynamically.
   - Accept a `--stratify-by` argument to automatically group evaluation metrics (e.g. by `accent_group` or `native_language`) and print the mean WER for each group.
   - Automatically copy all custom manifest columns (like `accent_group`, `native_language`, `occupation_domain`) into the final results dataframe to enable stratification on any variable.
3. **Training Script Extensions**: Upgraded both `finetune/indicwav2vec_finetune.py` and `finetune/whisper_finetune.py` to:
   - Accept `--config D` to train on the combined RESPIN, Vaani, MUCS, and Synthetic datasets.
   - Accept explicit `--train-manifests` and `--output` directory arguments for flexible command-line custom fine-tuning.
   - Support dual-compatibility in `load_manifest` to correctly read both NeMo-style (`audio_filepath`/`text`) and evaluation-style (`audio_path`/`reference_transcript`) keys.
4. **Data Preparation & Manifest Scripts**:
   - `prepare_lahaja.py`: Maps native speaker languages to five major accent groups (`hindi_belt`, `punjab_haryana`, `south_india`, `east_india`, `west_india`) and creates the evaluation manifest `data/manifests/lahaja.json`.
   - `prepare_respin.py`: Parses RESPIN Kaldi-style formats (`wav.scp`, `text`, `utt2dur`) and writes `data/manifests/respin_finance_train.json` with correct durations.
   - `prepare_vaani.py`: Downloads prioritized transcribed Hindi-belt district subsets of the IISc Vaani dataset, fixes duration calculations from the audio arrays, and writes `data/manifests/vaani_hindi_belt.json`.
   - Updated `finetune/prepare_data.py` to automatically combine these sources and compile `data/manifests/finetune_configD.json`.
5. **Config YAML Registration**: Registered `indicwav2vec-banking-configD` and `whisper-medium-banking-configD` in `config.yaml`.
6. **New SLURM Scripts**:
   - `slurm_jobs/job7_lahaja_zeroshot.sh`: Runs baseline models on LAHAJA with accent stratification.
   - `slurm_jobs/job8_configD_finetuning.sh`: Executes training of both models on Config D.

