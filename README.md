<p align="center">
    <h2 align="center">MDS: Data Selection for Multi-turn Dialogue Instruction Tuning <br>
        [AAAI'26 Oral]
    <br>
    <em>
    <a href="https://deepblue666.github.io/">Bo Li</a>, Shikun Zhang and Wei Ye 
     </em> </h2>
</p>

This repository releases the core data selection and evaluation pipeline used in our work on multi-turn dialogue instruction tuning (MDS). The released contents include:

- MDS global-stage selection code
- MDS local-stage selection code
- Conversion scripts for selected train/test dialogues into disk-style data
- Unified evaluation code for multi-turn dialogue predictions
- Released training and evaluation datasets
- An environment file `requirements.txt` listing the required packages and versions


---

## Overview

Multi-turn dialogue corpora are often noisy and structurally inconsistent. Compared with single-turn instruction data, they are more likely to contain topic drift, repetitive chit-chat, weak information progress, and mismatched answer formats across turns. MDS is designed to address this problem from a **dialogue-level data selection** perspective.

MDS consists of two stages:

1. **Global stage**  
   MDS first represents each dialogue in the **user-query trajectory space**, clusters dialogues into semantic bins, and performs bin-wise candidate selection with redundancy control. This stage aims to preserve broad semantic coverage while avoiding over-selection of high-frequency interaction patterns.

2. **Local stage**  
   MDS then refines the global candidate pool with **dialogue-level structural scoring**, including entity-grounded signals and query-answer form/style consistency. Final training subsets are selected in a budgeted bin-wise manner.

The full workflow in the paper is:

`dialogue pool -> global-stage candidate construction -> local-stage selection -> disk-style train/eval data -> model training -> model prediction -> evaluation`

This repository focuses on the **selection, data conversion, and evaluation** parts of that workflow.

---

## What Is Released

This repository includes the following components:

### Code
- `mds_global_stage.py`  
  Global-stage dialogue selection in the user-query trajectory space.

- `mds_local_stage.py`  
  Local-stage dialogue scoring and final bin-wise budgeted selection.

- `build_disk_data.py`  
  Conversion of selected train dialogues and released evaluation dialogues into disk-style data format for downstream training or evaluation.  
  If your final conversion script uses a different filename, replace `build_disk_data.py` in this README accordingly.

- `mds_dialogue_eval.py`  
  Unified evaluation script for model prediction files.

### Environment
- `requirements.txt`  
  Environment file listing the required packages and version numbers for this repository.

### Data
- `data/train/`
  - `baize_chat_data_use.jsonl`
  - `banking_train_50k_use.jsonl`

- `data/eval/`
  - `banking_test_300.jsonl`
  - `ConsistentChat_use_270.jsonl`
  - `dialogue_test_unseen_use.jsonl`
  - `mt_eval_part_use.jsonl`
---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── mds_global_stage.py
├── mds_local_stage.py
├── build_disk_data.py
├── mds_dialogue_eval.py
├── data/
│   ├── train/
│   │   ├── baize_chat_data_use.jsonl
│   │   └── banking_train_50k_use.jsonl
│   └── eval/
│       ├── banking_test_300.jsonl
│       ├── ConsistentChat_use_270.jsonl
│       ├── dialogue_test_unseen_use.jsonl
│       └── mt_eval_part_use.jsonl
└── outputs/
```

A typical workflow is:

```text
data/train/*.jsonl
    -> mds_global_stage.py
    -> outputs/mds_global_candidates*.jsonl
    -> mds_local_stage.py
    -> outputs/mds_final_selected*.jsonl
    -> build_disk_data.py
    -> disk-style train / eval data
    -> model training (external to this repo)
    -> model inference (external to this repo)
    -> mds_dialogue_eval.py
```

---

## Installation

We recommend Python 3.10 or above.

Install the environment with your preferred method. For example:

```bash
pip install -r requirements.txt
```

Depending on your environment, you may also need to install GPU-specific packages separately.

---

## Data Organization

### Training Data

Released training dialogue datasets are placed under:

```text
data/train/
```

Current training files:
- `baize_chat_data_use.jsonl`
- `banking_train_50k_use.jsonl`

These files are intended for running the MDS selection pipeline.

### Evaluation Data

Released evaluation datasets are placed under:

```text
data/eval/
```

Current evaluation files:
- `banking_test_300.jsonl`
- `ConsistentChat_use_270.jsonl`
- `dialogue_test_unseen_use.jsonl`
- `mt_eval_part_use.jsonl`

These files are intended for evaluation or for conversion into disk-style evaluation data.

### Dialogue Format

Each JSONL line stores one dialogue-level sample.  
The default format in this repository is:

```json
{
  "messages": [
    {"content": "I want to know the step by step guide to invest in share market in India.", "role": "user"},
    {"content": "Sure, I can help with that. Firstly, you need to open a demat and trading account with a registered stockbroker.", "role": "assistant"},
    {"content": "How do I find a registered stockbroker in India?", "role": "user"},
    {"content": "You can visit the websites of National Stock Exchange (NSE) or Bombay Stock Exchange (BSE) to get a list of registered stockbrokers in India.", "role": "assistant"},
    {"content": "What documents are required to open a demat and trading account?", "role": "user"},
    {"content": "You will need to provide identity proof (PAN card), address proof, bank details and a passport size photograph to open a demat and trading account.", "role": "assistant"},
    {"content": "How do I start trading once I have a demat and trading account?", "role": "user"},
    {"content": "You can start trading by placing buy and sell orders for stocks through your stockbroker either online or offline.", "role": "assistant"},
    {"content": "How do I track my investments?", "role": "user"},
    {"content": "You can track your investments through your demat account. It will provide you with a consolidated view of all your investments in various stocks and other financial instruments.", "role": "assistant"}
  ],
  "id": "conv_id_0"
}
```

Requirements:
- `messages` must be a list of dialogue turns
- each turn must contain `role` and `content`
- `role` should typically be `user` or `assistant`
- `id` should uniquely identify the dialogue

The released scripts may also support closely related variants, but the above format is the recommended default.

---

## Supported Input Formats

### 1. Dialogue Pool Format

A training dialogue file is expected to be in JSONL format, with one dialogue per line.  
The main format used in this repository is:

```json
{
  "messages": [
    {"role": "user", "content": "User utterance 1"},
    {"role": "assistant", "content": "Assistant reply 1"},
    {"role": "user", "content": "User utterance 2"},
    {"role": "assistant", "content": "Assistant reply 2"}
  ],
  "id": "conv_id_xxx"
}
```

### 2. Global / Local Selection Output Format

The selection scripts keep the original dialogue content and may append additional metadata such as:

- `conv_id`
- `bin_id` / `cluster_id`
- selection-related intermediate fields
- local-stage scores

The exact fields depend on the script configuration and intermediate processing steps.

### 3. Evaluation Input Format

The evaluation script expects a JSONL prediction file.  
Each line should contain at least:

```json
{
  "idx": 0,
  "conv_id": "dialog_001",
  "turn_index": 3,
  "prompt": "<prompt string>",
  "gold": "reference answer",
  "prediction": "model output"
}
```

The evaluation script will append metric outputs to each record and can also save summary statistics.

---

## Method Pipeline

### Step 1. Global-Stage Candidate Construction

The global stage builds dialogue representations from user turns, clusters dialogues into semantic bins, and performs bin-wise candidate selection with coverage and redundancy control.

Example:

```bash
python mds_global_stage.py \
  --input_jsonl data/train/baize_chat_data_use.jsonl \
  --output_jsonl outputs/mds_global_candidates.jsonl \
  --output_ids_json outputs/mds_global_candidate_ids.json \
  --n_clusters 1000 \
  --retain_ratio 0.5 \
  --min_keep_per_cluster 5 \
  --lambda_mmr 0.5
```

Main output:
- candidate dialogue JSONL
- selected dialogue id list in JSON

---

### Step 2. Local-Stage Final Selection

The local stage refines the candidate pool with dialogue-level structural scoring and performs budgeted final selection.

Example:

```bash
python mds_local_stage.py \
  --input_jsonl outputs/mds_global_candidates.jsonl \
  --output_jsonl outputs/mds_final_selected.jsonl \
  --output_ids_json outputs/mds_final_selected_ids.json \
  --judge_model /path/to/judge_model \
  --budget 10000 \
  --tau_form 1.0
```

Main output:
- final selected dialogue JSONL
- final selected dialogue id list in JSON

---

### Step 3. Convert Train/Eval Data into Disk-Style Format

The conversion script expands multi-turn dialogues into per-turn samples and stores them in disk-style format for downstream training or evaluation.

Example:

```bash
python build_disk_data.py
```

Before running, please update the paths in the script, including:
- model path
- input JSONL files
- output root directory

By default, the conversion script is designed to save:
- converted training sets under `train/`
- converted evaluation sets under `eval/`

inside the configured output root directory.

---

### Step 4. Model Training

After dialogue selection, the selected subsets are expanded into turn-level supervised samples and used for standard chat-style supervised fine-tuning. You can use any common SFT / LoRA training framework for this step.

A typical training stage looks like:

1. Run MDS global-stage selection on the original dialogue pool
2. Run MDS local-stage selection to obtain the final selected subset
3. Convert the selected dialogues into disk-style train data
4. Fine-tune a backbone chat model on the processed turn-level samples
5. Run inference on the evaluation sets
6. Evaluate the prediction files with `mds_dialogue_eval.py`

---

### Step 5. Run Model Inference

You can use any standard chat inference pipeline to generate predictions from the converted evaluation data.

The final prediction file should follow the evaluation JSONL format described above.

---

### Step 6. Run Unified Evaluation

The evaluation script provides a unified interface for several metrics.

Example:

```bash
python mds_dialogue_eval.py
```

Before running, please update the relevant settings in the script, such as:
- OpenAI-compatible API configuration
- evaluation input file paths
- output file paths
- enabled / disabled metric switches
- embedding model path if needed

---

## Evaluation Metrics

The unified evaluation script supports the following metrics:

### GPT-based Metrics
- **LLM-EVAL**
  - content
  - grammar
  - relevance
  - appropriateness
  - overall

- **G-EVAL**
  - coherence
  - naturalness
  - engagement
  - groundedness
  - overall

- **Entity-F1**
  - GPT-based entity extraction
  - micro F1 over extracted entities

### Local / Non-GPT Metrics
- **Embedding cosine similarity**
  - sentence-transformers based

- **BLEU-3**
  - sacreBLEU based

- **ROUGE-L**
  - rouge-score based

### Notes
- GPT-based metrics require a valid API setup.
- Depending on the model endpoint and rate limits, GPT-based evaluation may take time on large test sets.
- Slight score variations can occur across different API backends or judge models.

---

## Quick Start

A minimal end-to-end workflow is:

```bash
# 1. Global-stage selection
python mds_global_stage.py \
  --input_jsonl data/train/baize_chat_data_use.jsonl \
  --output_jsonl outputs/mds_global_candidates.jsonl \
  --output_ids_json outputs/mds_global_candidate_ids.json

# 2. Local-stage selection
python mds_local_stage.py \
  --input_jsonl outputs/mds_global_candidates.jsonl \
  --output_jsonl outputs/mds_final_selected.jsonl \
  --output_ids_json outputs/mds_final_selected_ids.json \
  --judge_model /path/to/judge_model \
  --budget 10000 \
  --tau_form 1.0

# 3. Convert selected train/eval data into disk format
python build_disk_data.py

# 4. Train your model with any standard SFT / LoRA framework

# 5. Run inference on evaluation data

# 6. Evaluate prediction files
python mds_dialogue_eval.py
```

---


## Script Notes

### `mds_global_stage.py`
Main functionality:
- read dialogue-level JSONL files
- extract user-query trajectory representations
- compute dialogue embeddings
- cluster dialogues into semantic bins
- perform bin-wise MMR-style candidate selection

### `mds_local_stage.py`
Main functionality:
- normalize dialogue format
- extract QA pairs from dialogues
- run turn-level form/style analysis with a judge model
- compute local structural signals
- perform bin-wise budgeted final selection

### `build_disk_data.py`
Main functionality:
- read selected train and evaluation dialogue files
- expand each dialogue into turn-level supervised samples
- tokenize prompts and labels
- save processed datasets in disk-style format

### `mds_dialogue_eval.py`
Main functionality:
- read prediction JSONL files
- compute GPT-based and local automatic metrics
- save per-sample results
- save or print aggregated results

---

## Common Issues

### 1. API timeout during evaluation
GPT-based evaluation may timeout on large batches or under strict rate limits.

Suggested fixes:
- reduce worker count
- lower the request concurrency
- retry failed samples
- increase timeout settings carefully

### 2. Missing model path in conversion script
Please update the tokenizer / model path in the conversion script before running it.

### 3. Input format mismatch
If your dialogue JSONL does not follow the recommended `messages` + `role` + `content` + `id` format, you may need a lightweight preprocessing step before running the pipeline.

### 4. Judge model mismatch
For more consistent reproduction, keep the local-stage judge model fixed across runs.

### 5. Training and prediction code is not included
This is expected. The repository releases the selection-specific and evaluation-specific parts of the project, while the training and prediction stages can be reproduced with standard chat fine-tuning frameworks.

---

## Citation

Please update this section with the final camera-ready citation.



---

