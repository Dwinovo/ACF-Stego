# SPL2026Priv

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#1-environment-setup)

An anonymized replication package for SPL 2026 experiments on LongMemEval-based communication asymmetry benchmarks.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Experiment Outputs](#experiment-outputs)
- [Anonymous Release Checklist](#anonymous-release-checklist)

## Directory Structure

```text
SPL2026Priv/
├── config/                          # Runtime, dataset, model, and experiment configs
│   ├── runtime.py                   # API/runtime environment variable bindings
│   ├── dataset.py                   # Dataset source and cache settings
│   ├── experiment.py                # Experimental stage/sample hyperparameters
│   ├── models.py                    # Model alias -> model path/model id resolver
│   └── paths.py                     # Canonical project paths
├── core/tools/                      # LongMemEval utilities, retrieval, and metrics
├── experiments/                     # Main experiment entry points (group1...group8, controlled)
├── scripts/
│   ├── run_v2_pipeline.sh           # One-command experiment pipeline
│   ├── score_v2_llm_judge.py        # Semantic LLM judge for realistic outputs
│   ├── analyze_v2_outputs.py        # Aggregate JSON outputs into paper tables/plots
│   ├── generate_v2_figures.py       # Render publication-ready figures
│   └── report_v2_tables.py          # Terminal preview for summary tables
├── tests/                           # Unit tests for analysis/metrics tools
├── .env.example                     # Environment variable template (safe placeholders)
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT license (Anonymous Authors)
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n spl2026priv python=3.10 -y
conda activate spl2026priv
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your own secrets before running remote APIs:
- `OPENAI_API_KEY`
- `HF_TOKEN`

### 2. Dataset Setup

Official download link (LongMemEval-S cleaned):
- https://huggingface.co/datasets/LIXINYI33/longmemeval-s/resolve/main/longmemeval_s_cleaned.json

Download and place the dataset at the expected path:

```bash
mkdir -p data/raw
wget -O data/raw/longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/LIXINYI33/longmemeval-s/resolve/main/longmemeval_s_cleaned.json"
```

Expected path:
- `data/raw/longmemeval_s_cleaned.json`

### 3. Run Main Experiments

One-command full pipeline (recommended paper profile):

```bash
bash scripts/run_v2_pipeline.sh --stage recommended
```

This runs:
- Controlled suite (`v2_controlled_asymmetry`)
- Realistic suite (`group1` ... `group8`)
- Semantic LLM judging for realistic outputs
- Aggregation for final tables and plot JSON

Useful alternatives:

```bash
# Fast debug run
bash scripts/run_v2_pipeline.sh --stage debug --judge-limit 20 --analysis-suffix debug

# Re-run only aggregation
python scripts/analyze_v2_outputs.py --experiment all

# Generate paper figures from aggregated artifacts
python scripts/generate_v2_figures.py
```

## Experiment Outputs

Main output directories:
- `data/outputs_v2/controlled/`
- `data/outputs_v2/controlled_sweep/`
- `data/outputs_v2/controlled_summary/`
- `data/outputs_v2/realistic/`
- `data/table/v2/`

Key table artifacts (CSV/JSON) are written under `data/table/v2/`, including:
- `paper_table_controlled.csv`
- `paper_table_realistic_task.csv`
- `paper_table_realistic_protocol.csv`

## Anonymous Release Checklist

Before uploading to an anonymous review platform:

1. Secrets
- Keep all keys in environment variables only.
- Ensure `.env` contains placeholders or is excluded from upload.

2. Paths
- Avoid machine-specific absolute paths.
- For local model checkpoints, set environment overrides such as:
  - `MODEL_PATH_QWEN2_5_7B_INSTRUCT`
  - `MODEL_PATH_DEEPSEEK_R1_DISTILL_QWEN_7B`
  - `MODEL_PATH_META_LLAMA_3_1_8B_INSTRUCT`

3. Identity metadata
- `LICENSE` uses `Anonymous Authors`.
- Do not upload `.git/` to anonymous submission systems.
- Confirm there is no author/email metadata in public config files.

4. Documentation style
- Keep README in English as the default public-facing document.
- Make reproduction steps copy-paste ready.
