# ACF-Stego

[English](README.md) | [简体中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#1-environment-setup)

Official codebase accompanying the paper on **Asymmetric Collaborative Framework (ACF)** under cognitive asymmetry.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Paper-Aligned Settings](#paper-aligned-settings)
- [Protocol Mapping](#protocol-mapping)
- [Experiment Outputs](#experiment-outputs)

## Directory Structure

```text
ACF-Stego/
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
│   └── analyze_v2_outputs.py        # Generate the two paper tables + one paper figure
├── tests/                           # Unit tests for analysis/metrics tools
├── .env.example                     # Environment variable template (safe placeholders)
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT license
├── README.md                        # English README
└── README.zh-CN.md                  # Chinese README
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n acf-stego python=3.10 -y
conda activate acf-stego
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

```bash
bash scripts/run_v2_pipeline.sh --stage recommended
```

This executes:
- controlled suite (`v2_controlled_asymmetry`)
- realistic suite (`group1` ... `group8`)
- semantic LLM judging for realistic outputs
- aggregation for paper tables and figures

Useful alternatives:

```bash
# Fast debug run
bash scripts/run_v2_pipeline.sh --stage debug --judge-limit 20 --analysis-suffix debug

# Re-run only aggregation artifacts
python scripts/analyze_v2_outputs.py --experiment all
```

## Paper-Aligned Settings

The paper results in this repository align with these settings:
- Dataset split: `longmemeval_s` (cleaned release)
- Base model: `Qwen/Qwen2.5-7B-Instruct` (`MODEL_NAME=QWEN2_5_7B_INSTRUCT`)
- Public context window: `LONGMEMEVAL_WINDOW_SESSIONS=5`
- Controlled truncation mismatch: `LONGMEMEVAL_DRIFT_KEEP_SESSIONS=3` (`drift_recent3`)
- Controlled sweep levels: `LONGMEMEVAL_CONTROLLED_SWEEP_KEEP_SESSIONS=4,3,2,1`
- ACF settings: `LONGMEMEVAL_ACF_K_VALUES=8,12,16`
- Judge model (0/1/2 semantic score): `LLM_JUDGE_MODEL=gemini-2.0-flash`

## Protocol Mapping

`group*` script IDs map to paper protocol names as follows:

| Script Group | Paper Protocol |
| --- | --- |
| `group1` | `Normal (No Stego)` |
| `group2` | `DISCOP` |
| `group3` | `METEOR` |
| `group4` | `ACF` |
| `group5` | `ACF+RET` |
| `group6` | `DISCOP+RET` |
| `group7` | `METEOR+RET` |
| `group8` | `Normal+RET` |

## Experiment Outputs

Main output directories:
- `data/outputs_v2/controlled/`
- `data/outputs_v2/controlled_sweep/`
- `data/outputs_v2/controlled_summary/`
- `data/outputs_v2/realistic/`
- `data/table/v2/`

Primary paper tables (`data/table/v2/`):
- `paper_table_controlled_cognitive_asymmetry.csv`
- `paper_table_realistic_integrated.csv`

Primary paper figures (`data/table/v2/figures/`):
- `figure_ber_vs_decoder_sessions.pdf`
