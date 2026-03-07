# SPL2026Priv

## Project Layout

- `data/`: datasets and processed data (`raw/`, `processed/`)
- `core/agent/`: agent core logic
- `core/tools/`: utility modules (RAG, etc.)
- `experiments/`: runnable experiment entry scripts
- `data/outputs/`: experiment outputs (grouped by subfolders: `group1` ... `group5`)
- `data/llm_scores/`: model scoring results
- `data/table/`: generated tables/PDF reports
- `data/docs/`: generated starter docs
- `data/index/`: vector index artifacts

## Configuration

Runtime config is Python-based in `config.py`.

API credentials are read from `.env`/environment variables:
`OPENAI_API_KEY`, `OPENAI_BASE_URL`.

## Run Experiments

Run Group3 (METEOR):

```bash
python -m experiments.group3
```

Run Group4 (asymmetric + context-consistent):

```bash
python -m experiments.group4
```

Run Group5 (asymmetric + context-mismatch + RAG):

```bash
python -m experiments.group5
```

## Outputs

Experiment outputs JSON records to:

- `data/outputs/group1/` ... `data/outputs/group5/`

## Scripts

Prepare RAG corpus before running Group5:

```bash
# Incremental generate: skip existing docs, only fill missing sampled starters
python scripts/prepare_rag_corpus.py generate

# Build index only
python scripts/prepare_rag_corpus.py index

# One-shot: generate missing docs then build index
python scripts/prepare_rag_corpus.py all
```

Analyze experiment outputs:

```bash
# LLM score only
python scripts/analyze_experiment_outputs.py score

# Build coherence report PDF
python scripts/analyze_experiment_outputs.py coherence

# Build metrics comparison PDF
python scripts/analyze_experiment_outputs.py metrics

# End-to-end analysis
python scripts/analyze_experiment_outputs.py all
```
