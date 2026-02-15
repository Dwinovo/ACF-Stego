# SPL2026Priv

## Project Layout

- `data/`: datasets and processed data (`raw/`, `processed/`)
- `core/agent/`: agent core logic
- `core/tools/`: utility modules (mem0 integration, etc.)
- `experiments/`: runnable experiment entry scripts
- `results/`: experiment artifacts (`figures/`, `outputs/`, `logs/`)

## Configuration

Runtime config is Python-based in `config.py`.

API credentials are read from `.env`/environment variables:
`OPENAI_API_KEY`, `OPENAI_BASE_URL`.

## Run Experiments

Run Group3:

```bash
python -m experiments.group3
```

## Outputs

Group3 outputs JSON records to:

- `results/outputs/`
