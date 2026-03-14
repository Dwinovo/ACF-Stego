# SPL2026Priv

## Project Layout

- `config/`: split configuration package
- `config/experiment.py`: experiment hyperparameters
- `config/runtime.py`: model/runtime/API settings
- `config/dataset.py`: dataset version and reproduction metadata
- `config/paths.py`: path constants
- `config/prompts.py`: prompt constants
- `config/models.py`: model enum and model-path helpers
- `core/tools/`: LongMemEval loading, retrieval, QA metrics, analysis helpers
- `experiments/`: V2 experiment entry scripts
- `scripts/score_v2_llm_judge.py`: post-hoc semantic LLM judge
- `scripts/analyze_v2_outputs.py`: aggregate raw outputs into summaries, CSV tables, and plot data
- `scripts/generate_v2_figures.py`: render paper figures from aggregated artifacts (PDF only by default, no baked-in titles)
- `scripts/report_v2_tables.py`: terminal preview for aggregated tables
- `data/raw/`: cached LongMemEval payload
- `data/outputs_v2/realistic/`: realistic cognitive asymmetry outputs by `group1` ... `group7`
- `data/outputs_v2/controlled/`: controlled cognitive asymmetry outputs by `group2` ... `group4`
- `data/outputs_v2/controlled_sweep/`: drift severity sweep outputs by `group2` ... `group4`
- `data/outputs_v2/controlled_summary/`: encoder-only summary asymmetry outputs by `group2` ... `group4`
- `data/table/v2/`: aggregated summaries, CSV tables, and plot json

## Configuration

Runtime config is Python-based in the `config/` package. Existing code still uses `import config`, but the implementation is now split by concern.

API credentials are read from `.env` / environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `REMOTE_EMBED_MODEL`
- `REMOTE_RERANK_MODEL`
- `REMOTE_RERANK_ENDPOINT`
- `LLM_JUDGE_MODEL`

Key experiment knobs:

- `EXPERIMENT_STAGE=debug|pilot|formal|recommended`
- `LONGMEMEVAL_REALISTIC_SAMPLE_SIZE`
- `LONGMEMEVAL_CONTROLLED_SAMPLE_SIZE`
- `LONGMEMEVAL_CONTROLLED_SUMMARY_SAMPLE_SIZE`
- `LONGMEMEVAL_CONTROLLED_SUMMARY_NOTE_MAX_TOKENS`
- `LONGMEMEVAL_CONTROLLED_SWEEP_SAMPLE_SIZE`
- `LONGMEMEVAL_CONTROLLED_SWEEP_KEEP_SESSIONS`
- `LONGMEMEVAL_WINDOW_SESSIONS`
- `LONGMEMEVAL_DRIFT_KEEP_SESSIONS`
- `LONGMEMEVAL_MAX_PROMPT_TOKENS`
- `LONGMEMEVAL_RETRIEVAL_MAX_TOKENS`
- `SKIP_EXISTING_OUTPUTS=0|1` (default `1`)
- `SECRET_BITS_LENGTH` (default `2000`, used as an upper-bound secret reservoir)

Workflow presets:

- `debug`: realistic `20`, controlled `20`, repeats `1`
- `pilot`: realistic `100`, controlled `50`, repeats `2`
- `formal`: realistic `500`, controlled `50`, repeats `3`
- `recommended`: realistic `120`, controlled `50`, repeats `3`

`LONGMEMEVAL_REALISTIC_SAMPLE_SIZE`, `LONGMEMEVAL_CONTROLLED_SAMPLE_SIZE`, and `LONGMEMEVAL_REPEATS` still override the stage defaults if you set them explicitly.

Default behavior keeps existing experiment outputs:

- experiment runs keep existing group JSON outputs and skip records whose target JSON already exists
- LLM judge overwrites previous judge fields unless you pass `--skip-existing`
- aggregation overwrites summaries, CSV tables, and plot data for the same suffix

## Run Experiments

Typical workflow:

1. Debug
   `EXPERIMENT_STAGE=debug`
2. Pilot
   `EXPERIMENT_STAGE=pilot`
3. Formal
   `EXPERIMENT_STAGE=formal`
4. Paper-friendly recommended run for this project
   `EXPERIMENT_STAGE=recommended`

One-command pipeline:

```bash
bash scripts/run_v2_pipeline.sh --stage recommended
```

Debug example:

```bash
bash scripts/run_v2_pipeline.sh --stage debug --judge-limit 20 --analysis-suffix debug
```

Run the full controlled suite:

```bash
python -m experiments.v2_controlled_asymmetry
```

This sequentially runs:

- `controlled`: `no_drift + drift_recent3`
- `controlled_summary`: `summary_only_enc`
- `controlled_sweep`: `recent5 -> recent4 -> recent3 -> recent2 -> recent1`

If you need to rerun only one controlled sub-experiment:

```bash
python -m experiments.v2_controlled_drift_sweep
python -m experiments.v2_controlled_summary
```

Run realistic Group1 baseline:

```bash
python -m experiments.v2_group1
```

Run realistic Group2 (DISCOP):

```bash
python -m experiments.v2_group2
```

Run realistic Group3 (METEOR):

```bash
python -m experiments.v2_group3
```

Run realistic Group4 (asymmetric):

```bash
python -m experiments.v2_group4
```

Run realistic Group5 (asymmetric + retrieval):

```bash
python -m experiments.v2_group5
```

Run realistic Group6 (DISCOP + retrieval):

```bash
python -m experiments.v2_group6
```

Run realistic Group7 (METEOR + retrieval):

```bash
python -m experiments.v2_group7
```

The realistic experiment is the main system evaluation and now keeps only the `no_drift` condition.

## LLM Judge

Score realistic outputs with the semantic judge:

```bash
python scripts/score_v2_llm_judge.py --experiment realistic
```

Score only the no-drift outputs for one group:

```bash
python scripts/score_v2_llm_judge.py --experiment realistic --only-group group6 --condition no_drift
```

The intended paper workflow is:

- `controlled`: no LLM judge; focus on `BER` and `DecodeSuccess`
- `realistic`: run LLM judge and use `LLMJudgeCorrect` / `LLMJudgeScore` as the main task metrics
- `EM/F1`: keep them as legacy benchmark-reference metrics

For stego groups, `SECRET_BITS_LENGTH=2000` is only a reservoir upper bound. Actual protocol comparison uses the consumed prefix reported in each run record via `consumed_bits`, and BER is computed on that actually embedded prefix rather than on the nominal 2000-bit budget.
Capacity is reported as embedded bits per 1k generated tokens (`bits/1kTok`).

## Analysis

Aggregate the full experiment suite into paper artifacts:

```bash
python scripts/analyze_v2_outputs.py --experiment all
```

Generate paper figures:

```bash
python scripts/generate_v2_figures.py
```

Optional terminal preview:

```bash
python scripts/report_v2_tables.py
```

## Outputs

Raw JSON records are written to:

- `data/outputs_v2/controlled/group2/` ... `data/outputs_v2/controlled/group4/`
- `data/outputs_v2/controlled_sweep/group2/` ... `data/outputs_v2/controlled_sweep/group4/`
- `data/outputs_v2/controlled_summary/group2/` ... `data/outputs_v2/controlled_summary/group4/`
- `data/outputs_v2/realistic/group1/` ... `data/outputs_v2/realistic/group7/`

Stego run records now include:

- `secret_bits_budget`
- `consumed_bits`

Aggregated artifacts are written to:

- `data/table/v2/controlled_summary.json`
- `data/table/v2/controlled_summary_summary.json`
- `data/table/v2/realistic_summary.json`
- `data/table/v2/paper_table_controlled.csv`
- `data/table/v2/paper_table_realistic_task.csv`
- `data/table/v2/paper_table_realistic_protocol.csv`
- `data/table/v2/plot_ber_vs_condition.json`
- `data/table/v2/plot_controlled_summary_asymmetry.json`
- `data/table/v2/plot_controlled_drift_severity_sweep.json`
- `data/table/v2/plot_task_correctness_vs_reliability.json`
- `data/table/v2/figures/figure1_controlled_asymmetry_ber.pdf`
- `data/table/v2/figures/figure2_task_vs_communication.pdf`
- `data/table/v2/figures/figure3_ber_vs_drift_severity.pdf`
- `data/table/v2/figures/figure4_token_length_capacity.pdf`
- `data/table/v2/figures/appendix_figure_c_summary_asymmetry_ber.pdf`

`paper_table_realistic_task.csv` is judge-first:

- `LLMJudgeCorrect`
- `LLMJudgeScore`
- `F1`
- `EM`

All table cells are reported as `mean ± std`.
