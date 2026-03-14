#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE="${EXPERIMENT_STAGE:-recommended}"
RUN_CONTROLLED=1
RUN_REALISTIC=1
RUN_JUDGE=1
RUN_ANALYZE=1
RUN_FIGURES=0
RUN_REPORT=0
ANALYSIS_SUFFIX=""
JUDGE_LIMIT=""
JUDGE_SKIP_EXISTING=0
SKIP_EXISTING_OUTPUTS_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_v2_pipeline.sh [options]

Default pipeline:
  1. controlled suite (drift -> summary -> sweep)
  2. realistic group1-group7
  3. realistic LLM judge
  4. aggregate analysis and export two final CSV tables

Options:
  --stage <debug|pilot|formal|recommended>
  --skip-controlled
  --skip-realistic
  --skip-judge
  --skip-analyze
  --with-figures
  --with-report
  --skip-figures
  --skip-report
  --analysis-suffix <name>
  --judge-limit <n>
  --judge-skip-existing
  --keep-existing-outputs
  --overwrite-existing-outputs
  -h, --help

Examples:
  bash scripts/run_v2_pipeline.sh --stage debug --judge-limit 20 --analysis-suffix debug
  bash scripts/run_v2_pipeline.sh --stage recommended
  bash scripts/run_v2_pipeline.sh --skip-judge --skip-report
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ $# -ge 2 ]] || { echo "Missing value for --stage" >&2; exit 1; }
      STAGE="$2"
      shift 2
      ;;
    --skip-controlled)
      RUN_CONTROLLED=0
      shift
      ;;
    --skip-realistic)
      RUN_REALISTIC=0
      shift
      ;;
    --skip-judge)
      RUN_JUDGE=0
      shift
      ;;
    --skip-analyze)
      RUN_ANALYZE=0
      shift
      ;;
    --with-figures)
      RUN_FIGURES=1
      shift
      ;;
    --with-report)
      RUN_REPORT=1
      shift
      ;;
    --skip-figures)
      RUN_FIGURES=0
      shift
      ;;
    --skip-report)
      RUN_REPORT=0
      shift
      ;;
    --analysis-suffix)
      [[ $# -ge 2 ]] || { echo "Missing value for --analysis-suffix" >&2; exit 1; }
      ANALYSIS_SUFFIX="$2"
      shift 2
      ;;
    --judge-limit)
      [[ $# -ge 2 ]] || { echo "Missing value for --judge-limit" >&2; exit 1; }
      JUDGE_LIMIT="$2"
      shift 2
      ;;
    --judge-skip-existing)
      JUDGE_SKIP_EXISTING=1
      shift
      ;;
    --keep-existing-outputs)
      SKIP_EXISTING_OUTPUTS_OVERRIDE=1
      shift
      ;;
    --overwrite-existing-outputs)
      SKIP_EXISTING_OUTPUTS_OVERRIDE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${STAGE}" in
  debug|pilot|formal|recommended)
    ;;
  *)
    echo "Invalid stage: ${STAGE}" >&2
    usage >&2
    exit 1
    ;;
esac

export EXPERIMENT_STAGE="${STAGE}"
if [[ -n "${SKIP_EXISTING_OUTPUTS_OVERRIDE}" ]]; then
  export SKIP_EXISTING_OUTPUTS="${SKIP_EXISTING_OUTPUTS_OVERRIDE}"
fi

cd "${REPO_ROOT}"

run_cmd() {
  echo
  echo "[run] $*"
  "$@"
}

print_profile() {
  python - <<'PY'
import config
print(
    "[profile] "
    f"stage={config.EXPERIMENT_STAGE} "
    f"controlled_sample_size={config.LONGMEMEVAL_CONTROLLED_SAMPLE_SIZE} "
    f"controlled_sweep_sample_size={config.LONGMEMEVAL_CONTROLLED_SWEEP_SAMPLE_SIZE} "
    f"controlled_summary_sample_size={config.LONGMEMEVAL_CONTROLLED_SUMMARY_SAMPLE_SIZE} "
    f"realistic_sample_size={config.LONGMEMEVAL_REALISTIC_SAMPLE_SIZE} "
    f"repeats={config.LONGMEMEVAL_REPEATS} "
    f"window_sessions={config.LONGMEMEVAL_WINDOW_SESSIONS} "
    f"drift_keep_sessions={config.LONGMEMEVAL_DRIFT_KEEP_SESSIONS} "
    f"acf_k_values={config.LONGMEMEVAL_ACF_K_VALUES} "
    f"skip_existing_outputs={config.SKIP_EXISTING_OUTPUTS}"
)
PY
}

print_profile

if [[ "${RUN_CONTROLLED}" -eq 1 ]]; then
  run_cmd python -m experiments.v2_controlled_asymmetry
fi

if [[ "${RUN_REALISTIC}" -eq 1 ]]; then
  for group_id in 1 2 3 4 5 6 7; do
    run_cmd python -m "experiments.v2_group${group_id}"
  done
fi

if [[ "${RUN_JUDGE}" -eq 1 ]]; then
  judge_cmd=(python scripts/score_v2_llm_judge.py --experiment realistic)
  if [[ -n "${JUDGE_LIMIT}" ]]; then
    judge_cmd+=(--limit "${JUDGE_LIMIT}")
  fi
  if [[ "${JUDGE_SKIP_EXISTING}" -eq 1 ]]; then
    judge_cmd+=(--skip-existing)
  fi
  run_cmd "${judge_cmd[@]}"
fi

if [[ "${RUN_ANALYZE}" -eq 1 ]]; then
  analyze_cmd=(python scripts/analyze_v2_outputs.py --experiment all)
  if [[ -n "${ANALYSIS_SUFFIX}" ]]; then
    analyze_cmd+=(--output-suffix "${ANALYSIS_SUFFIX}")
  fi
  run_cmd "${analyze_cmd[@]}"
fi

if [[ "${RUN_FIGURES}" -eq 1 ]]; then
  figure_cmd=(python scripts/generate_v2_figures.py)
  if [[ -n "${ANALYSIS_SUFFIX}" ]]; then
    figure_cmd+=(--suffix "${ANALYSIS_SUFFIX}")
  fi
  run_cmd "${figure_cmd[@]}"
fi

if [[ "${RUN_REPORT}" -eq 1 ]]; then
  report_cmd=(python scripts/report_v2_tables.py)
  if [[ -n "${ANALYSIS_SUFFIX}" ]]; then
    report_cmd+=(--suffix "${ANALYSIS_SUFFIX}")
  fi
  run_cmd "${report_cmd[@]}"
fi

echo
echo "[done] V2 pipeline finished."
