#!/usr/bin/env bash
#
# Generate collect_jobs.txt and submit a SLURM array (one task per country × horizon).
#
# Usage (from repo root):
#   cybench/runs/slurm/submit_collect.sh --list
#   cybench/runs/slurm/submit_collect.sh --no-plot --submit
#   cybench/runs/slurm/submit_collect.sh --plot --mode ready --submit
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_collect.sh [generate_collect_manifest.py options] [--submit] [--array RANGE]

  --submit          Run sbatch after writing the manifest
  --array RANGE     SLURM array range (default: 0-(N-1))

Manifest and collect job options (forwarded to generate_collect_manifest.py):
  --mode ready|all-available|planned
  --country CC      Repeatable
  --horizon eos|mid|qtr|early Repeatable
  --version N       Batch version suffix (e.g. 2 for baselines_DE_eos_v2)
  --plot            Dashboard drill-down PNGs (scatter, temporal; maps are dynamic)
  --no-plot         Metrics + preds + compare_models.html only (default)

Examples:
  # Collect v2 batches only (metrics, parallel SLURM):
  cybench/runs/slurm/submit_collect.sh --version 2 --country DE --horizon eos --no-plot --submit

  # Fast parallel collect (metrics only) for everything on lustre:
  cybench/runs/slurm/submit_collect.sh --no-plot --submit

  # Full collect with plots for ready batches only:
  cybench/runs/slurm/submit_collect.sh --plot --mode ready --submit

  # SHAP: run shap_importance.sh first; collect auto-embeds dashboard data from
  #   {output_root}/shap_importance/{crop}_{CC}_{horizon}/
  # when present (e.g. maize_NL_eos). Re-collect after SHAP finishes if needed.

  # After array completes, publish dashboards (login node):
  poetry run python cybench/runs/analysis/orchestrate_dashboard_publish.py \\
      --version 2 --country DE --horizon eos --stages publish,index --no-plot
EOF
}

if [[ -f "${SLURM_SUBMIT_DIR:-}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${SLURM_SUBMIT_DIR}/cybench/runs/slurm"
elif [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/cybench/runs/slurm/slurm_common.sh" ]]; then
  SLURM_DIR="${REPO_ROOT}/cybench/runs/slurm"
else
  SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="${REPO_ROOT:-$(cd "${SLURM_DIR}/../../.." && pwd)}"
MANIFEST="${SLURM_DIR}/collect_jobs.txt"
GEN_ARGS=()
DO_SUBMIT=false
ARRAY_RANGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) DO_SUBMIT=true; shift ;;
    --array) ARRAY_RANGE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) GEN_ARGS+=("$1"); shift ;;
  esac
done

cd "${REPO_ROOT}"

echo "[STEP 1/2] Writing collect manifest (fast directory scan)..."
poetry run python cybench/runs/analysis/generate_collect_manifest.py \
  -o "${MANIFEST}" \
  "${GEN_ARGS[@]}"

n=$(awk '!/^#/ && NF >= 3 {c++} END {print c+0}' "${MANIFEST}")
if [[ "${n}" -lt 1 ]]; then
  echo "[WARN] Manifest empty — nothing to submit" >&2
  exit 1
fi

if [[ -z "${ARRAY_RANGE}" ]]; then
  ARRAY_RANGE="0-$((n - 1))"
fi

echo "[INFO] ${n} collect task(s); manifest=${MANIFEST}"

if [[ "${DO_SUBMIT}" != true ]]; then
  echo ""
  echo "[WARN] No SLURM jobs submitted — you omitted --submit"
  echo "       To launch the array, re-run with: $0 ${GEN_ARGS[*]} --submit"
  exit 0
fi

echo "[STEP 2/2] Submitting SLURM array ${ARRAY_RANGE} (${n} tasks)..."
mkdir -p output/collect
export JOB_MANIFEST="${MANIFEST}"
job_id=$(sbatch --parsable \
  --job-name=cb_collect \
  --array="${ARRAY_RANGE}" \
  --export=ALL,JOB_MANIFEST="${MANIFEST}" \
  "${SLURM_DIR}/collect_results.sh")
echo "[DONE] Submitted job ${job_id} — check: squeue -u \"\$USER\" -n cb_collect"
