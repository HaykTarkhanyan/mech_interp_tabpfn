#!/usr/bin/env bash
# Run every benchmark's full (small) run one after another, then rebuild report.html.
# One process at a time, at most 2 threads each (common.py). Total ~80 min on this laptop (2026-10-27 estimates).
# Usage (from synthetic_data/benchmarks/, Git Bash):  bash run_all.sh      Progress: ../logs/bench_<tag>.out
cd "$(dirname "$0")"
specs=(
  "run_fb2ts_tabular.py"
  "run_wgan.py"
  "run_credence.py"
  "run_realcause.py"
  "run_xaibench.py"
  "run_iterpretability.py"
  "run_openxai.py --models lr --tag openxai_lr"
  "run_openxai.py --models ann --tag openxai_ann"
)
for spec in "${specs[@]}"; do
  tag=${spec%% *}; tag=${tag#run_}; tag=${tag%.py}
  [[ "$spec" == *"--tag "* ]] && tag=${spec##* }
  start=$(date +%s)
  timeout 1800 .venv/Scripts/python.exe -u $spec > "../logs/bench_$tag.out" 2>&1
  code=$?
  echo "$(date +%H:%M:%S) $tag exit $code after $(( $(date +%s) - start ))s"
done
.venv/Scripts/python.exe report.py && echo "$(date +%H:%M:%S) report.html rebuilt"
