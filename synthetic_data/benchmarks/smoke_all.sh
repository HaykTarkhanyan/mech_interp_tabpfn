#!/usr/bin/env bash
# Tiny runs of every runner, one after another, to catch errors before the full runs (~5 min in total).
# Usage (from synthetic_data/benchmarks/):  bash smoke_all.sh      Writes results/*_smoke.json and ../logs/bench_*_smoke.out
cd "$(dirname "$0")"
specs=(
  "run_realcause.py --n-samples 2 --tag realcause_smoke"
  "run_xaibench.py --datasets gaussianLinear --rhos 0.5 --n-val 10 --tag xaibench_smoke"
  "run_iterpretability.py --scales 1 --n-iter 100 --explainer-limit 20 --tag iterp_smoke"
  "run_fb2ts_tabular.py --epochs 200 --n-explain 30 --tag fb2ts_smoke"
  "run_wgan.py --reps 2 --n-truth 5000 --tag wgan_smoke"
  "run_credence.py --max-epochs 1 --reps 2 --tag credence_smoke"
)
for spec in "${specs[@]}"; do
  tag=${spec##* }
  start=$(date +%s)
  timeout 900 .venv/Scripts/python.exe -u $spec > "../logs/bench_$tag.out" 2>&1
  code=$?
  echo "$(date +%H:%M:%S) $tag exit $code after $(( $(date +%s) - start ))s"
done
