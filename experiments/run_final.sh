#!/bin/zsh
# The one test run, after the freeze in docs/analysis_plan.md. Stops at the first failure.
set -e
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh
L=experiments/logs
one() {  # model tag seed
  KW_MODEL=$1 python scripts/eval_probes.py --final --condition closed --tag $2 --seed $3 > $L/final_probes_closed_$2_seed$3.log 2>&1
  KW_MODEL=$1 python scripts/eval_probes.py --final --condition rag --tag $2 --seed $3 > $L/final_probes_rag_$2_seed$3.log 2>&1
  KW_MODEL=$1 python scripts/eval_policy.py --final --tag $2 --seed $3 > $L/final_policy_$2_seed$3.log 2>&1
}
base() {
  KW_MODEL=$1 python scripts/eval_baselines.py --final --tag $2 --seed $3 > $L/final_baselines_$2_seed$3.log 2>&1
}
for seed in 0 1 2; do
  one tiny-aya-global-t4-rerank t4_rerank $seed
  one aya-expanse-8b-rerank aya8b_rerank $seed
  base tiny-aya-global-t4-rerank t4_rerank $seed
  base aya-expanse-8b-rerank aya8b_rerank $seed
done
for seed in 0 1 2; do
  one tiny-aya-global-t4 t4_keyword $seed
done
echo finished
