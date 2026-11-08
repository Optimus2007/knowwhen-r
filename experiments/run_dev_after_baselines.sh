#!/bin/zsh
# Reuses saved predictions to add the always-retrieve comparison, then reruns the combinations on the reranked tags.
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh
while pgrep -f eval_baselines.py >/dev/null; do sleep 10; done
KW_MODEL=tiny-aya-global-t4-rerank python scripts/eval_policy.py --tag t4_rerank --reuse > experiments/logs/eval_policy_dev_t4_rerank_reuse.log 2>&1
KW_MODEL=aya-expanse-8b-rerank python scripts/eval_policy.py --tag aya8b_rerank --reuse > experiments/logs/eval_policy_dev_aya8b_rerank_reuse.log 2>&1
KW_MODEL=tiny-aya-global-t4-rerank python scripts/explore_combinations.py > experiments/logs/explore_combinations_t4_rerank.log 2>&1
KW_MODEL=aya-expanse-8b-rerank python scripts/explore_combinations.py > experiments/logs/explore_combinations_aya8b_rerank.log 2>&1
