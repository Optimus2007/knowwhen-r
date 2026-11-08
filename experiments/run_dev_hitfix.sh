#!/bin/zsh
# Reruns the scoring that read keyword passages on reranked tags, and the shared row baselines.
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh
KW_MODEL=tiny-aya-global-t4-rerank python scripts/eval_policy.py --tag t4_rerank --reuse > experiments/logs/eval_policy_dev_t4_rerank_hitfix.log 2>&1
KW_MODEL=aya-expanse-8b-rerank python scripts/eval_policy.py --tag aya8b_rerank --reuse > experiments/logs/eval_policy_dev_aya8b_rerank_hitfix.log 2>&1
KW_MODEL=tiny-aya-global-t4-rerank python scripts/explore_combinations.py > experiments/logs/explore_combinations_t4_rerank_hitfix.log 2>&1
KW_MODEL=aya-expanse-8b-rerank python scripts/explore_combinations.py > experiments/logs/explore_combinations_aya8b_rerank_hitfix.log 2>&1
KW_MODEL=tiny-aya-global-t4-rerank python scripts/eval_baselines.py --tag t4_rerank > experiments/logs/eval_baselines_t4_rerank_shared.log 2>&1
KW_MODEL=aya-expanse-8b-rerank python scripts/eval_baselines.py --tag aya8b_rerank > experiments/logs/eval_baselines_aya8b_rerank_shared.log 2>&1
