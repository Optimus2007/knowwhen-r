#!/bin/zsh
# Development analysis on the reranked runs. Test rows stay unused.
cd "$(dirname "$0")/.." && [ -f env.sh ] && source env.sh
run() { KW_MODEL=$1 python scripts/eval_probes.py --condition rag --tag $2 2>&1 | grep -v Warning > experiments/logs/eval_probes_rag_dev_$2.log
        KW_MODEL=$1 python scripts/eval_policy.py --tag $2 2>&1 | grep -v Warning > experiments/logs/eval_policy_dev_$2.log; }
run tiny-aya-global-t4-rerank t4_rerank
run aya-expanse-8b-rerank aya8b_rerank
