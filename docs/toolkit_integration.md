# Toolkit integration

Track 2 lists CuratorKIT, SafeTune, CircuitKIT and AuditKIT. The rules require at least two used together, and prefer three or more. Installed versions: curatorkit 0.1.21, circuitkit 0.2.0, auditkit 1.1.3, from commits dd9e835, e71d5e1 and 71316da. SafeTune is used from source at 3901a9f through PYTHONPATH.

| Toolkit | Role | Status |
|---|---|---|
| CuratorKIT | Exact and MinHash deduplication across datasets, which is how test contamination is removed. Provenance recorded in the manifest. | In use |
| CircuitKIT | Planned: discover components involved in factual recall on our questions (`Pipeline.from_custom_data`, EAP-IG), and test whether circuit informed layer choice helps the probe. Listed as validated for Tiny Aya and Aya Expanse 8B, but it may not fit in 16 GB. | Not yet run, needs model access |
| AuditKIT | Replays each policy's recorded answers as precomputed outputs through `ak.evaluate` with a custom `Metric`, which derives again correctness independently of the policy code, and aligns runs by sample ID. See `src/knowwhen/evaluation/audit.py`. | Built and tested, awaiting real runs |
| SafeTune | `LinearProbeGuardModel` runs our prompt only probe live. A flagged prompt goes to retrieval through `on_refuse` instead of a refusal. See `scripts/safetune_guard.py`. | In use, E026 |

## A bug found in AuditKIT

`auditkit._bootstrap.paired_bootstrap` in 1.1.3 resamples the differences and counts how often the resampled mean is at least as large as the observed mean. It never centres the differences under the null, so about half of resamples exceed the observed value for any effect. In a check with n = 2000, an effect 90.7 standard errors from zero gave p = 0.50 and `significant: False`. This also affects `compare_models` and `RunComparison.significance`, which call the same function.

We use a bootstrap confidence interval and a sign flip test instead (`paired_test`), and `tests/test_audit.py` checks both false alarm rate and power.

CircuitKIT does not contain a hallucination probe, and SafeTune's guard was built for harmful prompts, so the probes and policy here are our own code, and SafeTune supplies the runtime guard. This file will only claim what ran.

The bootstrap problem described above is tracked upstream as https://github.com/Lexsi-Labs/AuditKIT/issues/20, opened by another user, so the report should cite that issue and not present it as our discovery alone.
