# Progress log

## 5 November 2026

Verified the drive and read the official rules. Cloned the four toolkits and read their documentation and relevant source.

The first environment attempt failed: installers broke on exFAT sidecar files, and a wrapper script hid this behind a success code. Moved the environment into an APFS image stored on the SSD, then confirmed all three toolkits import and MPS works on torch 2.15.0 and transformers 5.22.1.

Reviewed the literature from primary sources. The original novelty framing did not survive: Probing-RAG already covers probe triggered retrieval, retrieval aware training data and OOD tests. Narrowed the claim.

Built 16,400 questions with cross dataset deduplication. Wrote retrieval, generation, probe, policy and metric code. 21 tests pass, including capture tests on a tiny random Cohere2 model. No results on a real model exist yet.

Later on 5 November, model access arrived. Decoding was slow at first. Two guesses about the cause were tested and proved wrong before the real one was found: Cohere's chat template adds a 361 token preamble to every prompt. Caching it gave a 3.8 times speedup, with results matching to rounding error in a float32 test. SQLite retrieval was too slow, so the index was rebuilt with Tantivy. A duplicate ID bug let 1 test question and 12 validation questions overlap with other splits. This was found through a row count mismatch and fixed with `scripts/finalize_splits.py`.

Early validation numbers show token log probability as strong as activation probes, and closed book accuracy of only 14 to 41%. The analysis plan was written before any test results were seen.

LLM based label audit on 180 train and validation answers, not a human audit: no false positives among 41 labelled correct, and 3 to 6% of the 139 labelled wrong are disputable. Built the AuditKIT replay. Found that AuditKIT's paired bootstrap returns p near 0.5 for any effect size, and replaced it for significance testing. Decided to keep Tiny Aya on the Mac and save the Brev credit for one Aya Expanse 8B run.

## 5 November 2026, policy evaluation

Wrote scripts/eval_policy.py, src/knowwhen/policies/sweep.py and tests/test_sweep.py. Probe selection moved from scripts/eval_probes.py into src/knowwhen/probes/select.py so both evaluations share it. Reading the label code while designing this showed that model refusals were being scored as wrong answers in the policy code, even though the text module says they are counted apart. That would have flattered any abstaining policy. Refusals now score as abstentions in both the policy arithmetic and the AuditKIT scorer, and the script stops if the two disagree. The scoring rules left open by the plan were added to docs/analysis_plan.md before any RAG answer existed. All 36 tests pass, but the script has not yet run end to end, because it needs RAG rows. Earlier copies of the edited files are in experiments/previous_versions/2026-11-05_before_policy_eval.

Closed book generation finished in 2.3 hours. A readiness check I wrote compared unique IDs with file lines and blocked the RAG start, because the 77 known duplicate lines make those counts differ. The data was fine: every final row is present. RAG was started by hand after confirming this.

Running the probe evaluation next to RAG generation stalled generation for about four minutes and pushed swap to 9 GB, so heavy analysis should not share the machine with a generation pass. RAG then measured 0.27 questions per second alone, slower than the 0.44 estimated from prefill speed, which puts the full pass near 17 hours rather than 10.

RAG was far slower than even the second estimate. Sampling the process showed the GPU busy but Metal holding 13.8 GB on a 16 GB machine, because long prompts at batch 16 keep hidden states for all layers. The first batch had looked fast only because prompts are sorted by length within each block. Generation was stopped at 160 rows and resumed at batch 8, logged in experiments/logs/gen_rag_b8.log. The first 160 rows therefore used batch 16 and the rest batch 8. Batch size only changes padding, which is masked, so answers should match up to rounding.

Prepared the Kaggle replication: multi GPU fixes in the runner, a float16 option, a clean time limit, a stop on non finite activations, a test of the preamble cache on the older Cohere architecture, and a bundle builder with a credential check. See docs/kaggle_run.md. 39 tests pass.

Logged in to Kaggle as optimus2005 and created a private dataset, optimus2005/knowwhen-r-generation-bundle, and a private notebook, optimus2005/knowwhen-aya-8b-generation, saved without running. A rebuild of the bundle crashed on exFAT and the upload that followed used the previous zip. Its contents were harmless but out of date, so the builder now works in the APFS image and version 2 of the dataset was checked byte for byte against the local script.

Kaggle smoke test passed on both models in about 12 GPU minutes, with no non finite activations. The cached preamble left 32 of 32 answers unchanged on Aya Expanse 8B in float16. Secrets only reach runs started from the Kaggle website; two runs pushed from the command line failed at the token step, using about 0.01 GPU hours each. On these 64 questions, RAG refusals were 9% on Tiny Aya and 11% on the 8B model, far above the 0.02% closed book rate, which confirms that the refusal scoring change mattered.

First end to end policy run on 944 partial RAG rows, tag e2e_partial, as a code test only: 68 evaluation questions, and AuditKIT agreed with the policy arithmetic on every replayed policy. It showed that the value policy abstains on everything at the headline costs. We agreed to keep the costs and add an exploratory fixed coverage analysis, recorded in docs/analysis_plan.md.

Tiny Aya closed and RAG finished on Kaggle in 1.8 hours of generation. Every final row is present and finite. The 77 zero rows are the unused earlier copies of duplicate questions, which never appear in the analysis. Across 16,320 final rows, T4 float16 and Mac bfloat16 gave identical text on 92.1% and the same label on 99.6%, with accuracy 0.2005 against 0.2009. On the 944 Mac RAG rows the label match was 99.6%.

First full development results on the T4 data, test split unused. Routed probe at a 70% target: 33.7% wrong at 69% coverage, against 81.6% wrong closed book answering everything. Value policy beat two tuned thresholds in domain by 0.03 utility (p 0.0015), and lost without significance on HotpotQA. Combination experiments (results/exploratory): an unmatched comparison was misleading because systems reached different coverage, so all systems were then compared at equal coverage. At equal coverage the difference probe added nothing, layer averaging added little, cross path checks did not help routing, and activations plus statistics was not reliably better than statistics alone on held out datasets. Conformal cuts held roughly in domain and failed on NQ and HotpotQA. Prior correction was mixed. The unmatched log is kept as explore_combinations_dev_t4_unmatched.log.

The Aya Expanse 8B run finished closed book in 37 minutes for all 16,400 rows, then ran out of GPU memory in RAG at 6,784 rows on one batch. The longest prompts are about 820 tokens, so length alone does not fully explain the 408 MiB request that failed, and I did not establish the exact cause. Hidden states for every layer are held during prefill, and the failing card had 13.4 GiB allocated. Generation now halves a batch that runs out of memory and carries on, which only changes batch composition. The partial BM25 RAG rows are kept but not used for results, since the 8B RAG pass moves to the reranked passages. Added a RAG only mode and one notebook that runs both models in about 4.5 GPU hours.

Aya Expanse 8B closed book output is installed and clean: every final row present, no non finite or zero activations. Closed accuracy on non test rows is PopQA .241, TriviaQA .626, NQ .331, HotpotQA .287, against .146, .360, .135, .153 for Tiny Aya. Labels only were read for this check.

Reranking finished for all 16,323 unique questions in 2.5 hours on the Mac, with three or more passages for every question.

Closed book probes on Aya Expanse 8B, development data, seed 0: activation probe macro AUROC .848 in domain against .813 for token statistics, a wider gap than on Tiny Aya. Mean answer tokens at layer 16 were chosen in every fold. Results are in results/probes/closed_dev_aya8b_seed0.csv. Added self_check.py for the P(Yes) and self consistency baselines, smoke tested on 16 Tiny Aya questions, and added both to the Kaggle notebook.

Tried a one hidden layer MLP against the linear probe on the closed book task, three seeds, with width and penalty picked on validation. It was lower in domain on both models, .871 against .889 on Tiny Aya and .839 against .848 on 8B, and lower on every unseen dataset except HotpotQA on 8B (.836 against .832). The linear probe stays. Scaling each row to unit length changed nothing that matters. AUROC is still rising at the full training size, by about .01 from 5,000 rows to all of them, and on 8B the calibration loss on TriviaQA grows sharply with less data (ECE .325 at 500 rows, .199 at all rows).

AuditKIT's paired bootstrap problem is already reported upstream as issue 20 on Lexsi-Labs/AuditKIT, opened by another user on 6 November, with the same cause and fix. We did not open a duplicate. Our paired_test stays in use until a fixed release is out.

Built 1,918 counterfactual passage pairs from training questions for a CircuitKIT study of how the passage answer reaches the output. On a custom CSV, discovery goes through every row whatever n_examples says, so the subset is now cut before the call. GPT-2 runs the full path in 48 seconds on six pairs. Tiny Aya on MPS did not finish 16 pairs in 25 minutes, so the study moves to Kaggle.

First CircuitKIT run on Kaggle installed and loaded Tiny Aya on Python 3.13, then ran out of GPU memory at batch 2 in the first discovery stage. Log kept as experiments/logs/kaggle_circuit_run1_oom.log. Changed to batch 1 and prompts under 700 characters.

Second CircuitKIT run also ran out of memory at batch 1, by a small margin, in the same TransformerLens per head step. Log kept as experiments/logs/kaggle_circuit_run2_oom.log. Passage windows cut from 160 to 70 characters each side, and the run now falls back to MLP only scope if heads still do not fit.

Third CircuitKIT run ran out of memory in both scopes, so the MLP fallback did not help. Log kept as experiments/logs/kaggle_circuit_run3_oom.log. On GPT-2, leaving only the embedding trainable gave identical node scores, so that change is in, since weight gradients are not needed for attribution.

Fourth CircuitKIT run failed with the same memory figures as the third, so weight gradients were not the cause, or at least not the only one. Log kept as experiments/logs/kaggle_circuit_run4_oom.log. No more retries until the main Kaggle run is finished.

Development analysis on reranked retrieval finished for both models. At about 70% of questions answered, 19.8% of all questions end in a wrong answer on Aya Expanse 8B against 33.3% for the closed book probe, and 28.5% against 50.1% on Tiny Aya. After retrieval the activation probe reaches 0.880 macro AUROC on the 8B model against 0.708 for token statistics. The tables did not yet compare routing with always retrieving plus a probe, so that test was added and is running with the baselines. Test data has not been touched.

Seeds 0, 1 and 2 give identical test results to the last digit, for probes and policies. The logistic probe is a convex fit with a deterministic solver, so `random_state` changes nothing, and the split and bootstrap seeds are fixed. The three seeds therefore say nothing about stability, and the report must not present them as replication. Uncertainty so far comes only from the bootstrap over test questions. The variation caused by the training rows has not been measured yet.

Test run finished. Holm family: 9 of 12 confirmed across the two models. Training subset check shows small movement, under half a point per system. PopQA quartile check shows the probe is not just reading subject popularity. Next are plots and the write up.

Report, abstract, README, cards, licence, requirements and demo are drafted. The SafeTune guard passed its live test, so three toolkits are in use and CircuitKIT stays blocked. Dataset licences for TriviaQA and PopQA were checked against the authors' repositories. Nothing has been published.

## 8 November 2026

Prepared the repository for publication. Doc and code backups and a new private notes file are now excluded by `.gitignore`, and the result backups stay public. Personal names were removed from the docs, and the risk register was brought up to date. The organisers' Track 2 email listed four papers. Two were added to the literature review from their abstracts, and two citations were added to the report. Nothing has been published yet.

The report is now compiled to reports/report.pdf from reports/report.tex with Tectonic, cache kept on the environment image. A new section 4 records what was slow, the bugs found and fixed, and the ideas that did not work, all from this log and the registry. report.md was brought in line with it. Paths in the code are now portable: the project root is the folder holding the code, and the storage check refuses a project on the internal disk. env.sh stays local and env.example.sh is the public template. A Git clean filter replaces local paths in committed logs and docs; the files on disk are unchanged. 54 tests pass. Local repository created with one commit under the GitHub handle and its noreply address. Nothing pushed.

Added the online answer cut (E028). The window rule was written into the analysis plan before the development sweep. Development chose W=100, and a single test run followed. On unseen datasets it fixes the coverage drift: 67.7 to 71.1% reached at a 70% target across four systems, wrong answer rates within 1.3 points of the exact batch cut. Report, abstract and registry updated.

Multilingual transfer (E029). Kaggle would not hand the HF token to two runs, so weights came from Cohere's Kaggle Models uploads with a file by file SHA256 check against the Hugging Face revision; the 8B run also needed a licence acceptance on Kaggle. Both models then ran 3,600 closed book prompts. The English trained probe kept AUROC within 0.035 of English in all five other languages on both models. Report, abstract, README and model card updated.

Publishing prepared, all private. Hugging Face: model repo kungfupanda12/knowwhen-r-probes (eight probes, about 0.3 MB), dataset repo kungfupanda12/knowwhen-r-traces (answers, Parquet predictions, splits, passages, results, logs and figures, about 200 MB), two collections, and a Space. Gradio Spaces now need a paid plan, so the public demo is a static page (demo_static) built from the same traces; its numbers match the Gradio demo and the report. HF Article draft in reports/hf_article.md. Form answers kept locally in submission/, not committed.
