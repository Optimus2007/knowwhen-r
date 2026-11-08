# Analysis plan

Fixed on 5 November 2026, after seeing about 1,800 train and validation answers and before any test result. Anything added later will be labelled exploratory. Test data has not been used for any decision so far.

## What the early look showed

On 361 validation questions, mean token log probability reached AUROC 0.879 with no training, and activation probes reached about 0.88. These pooled figures mix datasets of different difficulty, so they overstate within dataset skill. Closed book accuracy was 14 to 41% by dataset.

## Primary questions and tests

H1, transfer. Train on three datasets and test on the fourth, for all four. Compare with in domain probes on the same test questions. Primary metric: AUROC computed within each dataset. Report AUPRC, ECE and Brier beside it.

Probe value. Compare activation probes against mean log probability, minimum log probability, mean entropy, a logistic model on all token statistics, and self consistency on the test set if time allows. Also test activations plus token statistics together. The question is whether activations add anything beyond token statistics, tested with paired bootstrap over questions.

H2, decisions. For each cost setting, compare on test: closed book only; RAG always; fixed threshold abstention; fixed threshold retrieval; two tuned thresholds on the probe score; two tuned thresholds on log probability; the value policy; the oracle. Thresholds are tuned on validation data at the same costs. Primary metric: mean utility, with a paired bootstrap against the tuned threshold policies.

H3, retrieval efficiency. Report wrong answer rate, coverage and retrieval rate. Compare against RAG always at matched utility.

## Fixed settings

Costs: correct +1, wrong penalty in {1, 2, 4}, abstention 0, retrieval cost in {0.05, 0.1, 0.2}. The headline setting is wrong 2, retrieval 0.1, chosen now, before seeing results.

Layer, position and regularization are chosen on validation data only, with the choice made inside each leave one out fold. For OOD folds, validation data comes only from the three training datasets, so the held out dataset is never used for selection or calibration.

Seeds: splits use 13, probes use 0, 1, 2. Uncertainty is shown with 2,000 bootstrap resamples over questions.

## Checks that could undermine the results

Dataset source prediction from the same activations, reported beside probe scores. Within dataset AUROC throughout. A label audit on a random sample, with the auditor stated. PopQA results split by popularity to see whether probes only catch failures of recall.

## Reporting rule

Every comparison above is reported whatever it shows, including any where the probe or policy loses. Failed runs stay in the experiment registry.

## Addendum on scoring, 5 November 2026

Written before any RAG answer was generated and before any policy was scored, so nothing here was chosen after seeing a policy result. The original plan left these points open.

A refusal by the model counts as an abstention, not as a wrong answer, unless the text also contains a gold answer. Counting refusals as wrong would penalise the always answer baselines and favour any policy that abstains. On 14,912 closed book answers the refusal rate was 0.02%, and the RAG rate is not yet known.

"Fixed threshold" means a cut of 0.5 on the calibrated closed book probe, which ignores costs. Single thresholds tuned on validation at the same costs are reported beside them as exploratory rows, since they are the stronger version of the same baseline.

Oracle utility recovered is (policy utility minus best fixed action) divided by (oracle utility minus best fixed action), where the fixed actions are answer everything, retrieve everything and abstain on everything. The best fixed action is picked on the evaluation data, which makes the measure harder to pass. Values below zero mean the policy lost to that fixed action.

H3 matched comparison. With a wrong answer penalty, abstaining alone can beat RAG always, so a matched comparison that allows abstention says little about retrieval. The comparison therefore uses answer or retrieve only: the fewest retrievals, with the cut chosen on validation data, that match RAG always in utility, and separately in correct answer rate.

Exploratory rows, not part of the primary tests: two thresholds on entropy and on the token statistics model; the value policy built from token statistics instead of activations, which asks whether activations add anything to the decision itself; the value policy without the post retrieval probe; and, on OOD folds, the value policy with probes recalibrated on validation rows of the held out dataset. That last row is a diagnostic of what miscalibration costs. It uses held out labels, so it is not a usable method and never counts as a result for the policy. Test rows are not used for it.

The primary test stays the value policy against the two tuned threshold policies at the headline costs. Other cost settings are reported with their intervals but are not treated as separate confirmations.

## Addendum on hardware, 5 November 2026

Decided before any policy was scored and with the test split still unused. Tiny Aya closed book and RAG generation both move to one setup, Kaggle T4 x2 in float16, stored as `tiny-aya-global-t4`, because RAG on the Mac would take about 15 hours and strain its memory. Keeping both conditions on one setup avoids mixing hardware inside a single comparison. On 64 questions, T4 float16 matched the Mac's bfloat16 labels on 63 of 64 closed book and 64 of 64 RAG answers. The Mac closed book run is kept as a full hardware cross check. Development numbers reported so far came from the Mac run and will be measured again.

## Exploratory addition on coverage, 5 November 2026

Added after the first end to end test on 68 development questions, so this is exploratory and does not replace any planned comparison. The headline costs stay as they were. At a wrong answer penalty of 2, answering needs at least 67% confidence, and a model that is right 14 to 41% of the time is then told to abstain on nearly everything. The hackathon asks for fewer wrong answers while still answering most questions, which the utility metric alone does not show.

Each policy answers its most confident 50, 70 or 90% of questions, using a validation quantile as the cut, and reports the coverage it actually reaches on evaluation rows. Compared are closed book answers ranked by probe, log probability and token statistics; RAG answers ranked by the post retrieval probe; and routing, which sends a question to retrieval when the forecast for the retrieved answer beats the closed book estimate, then ranks by the probe for whichever answer it ends with. Routing is run on both probes and token statistics. Wrong answer rates are compared with the same paired test. Because realized coverage can differ between policies, wrong answer rates must be read with coverage beside them.

## Exploratory combinations, 5 November 2026

Written before running them. All of this is exploratory, runs on development rows only, and does not change any planned comparison. The test split is not loaded by `scripts/explore_combinations.py`.

Variants differ only in the features behind two calibrated estimates, one for the closed answer and one for the RAG answer: single layer probe; layer band average in the style of LayerMix; token statistics from both passes; cross path checks, meaning whether the two answers agree and whether each appears in the retrieved passages, none of which use gold answers; combinations of those; and a probe on the difference between RAG and closed activations. Every question goes to the likelier answer and the most confident 50, 70 or 90% are answered. All these systems pay for retrieval on every question, and are scored that way. Cuts, bands and combiners are fitted on validation rows.

The comparison that decides whether activations matter is `band_stats_cross` against `stats_cross`, which has the same pipeline without activations. If activations add nothing there, the report will say so.

Because these variants are compared on the development evaluation rows, one configuration will be frozen before the test run, and its test result is the only number that counts. That configuration is added to the final evaluation as a labelled exploratory system and does not replace the planned systems.

Also checked: whether a conformal cut keeps the wrong answer rate under its bound in domain and on held out datasets, and whether label free prior correction reduces calibration error on held out datasets.

## Retrieval change, 6 November 2026

Decided from validation rows only, before any policy result on reranked passages existed. Error analysis on development rows showed that RAG was right on 76% of questions when the answer was in the top three passages and on 6% when it was not, and that the lowest wrong answer rate reachable at 70% coverage was close to what routing already achieved. So retrieval quality, not detection, was the limit. On 300 validation questions, a cross encoder (ms-marco-MiniLM-L-6-v2) over the keyword top 100 raised answer presence in the top three from 37.7% to 54.3%. Reranking uses no labels and treats every split the same way. Keyword retrieval results are kept and reported as an ablation, so the effect of retrieval quality is visible. The test split remains unused.

## Configuration frozen for the test run, 6 November 2026

Written after all development results and before the test split was used for anything. The earlier planned comparisons all stay and are reported whatever they show.

The system carried to test is the single layer linear activation probe, with layer, position and regularization chosen on validation rows inside each fold. It is run on reranked passages. Development results gave no reason for anything larger: on both models at 50, 70 and 90% coverage, the layer band, token statistics and cross path variants did not beat one layer by a reliable margin, in domain or on any held out dataset, and the MLP was lower than the linear probe. These variants are not run on test. The combination rows in the report are development results and will be labelled that way.

The hackathon statement asks for fewer wrong answers while still answering most questions, so the coverage result will lead the write up. This is a change of emphasis made after seeing development data, and it will be stated as such. The coverage comparison began as exploratory, and it is still reported beside the primary utility tests, not in place of them.

Coverage tests for test data are fixed now as one family of six, at the 70 and 90% targets: routing against the closed book probe, against always retrieve with the post retrieval probe, and against routing on token statistics, using in domain rows and the wrong answer share. Each model is corrected separately with Holm's method. Results at 50% coverage and on held out datasets are reported with intervals and are not treated as confirmations. Development results already showed that routing's lead over always retrieve is small, so the expected claim for routing is fewer retrievals at similar error. If the test result differs, the report will follow the test result.

Systems are compared at the coverage they actually reach on evaluation rows, which differs slightly from the target, and that coverage is printed beside every rate. Runs use seeds 0, 1 and 2 on Tiny Aya and Aya Expanse 8B with reranked passages. Keyword retrieval is run on Tiny Aya only as an ablation, because the 8B keyword run did not finish. P(Yes) and self consistency run on the same test rows, and self consistency is scored on rows with samples for every column. Any fix needed after the test run will be logged, and both sets of numbers kept.

## Stability check added after the test run, 6 November 2026

The three probe seeds gave identical results, because the logistic fit is deterministic. So seeds measured nothing. To see how much results move with the training rows, the probes are refit on five random 90% subsets of the training rows, with layer, position and strength held at the frozen in domain choice, and the coverage comparisons are rescored on test. Nothing about the system changes as a result of this check, and it is labelled as added after the test run.

## Exact coverage check added after the test run, 6 November 2026

The planned cut comes from validation rows of other datasets, so on a held out dataset the systems reach different coverage, for example 58% to 72% at the 70% target on HotpotQA with the 8B model. Comparing wrong answer rates across those rows mixes ranking quality with coverage. `scripts/matched_unseen.py` sets each cut at a quantile of that system's confidence on the evaluation rows, which uses no labels and gives the same coverage to every system. Held out comparisons in the report use these numbers, and also state the coverage the planned cut really reached, since that drift is itself a limitation. In domain confirmatory tests stay as fixed in the freeze, with the exact coverage figures printed beside them.

## Online cut, added 8 November after the test run

Purpose: fix the coverage drift reported in the exact coverage section, without labels. Each question's cut is the (1 minus target) quantile of the confidence of the previous W questions. The stream starts with validation rows from the training datasets, then switches to the held out dataset, in 20 random orders. Script: `scripts/online_cut.py`.

Choice of W, written before any development numbers were seen. Candidates are 50, 100, 200, 400 and 800. Using development tables only (validation rows, no test rows), take the W with the smallest mean absolute gap between coverage reached and target for the routed probe system, averaged over the four held out folds, both models, and targets 0.7 and 0.9. If several are within 0.005 of the best, take the largest of them, since it is the most stable. That single W is then run once on the test tables for all four systems and reported next to the planned cut and the exact batch cut. These results are labelled as added after the test run, and no confirmatory claim rests on them.

Development result (logs `experiments/logs/online_cut_dev_*.log`): mean absolute coverage gap for the routed probe was 0.0131 at W=50, 0.0132 at 100, 0.0206 at 200, 0.0352 at 400 and 0.0458 at 800, against 0.0549 for the planned cut. By the rule above W=100 is fixed for the test run.

## Multilingual transfer, added 8 November after the test run, written before any generation

Question: do the frozen English trained closed book probes still rank answers by correctness when the question is in another language? Data: `scripts/prepare_mkqa.py`, 600 MKQA questions in English, Spanish, German, French, Russian and Turkish, entity and short phrase answers only, questions already in our set removed, gold answers in the question's language plus English. Generation: closed book only, same prompt template with only the question translated, float16 on Kaggle T4 pairs, both models (`experiments/kaggle/run_mkqa.ipynb`). A 12 prompt local check showed Tiny Aya often answers foreign questions in English, which is why English gold answers count.

Analysis, fixed now: the in domain closed answer probe exported in `artifacts/probes/<tag>/closed_answer.npz` is applied unchanged, with no refit and no recalibration. Per language we report accuracy, refusal rate (English refusal patterns only, a known gap), AUROC of the probe, AUROC of mean log probability, and AUROC of the token statistics model refit on the original training rows, with 2,000 bootstrap intervals over questions. The primary comparison is each language against English on the same 600 questions. Secondary: wrong answer rate at 70% answered using the online cut from E028. No result here changes the frozen system or any earlier claim.

Run note, 8 November, before any multilingual results: Kaggle would not pass the Hugging Face token to the run, so the weights now come from Cohere's own uploads on Kaggle Models (`cohere-labs/tiny-aya` variant global, `cohereforai/aya-expanse` variant 8b). Each weight file is checked against the SHA256 of the Hugging Face revision used for every other run (`configs/model_weight_hashes.json`), and the run stops on any mismatch. Config and tokenizer files are copied from that same revision, so the chat template and cached preamble are unchanged.
