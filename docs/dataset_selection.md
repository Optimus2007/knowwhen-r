# Dataset selection

The design needs short factual questions with independent gold answers, so correctness never depends on another model's opinion. It also needs several distinct sources, so a dataset can be held out completely.

| Dataset | Licence on the card | Used | Why |
|---|---|---|---|
| NQ-open | CC BY SA 3.0 | Yes | Real search queries with human answers. Few aliases per question (mean 1.4), so string matching will miss some correct answers. |
| TriviaQA (rc.nocontext) | Listed as unknown on the HF card; the authors' repository states Apache 2.0 for code and data (checked 7 Nov) | Yes | Trivia style questions with many aliases (mean 15.8). |
| HotpotQA (distractor) | CC BY SA 4.0 | Yes | Multi hop, so it should be hardest for one shot retrieval. Yes and no answers are dropped, because a coin flip adds accuracy without factual signal. |
| PopQA | Not listed on the HF card; the authors' repository is MIT (checked 7 Nov) | Yes | Templated long tail facts with entity popularity and relation type, which supports the popularity analysis and a relation family holdout. |
| TruthfulQA | Apache 2.0 | No | 817 questions built to trigger misconceptions, with free form answers that do not suit string matching. |
| HaluEval | Apache 2.0 | No | Its labels describe ChatGPT generated hallucinations, not this model's answers. |
| RAGTruth | Not listed on the processed card | No | Annotates other models' outputs, so it cannot label our model's errors. |

Evidence comes from `wikimedia/wikipedia` 20231101.en (CC BY SA 3.0 and GFDL), not from the gold paragraphs that come with some datasets. Using gold paragraphs would make retrieval unrealistically easy and would leave no cases where evidence is missing.

## Splits

From `scripts/prepare_data.py`, seed 13. Each dataset gives 2,500 train, 600 validation and 1,000 test questions, 16,400 in total. Test questions come from official validation splits. Train and validation come from official training splits. PopQA has one split, so it is divided by a hash of the subject, and the check found zero subject overlap between test and the rest.

CuratorKIT's exact and MinHash (0.8) deduplicators run over all datasets together, with test items placed first. Both keep the first occurrence, so a near duplicate is always removed from training, not from test. Of 72,089 candidates, 66,853 survived exact matching and 65,774 survived MinHash. The manifest is `data/processed/questions_manifest.json`.

One bug was found and fixed. `prepare_data.py` filtered by question ID, and TriviaQA repeats some rows under one ID, so 77 repeated IDs survived, leaving 1 question text shared between test and another split and 12 between train and validation. `scripts/finalize_splits.py` now keeps one row per normalized question, preferring test, and all analysis uses `data/processed/final_rows.json`. A check after filtering found no question text in more than one split.

## Labels

An answer counts as correct if a gold alias appears in it as a whole token span after normalization. The labels depend only on the datasets' gold answers. String matching makes errors in both directions, so a sample will be audited and the error rate reported.

The audit will be labelled as what it is, an LLM based label audit, not a human one.

## Label audit

180 closed book answers from train and validation (45 per dataset, seed 7, saved in `data/processed/label_audit_sample.json`) were read against their gold answers. This was an LLM based label audit, not a human one, and the test set was not used.

41 answers were labelled correct, and none of those looked wrong. With 41 items, that still allows a false positive rate of up to about 7%. Of the 139 labelled wrong, 4 are clearly correct: "atmospheric sciences" for the WMO's remit, "Xerxes I" against gold "Persians", a reply naming Yucca Mountain, and "Cumbria" against gold "England". Another 4 are borderline: "California" against "Los Angeles", "Netherlands" against "United Kingdom of the Netherlands", "1920s" against "early 20th century", and "photographer" against "cinematographer".

So about 3 to 6% of wrong labels are disputable, and accuracy in the sample is likely 25 to 27%, not the measured 23%. The bias runs one way, so reported accuracy is slightly pessimistic. NQ also contains time sensitive questions whose gold answers have aged, such as "when does the next season of Grey's start".

Added 8 November: MKQA (Longpre, Lu and Daiber 2021), from the original file in apple/ml-mkqa (SHA256 f77ac6bf...ef3d3), licensed CC BY-SA 3.0. Used only as a multilingual transfer test, never for training. Questions already in our set were removed by exact normalized match (74 dropped), so near duplicates of NQ questions may remain; the probes were trained on NQ, so this would flatter English and the translations equally.
