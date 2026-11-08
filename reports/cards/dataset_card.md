---
license: other
license_name: mixed-source-terms
language:
- en
- es
- de
- fr
- ru
- tr
tags:
- hallucination-detection
- retrieval-augmented-generation
- selective-prediction
pretty_name: KnowWhen-R traces
---

# KnowWhen-R traces

Everything the KnowWhen-R experiments produced except the activations, which are too large to host: every model answer with its label and token statistics, the probe and policy scores for every fold, the question splits, the retrieved passages, the result tables and the run logs, including the failed runs. Code and the full report are at https://github.com/Optimus2007/knowwhen-r, and the probes are in the model repository `kungfupanda12/knowwhen-r-probes`.

| Path | Contents |
|---|---|
| `runs/tiny-aya-global-t4/` | Tiny Aya Global on Kaggle T4: closed book answers, and answers with keyword passages |
| `runs/tiny-aya-global-t4-rerank/` | Tiny Aya Global with reranked passages, plus P(Yes) self ratings and five samples at temperature 0.7 |
| `runs/aya-expanse-8b/` | Aya Expanse 8B closed book answers, and a keyword RAG run that ran out of GPU memory at 6,784 rows (kept, not used) |
| `runs/aya-expanse-8b-rerank/` | Aya Expanse 8B with reranked passages, plus P(Yes) and samples |
| `runs/*-mkqa/` | Closed book answers to 600 MKQA questions in six languages, both models |
| `runs/tiny-aya-global-mac-crosscheck/` | Earlier Tiny Aya runs on a Mac, used only to check that labels agree with the T4 runs (99.6%) |
| `predictions/` | Probe and policy scores per question and fold (Parquet). `final_*` use the test split, `dev_*` do not |
| `data/questions.jsonl`, `final_rows.json` | Questions with gold answers and the splits: 9,954 train, 2,398 validation, 3,968 test |
| `data/mkqa_questions.jsonl` | The multilingual test set |
| `data/retrieved.jsonl`, `retrieved_reranked.jsonl` | Keyword and reranked passages from English Wikipedia (20231101) |
| `data/label_audit_sample.json` | The 180 train and validation answers used for the label audit |
| `results/`, `logs/` | Every result table and run log, the same files as in the code repository |

Each line in `runs/*/closed.jsonl` and `rag.jsonl` has the question id, the answer text, whether it matched a gold alias (`correct`), whether the model declined (`refusal`) and token statistics such as mean log probability and entropy.

Sources and licences: NQ Open (CC BY SA 3.0), TriviaQA (Apache 2.0), HotpotQA (CC BY SA 4.0), PopQA (MIT, from the authors' repository), MKQA (CC BY SA 3.0), Wikipedia (CC BY SA 3.0 and GFDL). Model outputs come from Tiny Aya Global and Aya Expanse 8B under CC BY NC 4.0, so noncommercial use only. Each part keeps the terms of its source.

Labels come from matching gold aliases after normalization. An LLM based label audit of 180 answers found no false positives among 41 labelled correct and 3 to 6% disputable among 139 labelled wrong. Tiny Aya sampling stopped at a time limit, so some rows have no samples.

Credits: the question sets above are from Kwiatkowski et al. 2019 and Lee, Chang and Toutanova 2019 (NQ Open), Joshi et al. 2017 (TriviaQA), Yang et al. 2018 (HotpotQA), Mallen et al. 2022 (PopQA) and Longpre, Lu and Daiber 2021 (MKQA). Passages are from the Wikimedia English Wikipedia dump of 1 November 2023, reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`, trained on MS MARCO (Bajaj et al. 2016). The full reference list is in the report in the code repository.
