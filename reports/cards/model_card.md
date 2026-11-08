---
license: cc-by-nc-4.0
base_model:
- CohereLabs/tiny-aya-global
- CohereLabs/aya-expanse-8b
tags:
- hallucination-detection
- probing
- selective-prediction
---

# KnowWhen-R probes

Linear probes on the hidden states of two frozen models, for deciding whether to answer from memory, use retrieved passages or abstain. These are small arrays, not a language model.

Folders: `tiny-aya-global/` for Tiny Aya Global and `aya-expanse-8b/` for Aya Expanse 8B. Code, prompts and the full report are at https://github.com/Optimus2007/knowwhen-r, and every answer and score behind the numbers below is in the dataset repository `kungfupanda12/knowwhen-r-traces`.

Each `.npz` holds `coef`, `intercept`, `mean`, `scale`, `temperature`, `hidden_state`, `position` and `c`. To score a hidden state `h`, use `sigmoid((((h - mean) / scale) @ coef + intercept) / temperature)`. The output is the chance that the answer is right. `hidden_state` indexes `output_hidden_states`, so it is the output of decoder layer `hidden_state - 1`. `position` is `pre` for the last prompt token and `mean` for the mean over answer tokens.

| File | Reads | Predicts |
|---|---|---|
| `closed_answer.npz` | Closed book pass | The closed book answer is right |
| `closed_forecast.npz` | Prompt only | The closed book answer will be right |
| `retrieved_forecast.npz` | Closed book pass | A retrieved answer would be right |
| `retrieved_answer.npz` | RAG pass | The retrieved answer is right |

Trained on 9,954 questions from NQ, TriviaQA, HotpotQA and PopQA, calibrated on 2,398. On 3,968 test questions the probe on retrieved answers reached 0.868 macro AUROC on Tiny Aya and 0.877 on Aya Expanse 8B. On held out datasets it ranged from 0.73 to 0.90, and calibration got worse, with ECE up to 0.13.

Other languages: on the same 600 MKQA questions in Spanish, German, French, Russian and Turkish, the closed book probe kept its AUROC within 0.035 of English with no retraining. Retrieval outside English was not tested.

Limits: short factual questions only, two models from one family, labels from string matching. The probes are tied to the prompts in the code repository. Calibration drifts on new domains, and so does a fixed coverage cut, so set the cut from the confidence of the last 100 questions, as `scripts/online_cut.py` does. Answers that pass the probe are still wrong often, about 41% on Tiny Aya and 33% on Aya Expanse 8B at 70% coverage, so this is not suitable for high stakes use. Noncommercial use only, following the base models' licence.

## Scoring a hidden state

```python
import numpy as np

p = np.load("tiny-aya-global/retrieved_answer.npz")
def score(h):  # h: hidden state at p["hidden_state"], averaged over answer tokens if p["position"] == "mean"
    z = ((h - p["mean"]) / p["scale"]) @ p["coef"] + p["intercept"]
    return 1 / (1 + np.exp(-z / p["temperature"]))
```

The prompts the probes expect are in `src/knowwhen/models/prompts.py` in the code repository. A different prompt changes the hidden states, so scores would no longer be calibrated.
