# Novelty and research gap

## Already done by others

Predicting errors from activations is well established (SAPLMA, Orgad et al., Semantic Entropy Probes). So is doing it before generation (Ji et al., Moreno Cencerrado et al.). Using internal state to trigger retrieval is done by SeaKR, CtrlA, Probing-RAG and Jin et al. Training on outcomes with and without retrieval is done by Probing-RAG, which also tests out of domain sets. Cost aware thresholds on calibrated error are done for model cascades (Zellinger and Thomson, UCCI). A three way retrieve or abstain framing exists for LLM judges (Badshah et al.). Selective QA under domain shift is Kamath et al.

The original framing, "decision aware hallucination detection and selective retrieval", would therefore be derivative if presented as a new method.

## Update after reading SABER, 5 November 2026

SABER (Trust or Abstain? A Self-Aware RAG Approach, arXiv 2605.18792) uses probes on hidden states of a frozen model to trust the parametric answer, trust the retrieved answer, or abstain. It scores abstention with +1, -1 and 0, and reports leave one dataset out results. Earlier points 1 and 3 of our claim, as first worded, overlapped with this. Everything below replaces them.

What SABER says it does not cover, and where this project differs: it assumes retrieved context is already supplied and does not address the retrieval step, so retrieval has no cost and is never skipped. It uses three sampled reasoning traces per side plus a self evaluation pass, where we use one greedy pass. It covers Llama and Qwen, where we use multilingual Cohere models under this hackathon's rules. We should cite SABER as the closest prior work and compare on design, since we did not run it.

## Update after reading Query-Level Uncertainty, 6 November 2026

Chen et al. (ICLR 2026, arXiv 2506.09669) already use a signal computed before generation to decide whether to call RAG, so deciding before retrieval is not new by itself and we should not word it that way. The difference is narrower. Their signal is a training free self rating and it forecasts whether the closed book answer is right. Ours is trained, also forecasts how the retrieved answer would do, prices retrieval, decides a second time once the retrieved answer exists, and can abstain, and it is tested on unseen datasets. The P(Yes) baseline in E015 tests whether the training is worth it.

## The refined claim

Among papers read, we did not find one that does all of the following with a frozen open model:

1. Decides whether to pay for retrieval before running it, using an activation forecast of how the retrieved answer would do, then decides again afterwards, with retrieval priced in a cost sweep.
2. Reports wrong answer rate at fixed coverage for such a system, with token statistics put through the same pipeline as a control for whether activations add anything.
3. Separates how much loss on an unseen dataset comes from ranking and how much from calibration.

This is an integration and measurement contribution, not a new probing method. Submission wording stays "we did not find", never "first". The search was limited, and RaDIO and DioR remain unread.
