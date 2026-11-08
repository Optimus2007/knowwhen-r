# Literature review

Written 5 November 2026. arXiv identifiers were checked against the arXiv API (`docs/sources/arxiv_meta.json`). The column "Read" records how closely each paper was actually read, so nothing here looks more verified than it is. "Text" means the extracted PDF was searched and the relevant sections read. "Abstract" means only the abstract was read. Search engine summaries were not treated as sources.

## Detecting errors from internal states

| Paper | Read | What it does | How it was evaluated | Limits, and relation to us |
|---|---|---|---|---|
| SAPLMA, Azaria and Mitchell 2023 (2304.13734) | Abstract | Classifier on hidden states predicts whether a statement is true. | True and false statements, plus LLM generated statements. | Early evidence for the probing idea we build on. |
| Kadavath et al. 2022 (2207.05221) | Abstract | Models estimate P(True) for their answers and P(IK) for questions. | Large proprietary models. | P(IK) is the conceptual ancestor of a question only probe. |
| INSIDE, Chen et al. 2024 (2402.03744) | Text | EigenScore: eigenvalues of the covariance of hidden embeddings across several sampled answers, plus test time feature clipping. | LLaMA, LLaMA2 and OPT on CoQA, SQuAD, TriviaQA, Natural Questions. | Unsupervised, but needs several generations per question. A candidate baseline if time permits. |
| Semantic Entropy Probes, Kossen et al. 2024 (2406.15927) | Text | Probes approximate semantic entropy from one generation. | Llama 2 and 3, Mistral, on QA sets including TriviaQA and SQuAD. | Reports better OOD behaviour than accuracy probes and warns accuracy labels are noisy. Directly relevant to H1, because our probes are accuracy probes. |
| Orgad et al. 2024 (2410.02707) | Text | Truthfulness signal concentrates at exact answer tokens. Detectors fail to generalize across datasets. | Llama 3 8B and Mistral 7B on TriviaQA, Natural Questions, HotpotQA and others. | The main negative prior for H1. It motivates reading activations at answer tokens and testing transfer directly. |
| ICR Probe, Zhang et al. 2025 (2507.16488) | Text | ICR Score measures how much each module contributes to residual stream updates; a probe reads this across layers. | Gemma 2, Llama 3 8B, Qwen2.5 7B on HaluEval, SQuAD, TriviaQA, HotpotQA. | Detection only, and its limitations section says it does not address mitigation. |
| PRISM, Zhang et al. 2024 (2411.04847) | Text | Prepends a prompt that makes truthfulness structure more consistent across domains, improving supervised detectors. | LLaMA2 7B and 13B, on datasets including TruthfulQA. | Targets cross domain transfer, like H1. Detection only. A cheap variant could be tested as an ablation. |
| Simple Factuality Probes, Han et al. 2025 (Findings of EMNLP 2025) | Text | Lightweight probes on hidden states detect hallucination in long form text from one sample. | Models up to Llama 3.1 405B, on LongFact style data. | Reports up to 100x fewer FLOPs than multi sample methods. Long form setting, whereas we study short answers. |
| No Answer Needed, Moreno Cencerrado et al. 2026 (2509.10625) | Text | Difference of means direction at the last question token predicts upcoming correctness. | 7B to 70B models, TriviaQA to other knowledge sets. | Reports good transfer on knowledge data, failure on maths. Close to our pre generation probe, and no decision layer. |
| Ji et al. 2024 (2407.03282) | Abstract | Estimates hallucination risk from internal states before generating. | 15 NLG tasks. | Same pre generation idea. |
| Gottesman and Geva 2024 (2406.12673) | Abstract | Estimates entity knowledge without generating. | Entity level tasks. | Supports the use of cheap pre generation signals. |
| Cheang et al. 2025 (2510.09033) | Abstract | Hidden states track recall, not truth. Hallucinations from spurious associations overlap with correct outputs. | Comparison of hallucination types. | A warning: our probe may catch only "no recall" errors. We should measure this. |
| Ferrando et al. 2024 (2411.14257) | Abstract | SAE directions for entity recognition affect refusal and hallucination. | Mechanistic analysis. | Mechanistic support for a "do I know this entity" signal. |
| Liu et al. 2024 (2407.08582); Bürger et al. 2024 (2407.12831); Marks and Tegmark 2023 (2310.06824) | Abstract | Evidence for shared truth directions across datasets. | Mostly true and false statements. | The optimistic side of the H1 debate. |
| Sahoo et al. 2026 (2606.02907) | Abstract | Probes can detect dataset format rather than the intended property. | Qwen3 14B on three benchmarks. | Methodological warning: we must report dataset source prediction beside our probes. |
| SelfCheckGPT, Manakul et al. 2023 (2303.08896) | Text | Consistency across sampled responses signals hallucination, with no internal access. | 238 GPT-3 WikiBio passages, manually annotated. | Our self consistency baseline follows this idea. Sampling is the cost a probe avoids. |

## Retrieval decided by the model's own state

| Paper | Read | What it does | Limits, and relation to us |
|---|---|---|---|
| Mallen et al. 2022 (2212.10511) | Abstract | PopQA. Models fail on long tail facts, retrieval helps there and can hurt on popular ones. | Motivates selective retrieval and gives us popularity metadata. |
| FLARE (2305.06983), DRAGIN (2403.10081) | Abstract | Retrieve during generation, triggered by token level confidence or attention. | Logit based triggers. No abstention. |
| Self-RAG (2310.11511) | Abstract | Trains the model to emit retrieval and critique tokens. | Needs fine tuning. We keep the model frozen. |
| Adaptive-RAG (2403.14403) | Abstract | External classifier routes by question complexity. | Not based on the model's internal state. |
| SeaKR, Yao et al. 2024 (2406.19215) | Text | Uses uncertainty from internal states (Gram determinant across sampled answers) to trigger retrieval and rerank. | Very close on "internal state triggers retrieval". Text has no mention of abstention. |
| CtrlA, Liu et al. 2024 (2405.18727) | Text | Honesty steering and a confidence probe trigger retrieval during generation. | Very close. Its refusal module sends refusals back into retrieval and regeneration. It is not an abstention policy, and no cost model appears. |
| Probing-RAG, Baek et al. 2024 (2410.13339) | Text | Prober on hidden states of rationale and answer tokens decides on further retrieval. Trained on correctness of generations made both with and without retrieval. 26,060 training samples. Tests on out of domain datasets (MuSiQue, 2Wiki). Reports about 50% fewer retrievals. | The closest paper to ours. It already covers probe triggered retrieval, retrieval aware training data, and OOD tests. The text has no abstention or calibration, and "cost" appears once. |
| Jin et al. 2025 (2509.06472) | Text | Hidden state confidence model measures how much context raises confidence; trains a reranker; triggers retrieval on low confidence. | Close on measuring the effect of retrieval through hidden states, but the aim is reranking. |
| Moskvoretskii et al. 2025 (2501.12835) | Text | 35 systems, 27 uncertainty methods, 6 QA sets, Llama 3.1 8B. Simple uncertainty methods match complex adaptive RAG and are cheaper. OOD transfer changes downstream performance little, while self knowledge drops. | Sets the bar: we must beat plain entropy baselines, not only our own thresholds. |
| RaDIO (AAAI), DioR (ACL 2025) | Not read | Detector triggered dynamic RAG. | The AAAI page blocked automated access. Search summaries only, so nothing is claimed about them. To be read if accessible. |

## Selective prediction, abstention and costs

| Paper | Read | What it does | Relation to us |
|---|---|---|---|
| Chow 1970 | Not read, standard result | Reject rule: with a fixed rejection cost, the optimal rule thresholds posterior error. | Formal reason a single risk score with a threshold is already optimal if well calibrated. |
| Geifman and El-Yaniv 2017 (1705.08500) | Abstract | Selective classification with risk guarantees. | Risk coverage framing. |
| Guo et al. 2017 (1706.04599) | Abstract | Modern networks are miscalibrated; temperature scaling helps. | Our calibration method. |
| Kamath et al. 2020 (2006.09462) | Text | Selective QA under domain shift. A calibrator trained on model behaviour beats softmax probability; OOD exposure helps it. | The closest ancestor of H1 plus abstention, for extractive QA and without retrieval. |
| Wen et al. 2024 survey (2407.18418) | Abstract | Survey of abstention. | Map of the area. |
| R-Tuning (2311.09677), Feng et al. 2024 (2402.00367) | Abstract | Teach refusal, or abstain through multi LLM collaboration. | Different mechanisms. |
| Zellinger and Thomson 2025 (2502.09054) | Abstract | LLM cascades with early abstention under costs. | Same shape of decision: continue, escalate, abstain. The escalation is to a larger model, not to evidence. |
| UCCI, Kotte et al. 2026 (2605.18796) | Abstract | Isotonic calibrated error probability for cascade routing, with cost optimal thresholds under stated assumptions. | Formalizes the argument we use, for routing between models. |
| BAS, Wu et al. 2026 (2604.03216) | Abstract | Decision theoretic metric for answer or abstain. | Candidate evaluation idea. Two way only. |
| Badshah et al. 2026 (2608.17994) | Abstract | Judge, retrieve or abstain with risk guarantees for LLM judges. | The closest three action framing found. Different task (judging), and uses thresholds calibrated on human labels. Must be cited prominently. |

## Added 5 November 2026, while looking for methods to combine

SABER (arXiv 2605.18792) is the closest prior work. See docs/novelty_and_research_gap.md. Its leave one dataset out section reports accuracy gaps within about 3 points on five datasets, with larger drops on two conflict format datasets.

The Hallucination Signal Is a Mean Shift (Lee, Seo and Lim, arXiv 2608.28930, EMNLP 2026) finds that an L2 logistic regression matches twelve more complex probes, and that aggregating a band of layers matches the best single layer without choosing it (LayerMix). This supports keeping linear probes as the main model. Its authors limit the claim to a paired example setting and report that its direction falls to chance on model generated text in a small pilot, so its geometry cannot be assumed to hold here. We only borrow the layer averaging idea, as a test.

Query-Level Uncertainty (Chen, de Melo, Suchanek and Varoquaux, ICLR 2026, arXiv 2506.09669) is the closest work to our step before retrieval. It asks a model whether it can answer, with no answer generated, and combines P(Yes) across tokens and layers with fixed weights into a training free score called Internal Confidence. Above a threshold the model answers from memory, otherwise it calls RAG, and the paper shows accuracy against running time for Phi 3.8B on TriviaQA with the search results that dataset supplies. Retrieval is not priced in a utility, nothing is decided after retrieval, the system does not abstain, there is no leave one dataset out test, and no supervised probe appears as a baseline. Its average AUROC is 59.8 to 67.1 across three models, and the authors say it still trails answer level methods. We add plain P(Yes) as a baseline in E015, using the last token and last layer only, so it is not their full Internal Confidence.

ReDeEP (arXiv 2410.11414, ICLR 2025) separates use of retrieved context from use of parametric knowledge when detecting RAG hallucinations, and LUMINA (arXiv 2509.21875) builds on that idea. Our difference probe between closed book and RAG states is a far simpler relative of this. We read their abstracts and summaries, not the full papers.

Conformal Abstention (Abbasi-Yadkori et al., arXiv 2405.01563) calibrates an abstention cut with conformal risk control, using agreement between sampled answers. Its bound assumes exchangeable data, so we expect it to hold in domain and not necessarily on a held out dataset.

Saerens, Latinne and Decaestecker (Neural Computation 14(1), 21 to 41, 2002, DOI 10.1162/089976602753284446) estimate a new class prior from unlabelled classifier outputs with a simple EM procedure, and rescale the outputs without refitting.

## What the review changes

Internal state routing for retrieval is established, so it cannot be our contribution. Plain uncertainty is a strong baseline, so entropy has to be a first class comparison. Evidence on OOD transfer conflicts, which makes H1 a real open question for a small multilingual model, with an honest negative result possible. Decision theory also limits H2: with one calibrated score and fixed costs, two thresholds are already optimal, so a policy can only beat them through additional information.

## Second search, 6 November 2026, across ACL Anthology, OpenReview, NeurIPS and others

Most directly relevant: Mallen et al. (arXiv 2212.10511), the source of PopQA, report that adaptive retrieval helps less for small models, which tend to retrieve nearly always. This matches our data and sets expectations for how much retrieval a small model can skip. LLM Independent Adaptive RAG (Marina et al., EMNLP 2025) gets comparable results to LLM based uncertainty methods from 27 external features of the question, and is a reminder that cheap non model signals are a fair baseline. Interpretable Uncertainty for Adaptive Retrieval and Reasoning (Dey et al., arXiv 2607.07380) estimates insufficiency and conflict from hidden states in one pass to trigger RAG or more reasoning. AB-RAG (Kamthan, arXiv 2606.29090) adapts retrieval depth using certainty, answer evidence agreement and retrieval score variance, training free. Query Level Uncertainty (OpenReview) uses internal confidence before generation to answer, invoke RAG or abstain, which overlaps with our pre retrieval step and must be read in full before we state how we differ. DioR (ACL 2025) and RaDIO (AAAI 2025) use internal states to decide when to retrieve during generation; the abstracts are read, not the full papers. For transfer, CrossHallu (arXiv 2607.04029) tests hallucination signals across languages and includes Aya 23 8B, and H-Neurons (arXiv 2512.01797) tests transfer from TriviaQA to BioASQ. The paper cited as arXiv 2609.37493 could not be found, so it is not used.

## Added 8 November 2026, from the organisers' Track 2 email

The organisers' email listed four papers. Only the two below bear on this project, and for both we read the arXiv abstract and search summaries, not the full text.

| Paper | Read | What it does | Relevance |
|---|---|---|---|
| Sengupta, Seth and Sankarapu 2025, Interpretability as Alignment (2509.08592) | Abstract | Position paper. Argues mechanistic interpretability should be a design constraint for alignment, giving auditability and governance, rather than a post hoc diagnostic. | Framing only. Our probe is used inside the decision, which fits the argument. Cited in the report's related work. |
| Bhattacharjee, Sharma, Sankarapu and Seth 2026, Provenance-Grounded Gating and Adaptive Recovery (2606.11127) | Abstract | Gates synthetic post training data with hallucination checks that use the exact source evidence, and repairs rejected samples. The CuratorKIT paper (2606.21631) packages it. | About curating training data, not about inference time decisions. Not cited for results. We use CuratorKIT only for deduplication. |

The other two, C-ΔΘ and quantization permanent unlearning, are about refusal editing and unlearning in the weights and do not bear on this problem statement.

