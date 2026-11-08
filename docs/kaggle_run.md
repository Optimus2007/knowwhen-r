# Running the Aya Expanse 8B replication on Kaggle

The Mac cannot hold this model comfortably, so the replication runs on Kaggle's free GPU T4 x2 accelerator, which gives two 16 GB cards. The model is split across both. T4 cards have no fast bfloat16, so this run uses float16, which is also the model's native format. It is a separate replication, so it is never mixed with the Tiny Aya results from the Mac.

## One time setup

Sign in to Kaggle and verify a phone number, which unlocks GPUs and internet access. Create a new read only Hugging Face token just for this and add it in the notebook under Add-ons, Secrets, with the name `HF_TOKEN`. The token should be deleted on Hugging Face when the run is over. Build the bundle with `python scripts/build_kaggle_bundle.py`, then upload `artifacts/kaggle_bundle.zip` as a private Kaggle dataset. It holds only the generation code, the questions and the top three passages, and the build stops if it finds anything that looks like a credential.

## Running

Create a notebook, attach the dataset, set the accelerator to GPU T4 x2 and turn internet on. Run the smoke test first, in one cell:

```python
import glob, subprocess, sys
run = glob.glob("/kaggle/input/**/kaggle_run.py", recursive=True)[0]
subprocess.run([sys.executable, run, "--smoke"], check=True)
```

The smoke test checks that both GPUs are visible, that the cached preamble leaves answers unchanged on this model, and how long 64 questions take in each condition. Those 64 questions all come from the start of the file, so the timing is rough. Do not start the full run if the preamble check fails or any step stops with non finite activations.

For the full run, drop `--smoke` and use Save Version, then Save and Run All, so it keeps going with the browser closed. It stops by itself after eight hours of generation by default, which leaves time under the session limit to save outputs. To continue, attach the previous version's output to a new notebook and pass `--resume "/kaggle/input/**/kw"` with the folder that holds `kw` and `kw_env`.

Outputs are in `/kaggle/working`: answers under `kw/artifacts/runs`, activations under `kw_env/acts`, and package versions in `versions.json`. All questions in both conditions would come to about 13 GB of activations. If the smoke test or Kaggle's output limit makes that impractical, the run will be cut to a subset, and that decision will be recorded here before results are seen.

## Status

Smoke test passed on two T4 cards on 5 November. Runs must be started from the Kaggle website, because secrets do not reach runs pushed from the command line. Tiny Aya full runs use `--tag t4` so they cannot overwrite the Mac results.

On 6 November the 8B run completed closed book and ran out of GPU memory during RAG. `experiments/kaggle/run_rerank.ipynb` now runs RAG for both models on the reranked passages, writing to the `t4-rerank` and `rerank` tags, and the closed outputs from the earlier runs are reused because they do not depend on retrieval.

On 8 November the multilingual run could not read the `HF_TOKEN` secret, so `kaggle_run.py` gained `--kaggle-model`. It takes the weights from Kaggle Models, checks every weight file against `configs/model_weight_hashes.json`, and takes config and tokenizer files from the bundle, which copies them from the same Hugging Face revision. No token is needed then, so the notebook can be pushed with `kaggle kernels push -p experiments/kaggle/mkqa_kernel`.
