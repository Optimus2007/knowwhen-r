"""Entry point for Kaggle. See docs/kaggle_run.md.

  --smoke          short check: GPUs, preamble cache, and timing on 64 questions per condition
  (no flag)        closed then RAG, stopping cleanly before the session limit
  --resume DIR     copy outputs of an earlier version first, then carry on
  --conditions rag --retrieved retrieved_reranked.jsonl --tag t4-rerank
                   RAG only on the reranked passages; the closed answers do not depend on retrieval

The Hugging Face token comes from Kaggle Secrets and goes only into the child process environment.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working")
ROOT, ENV = WORK / "kw", WORK / "kw_env"
TRANSFORMERS = "transformers==5.22.1"


def child_env(need_token: bool = True) -> dict:
    env = dict(os.environ, KW_REMOTE="1", KW_ROOT=str(ROOT), KW_ENV=str(ENV), PYTHONPATH=str(HERE / "src"), PYTORCH_ALLOC_CONF="expandable_segments:True")
    if need_token:
        from kaggle_secrets import UserSecretsClient
        env["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    return env


def local_model(model: str, pattern: str) -> Path:
    """Weights attached from Kaggle Models, checked file by file against the Hugging Face revision we
    used. Config and tokenizer files come from the bundle, copied from that same revision, so the
    chat template and preamble are unchanged. Stops if any weight file differs."""
    import hashlib
    expected = json.load(open(HERE / "data" / "model_weight_hashes.json"))[model]["files"]
    src = None
    for cand in glob.glob(pattern, recursive=True):
        if all(os.path.exists(os.path.join(cand, f)) for f in expected):
            src = Path(cand)
            break
    if src is None:
        raise SystemExit(f"no attached folder under {pattern} holds {sorted(expected)}")
    for f, sha in expected.items():
        h = hashlib.sha256()
        with open(src / f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != sha:
            raise SystemExit(f"{f} in {src} does not match the Hugging Face revision; stopping")
        print("verified", f, flush=True)
    out = WORK / "models" / model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    for f in expected:
        if not (out / f).exists():
            os.symlink(src / f, out / f)
    for f in (HERE / "data" / "model_files" / model.split("/")[-1]).iterdir():
        shutil.copy(f, out / f.name)
    return out


def run(cmd: list[str], env: dict) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--model", default="CohereLabs/aya-expanse-8b")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--conditions", default="closed,rag")
    ap.add_argument("--retrieved", default="retrieved.jsonl")
    ap.add_argument("--checks", action="store_true", help="also run the P(Yes) and self consistency baselines")
    ap.add_argument("--tag", default="", help="output name suffix, which keeps these runs apart from the Mac runs")
    ap.add_argument("--kaggle-model", default="", help="glob for weights attached from Kaggle Models; then no token is needed")
    ap.add_argument("--questions", default="questions.jsonl", help="question file, e.g. mkqa_questions.jsonl")
    ap.add_argument("--hours", type=float, default=8.0, help="total generation budget; leaves room under the session limit to save outputs")
    args = ap.parse_args()

    env = child_env(need_token=not args.kaggle_model)   # first, so a missing secret fails before any GPU time goes on installs
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", TRANSFORMERS, "accelerate"], check=True)
    import torch
    n = torch.cuda.device_count()
    print("gpus", [torch.cuda.get_device_name(i) for i in range(n)], "torch", torch.__version__, flush=True)
    if n < 2 and "8b" in args.model:
        raise SystemExit("this model needs the GPU T4 x2 accelerator; check the notebook settings")

    for d in (ROOT / "data" / "processed", ROOT / "artifacts", ENV):
        d.mkdir(parents=True, exist_ok=True)
    for name in ("questions.jsonl", "retrieved.jsonl", "retrieved_reranked.jsonl", "mkqa_questions.jsonl"):
        if (HERE / "data" / name).exists():
            shutil.copy(HERE / "data" / name, ROOT / "data" / "processed" / name)
    if args.resume:
        src = Path(glob.glob(args.resume)[0])
        shutil.copytree(src / "kw" / "artifacts", ROOT / "artifacts", dirs_exist_ok=True)
        shutil.copytree(src / "kw_env", ENV, dirs_exist_ok=True)

    py = [sys.executable, str(HERE / "scripts" / "run_generation.py"), "--model", args.model, "--device", "auto", "--dtype", args.dtype, "--batch", str(args.batch)]
    if args.kaggle_model:
        py += ["--model-path", str(local_model(args.model, args.kaggle_model))]
    start = time.time()
    if args.smoke:
        run([sys.executable, str(HERE / "scripts" / "check_prefix.py"), "--model", args.model, "--dtype", args.dtype], env)
        for cond in ("closed", "rag"):
            t = time.time()
            run(py + ["--condition", cond, "--limit", "64", "--tag", f"smoke-{args.dtype}"], env)
            print(f"smoke {args.model} {args.dtype} {cond}: {time.time() - t:.0f}s including model load", flush=True)
        return
    for cond in args.conditions.split(","):
        left = args.hours - (time.time() - start) / 3600
        if left <= 0.05:
            print("no time left for", cond, flush=True)
            break
        run(py + ["--condition", cond, "--retrieved", args.retrieved, "--questions", args.questions, "--max-hours", f"{left:.3f}"] + (["--tag", args.tag] if args.tag else []), env)
    if args.checks:
        check = [sys.executable, str(HERE / "scripts" / "self_check.py"), "--model", args.model, "--device", "auto", "--dtype", args.dtype, "--batch", str(args.batch)]
        check += ["--tag", args.tag] if args.tag else []
        run(check + ["--mode", "yes", "--max-hours", "0.5"], env)
        run(check + ["--mode", "consistency", "--max-hours", "1.5"], env)
    (WORK / f"versions_{args.model.split('/')[-1]}.json").write_text(json.dumps(dict(torch=torch.__version__, transformers=TRANSFORMERS, gpus=[torch.cuda.get_device_name(i) for i in range(n)], model=args.model)))


if __name__ == "__main__":
    main()
