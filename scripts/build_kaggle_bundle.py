"""Builds the private Kaggle dataset: the code generation needs, questions and top 3 passages.

Nothing else from the project goes in, and the build stops if anything looks like a credential.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile

from knowwhen.paths import ARTIFACTS, DATA, ENV, ROOT, inside_project

CODE = ["paths.py", "text.py", "__init__.py", "models/__init__.py", "models/prompts.py", "models/runner.py"]
SCRIPTS = ["run_generation.py", "check_prefix.py", "self_check.py", "kaggle_run.py"]
SECRET = re.compile(r"hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH) PRIVATE KEY")


def main() -> None:
    # Built in the APFS image: deleting a folder on exFAT trips over its sidecar files.
    out = inside_project(ENV / "tmp" / "kaggle_bundle")
    if out.exists():
        shutil.rmtree(out)   # only ever this generated folder, rebuilt from the sources below
    (out / "src" / "knowwhen" / "models").mkdir(parents=True)
    (out / "scripts").mkdir()
    (out / "data").mkdir()
    for f in CODE:
        shutil.copy(ROOT / "src" / "knowwhen" / f, out / "src" / "knowwhen" / f)
    for f in SCRIPTS:
        shutil.copy(ROOT / "scripts" / f, out / "scripts" / f)
    shutil.move(out / "scripts" / "kaggle_run.py", out / "kaggle_run.py")
    shutil.copy(DATA / "processed" / "questions.jsonl", out / "data" / "questions.jsonl")
    shutil.copy(DATA / "processed" / "mkqa_questions.jsonl", out / "data" / "mkqa_questions.jsonl")
    # Config and tokenizer files of the exact Hugging Face revisions used, so runs can take weights
    # from Kaggle Models and check them against these hashes instead of needing a token.
    shutil.copy(ROOT / "configs" / "model_weight_hashes.json", out / "data" / "model_weight_hashes.json")
    hashes = json.load(open(ROOT / "configs" / "model_weight_hashes.json"))
    for repo, info in hashes.items():
        snap = ENV / "cache" / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / info["sha"]
        dest = out / "data" / "model_files" / repo.split("/")[-1]
        dest.mkdir(parents=True)
        for name in ("config.json", "generation_config.json", "model.safetensors.index.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"):
            shutil.copy(snap / name, dest / name)
    for name in ("retrieved.jsonl", "retrieved_reranked.jsonl"):
        with open(DATA / "processed" / name) as f, open(out / "data" / name, "w") as g:
            for line in f:
                r = json.loads(line)
                g.write(json.dumps(dict(id=r["id"], passages=r["passages"][:3]), ensure_ascii=False) + "\n")
    files = sorted(p for p in out.rglob("*") if p.is_file() and not p.name.startswith("._"))
    for p in files:
        if SECRET.search(p.read_text(errors="ignore")):
            raise SystemExit(f"possible credential in {p}; bundle not built")
    zip_path = inside_project(ARTIFACTS / "kaggle_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(out))
    print(len(files), "files,", round(zip_path.stat().st_size / 1e6, 1), "MB zip")
    for p in files:
        print(p.relative_to(out), round(p.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
