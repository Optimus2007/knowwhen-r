"""Build the question set: load, deduplicate across datasets with CuratorKIT, split, save."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter

from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.schema import DataSample
from datasets import load_dataset

from knowwhen.paths import DATA, check_storage, inside_project

POPQA_FAMILY = {
    "occupation": "people", "place of birth": "people", "father": "people", "mother": "people", "religion": "people",
    "genre": "arts", "producer": "arts", "director": "arts", "screenwriter": "arts", "composer": "arts", "author": "arts",
    "country": "geography", "capital of": "geography", "capital": "geography",
    "color": "other", "sport": "other",
}


def bucket(key: str) -> float:
    """Deterministic number in [0, 1) from a string, used for entity level splits."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32


def record(dataset, uid, question, answers, pool, **meta):
    answers = sorted({str(a).strip() for a in answers if str(a).strip()})
    return dict(id=f"{dataset}:{uid}", dataset=dataset, question=question.strip(), answers=answers, pool=pool, meta=meta)


def load_all(cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []

    def take(ds, n):
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        return ds.select(idx[:n])

    for pool, split in (("test", "validation"), ("train", "train")):
        for ex in take(load_dataset("google-research-datasets/nq_open", split=split), cap):
            uid = hashlib.sha1(ex["question"].encode()).hexdigest()[:12]
            rows.append(record("nq", uid, ex["question"], ex["answer"], pool))
        for ex in take(load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split), cap):
            a = ex["answer"]
            rows.append(record("triviaqa", ex["question_id"], ex["question"], a["aliases"] + [a["value"]], pool))
        for ex in take(load_dataset("hotpotqa/hotpot_qa", "distractor", split=split), cap):
            if ex["answer"].lower() in ("yes", "no"):
                continue  # a coin flip would inflate accuracy and add no factual signal
            rows.append(record("hotpotqa", ex["id"], ex["question"], [ex["answer"]], pool, type=ex["type"], level=ex["level"]))
    for ex in load_dataset("akariasai/PopQA", split="test"):
        # Split by subject so one entity never crosses train and test.
        pool = "test" if bucket(ex["subj"] or ex["s_uri"]) < 0.3 else "train"
        rows.append(record("popqa", ex["id"], ex["question"], json.loads(ex["possible_answers"]), pool,
                           relation=ex["prop"], family=POPQA_FAMILY[ex["prop"]], s_pop=ex["s_pop"], subject=ex["subj"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cap", type=int, default=12000, help="candidates per dataset and pool before dedup")
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--test", type=int, default=1000)
    args = ap.parse_args()
    check_storage()
    rows = load_all(args.cap, args.seed)
    print("loaded", Counter((r["dataset"], r["pool"]) for r in rows))

    # Test items go first: both deduplicators keep the first occurrence, so a near
    # duplicate is always dropped from training and never from the test set.
    rows.sort(key=lambda r: r["pool"] != "test")
    samples = [DataSample(id=r["id"], source_uri=r["dataset"], instruction=r["question"], task_type="instruction_following") for r in rows]
    after_exact = ExactDeduplicator().run(samples)
    after_near = MinHashDeduplicator(threshold=0.8).run(after_exact)
    keep = {s.id for s in after_near}
    kept = [r for r in rows if r["id"] in keep]
    dedup_stats = dict(loaded=len(rows), after_exact=len(after_exact), after_minhash=len(after_near),
                       removed_by_pool=Counter(r["pool"] for r in rows if r["id"] not in keep))
    print(dedup_stats)

    rng = random.Random(args.seed)
    final = []
    for name in ("nq", "triviaqa", "hotpotqa", "popqa"):
        test = [r for r in kept if r["dataset"] == name and r["pool"] == "test"]
        train = [r for r in kept if r["dataset"] == name and r["pool"] == "train"]
        rng.shuffle(test)
        rng.shuffle(train)
        for r in test[: args.test]:
            final.append({**r, "split": "test"})
        for r in train[: args.val]:
            final.append({**r, "split": "val"})
        for r in train[args.val : args.val + args.train]:
            final.append({**r, "split": "train"})
    rng.shuffle(final)  # mixed order, so stopping early leaves a balanced prefix
    out = inside_project(DATA / "processed")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "questions.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists. Move it aside before regenerating.")
    with path.open("w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = dict(args=vars(args), dedup=dedup_stats, counts={f"{k[0]}/{k[1]}": v for k, v in Counter((r["dataset"], r["split"]) for r in final).items()},
                   curatorkit_provenance=[p.model_dump(mode="json") for p in after_near[0].provenance] if hasattr(after_near[0], "provenance") else None)
    (out / "questions_manifest.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary["counts"], indent=1))


if __name__ == "__main__":
    main()
