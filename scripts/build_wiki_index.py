"""Build the Tantivy passage index from the Wikipedia parquet shards."""
from __future__ import annotations

import glob
import json
import time

import pyarrow.parquet as pq
import tantivy

from knowwhen.paths import LOGS, WIKI_INDEX, WIKI_PARQUET, check_storage, inside_project
from knowwhen.retrieval.wiki import build_schema, split_passages


def main() -> None:
    check_storage()
    path = inside_project(WIKI_INDEX)
    if path.exists():
        raise SystemExit(f"{path} already exists. Move it aside before rebuilding.")
    shards = sorted(glob.glob(str(WIKI_PARQUET / "*.parquet")))
    if not shards:
        raise SystemExit("No parquet shards found")
    path.mkdir(parents=True)
    index = tantivy.Index(build_schema(), path=str(path))
    writer = index.writer(heap_size=1_000_000_000, num_threads=4)
    start, articles, passages = time.time(), 0, 0
    for n, shard in enumerate(shards):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=2000, columns=["title", "text"]):
            for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
                articles += 1
                for p in split_passages(text):
                    writer.add_document(tantivy.Document(title=title, body=p))
                    passages += 1
        writer.commit()
        print(f"shard {n + 1}/{len(shards)} articles={articles} passages={passages} t={time.time() - start:.0f}s", flush=True)
    writer.wait_merging_threads()
    meta = dict(articles=articles, passages=passages, seconds=time.time() - start, shards=len(shards), engine="tantivy")
    (LOGS / "wiki_index_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
