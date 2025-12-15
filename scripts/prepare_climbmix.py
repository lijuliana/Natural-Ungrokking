"""Download ClimbMix shards, train BPE-8192 tokenizer, write uint16 shards.

ClimbMix (karpathy/climbmix-400b-shuffle) is v1's actual training corpus
(RESEARCH_LOG 2026-06-10) and the project's standard web slice. Same
pipeline as prepare_tinystories.py; val pinned to shard_06542 like v1.

  python scripts/prepare_climbmix.py --num-shards 14
"""

import argparse
import time
import urllib.request
from pathlib import Path

BASE_URL = ("https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle"
            "/resolve/main")
VAL_SHARD = 6542
DOC_CAP = 10_000  # chars, matches v1 prepare.py


def download(raw_dir: Path, indices):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        name = f"shard_{i:05d}.parquet"
        dst = raw_dir / name
        if dst.exists():
            continue
        for attempt in range(3):
            try:
                print(f"downloading {name} (attempt {attempt + 1})")
                urllib.request.urlretrieve(f"{BASE_URL}/{name}",
                                           str(dst) + ".tmp")
                (raw_dir / (name + ".tmp")).rename(dst)
                break
            except Exception as e:
                print(f"  failed: {e}")
                time.sleep(5)
        else:
            raise RuntimeError(f"could not download {name}")


def docs(paths):
    import pyarrow.parquet as pq
    for p in paths:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg, columns=["text"])["text"].to_pylist():
                yield text[:DOC_CAP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/climbmix/bpe8192")
    ap.add_argument("--raw", default="data/climbmix/raw")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--num-shards", type=int, default=14)
    ap.add_argument("--tokenizer-docs", type=int, default=200_000)
    args = ap.parse_args()

    from fogen.data import train_tokenizer, load_tokenizer, write_shards

    out, raw = Path(args.out), Path(args.raw)
    train_idx = list(range(args.num_shards))
    download(raw, train_idx + [VAL_SHARD])
    train_paths = [raw / f"shard_{i:05d}.parquet" for i in train_idx]
    val_path = raw / f"shard_{VAL_SHARD:05d}.parquet"

    if not (out / "tokenizer.json").exists():
        print("training tokenizer...")
        import itertools
        train_tokenizer(itertools.islice(docs(train_paths[:2]),
                                         args.tokenizer_docs),
                        args.vocab, out)
    tok = load_tokenizer(out)
    print(f"vocab: {tok.get_vocab_size()}")

    print("writing train shards...")
    manifest = write_shards(docs(train_paths), tok, out / "shards")
    print(f"train tokens: {manifest['total_tokens']:,}")
    print("writing val shards...")
    vm = write_shards(docs([val_path]), tok, out / "val_shards")
    print(f"val tokens: {vm['total_tokens']:,}")
    print("done")


if __name__ == "__main__":
    main()
