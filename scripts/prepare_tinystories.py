"""Download TinyStories, train BPE-8192 tokenizer, write uint16 shards.

Run on the GPU/head node (needs ~6GB disk + HF download). Then:
  aws s3 sync data/tinystories s3://fogen-phase/data/tinystories
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tinystories/bpe8192")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--tokenizer-docs", type=int, default=200_000)
    args = ap.parse_args()

    from datasets import load_dataset
    from fogen.data import train_tokenizer, load_tokenizer, write_shards

    out = Path(args.out)
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"dataset: {len(ds)} stories")

    if not (out / "tokenizer.json").exists():
        print("training tokenizer...")
        train_tokenizer((ds[i]["text"] for i in range(min(args.tokenizer_docs, len(ds)))),
                        args.vocab, out)
    tok = load_tokenizer(out)
    print(f"vocab: {tok.get_vocab_size()}")

    print("writing shards...")
    manifest = write_shards((r["text"] for r in ds), tok, out / "shards")
    print(f"total tokens: {manifest['total_tokens']:,}")

    val = load_dataset("roneneldan/TinyStories", split="validation")
    write_shards((r["text"] for r in val), tok, out / "val_shards")
    print("done")


if __name__ == "__main__":
    main()
