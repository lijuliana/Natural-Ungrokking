"""One-shot sanity check of the rvp battery before registration."""
from collections import Counter, defaultdict

from fogen.data import load_tokenizer
from fogen.probes.rvp import build_rvp_battery

items = build_rvp_battery(0)
print(f"total items: {len(items)}")

counts = Counter((it["family"], it["condition"], it["split"]) for it in items)
for key in sorted(counts):
    print(f"  {key[0]:22s} {key[1]:8s} {key[2]:7s} {counts[key]:4d}")

# per-family template ids and per-template n
tids = defaultdict(Counter)
for it in items:
    tids[(it["family"], it["condition"], it["split"])][it["template_id"]] += 1
print("\nmin/max per-template n by (family, condition, split):")
for key in sorted(tids):
    ns = tids[key].values()
    print(f"  {key[0]:22s} {key[1]:8s} {key[2]:7s} "
          f"templates={len(ns):3d} n=[{min(ns)},{max(ns)}]")

# uniqueness of item_ids; correct != distractor; both continuations non-empty
ids = set()
tok = load_tokenizer("data/tinystories/bpe8192")
bad = 0
for it in items:
    assert it["item_id"] not in ids, it["item_id"]
    ids.add(it["item_id"])
    assert it["correct"] != it["distractor"], it
    for cont in (it["correct"], it["distractor"]):
        enc = tok.encode(it["prefix"] + cont).ids
        pre = tok.encode(it["prefix"]).ids
        if len(enc) <= len(pre):
            bad += 1
            print("EMPTY-CONT:", it["prefix"], "|", cont)
print(f"\ntokenization: {bad} empty continuations")

# spot-check a few items per family
seen = set()
print("\nsamples:")
for it in items:
    k = (it["family"], it["condition"])
    if k in seen:
        continue
    seen.add(k)
    print(f"  [{it['probe']}] '{it['prefix']}' -> '{it['correct']}' "
          f"vs '{it['distractor']}' (tid={it['template_id']})")
