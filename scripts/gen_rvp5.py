"""Generate the rvp5 out-of-distribution probe battery (threat T14).

Three families, same forced-choice schema as rvp1-rvp4, all items
split=heldout. OOD-ness is in the frames, not the inventories: name
slots use the registered GIRLS_V2/BOYS_V2 classes so the support
semantics of the rule under test are unchanged, while every frame uses
sentence structures absent from all prior batteries AND pre-pronoun
contexts absent from the Step-6 rescue injection frames (no "and then
/ so / while / because / before / after / where / as soon as / and
later" immediately before the pronoun), so the battery stays valid on
intervention cells later.

Hard constraints enforced here:
- no RESCUE_GIRLS / RESCUE_BOYS name appears in any item (D1-D3);
- no prior battery template_id or two-word frame is reused;
- deterministic output (no RNG); item_ids are stable.

  python scripts/gen_rvp5.py --out data/probes/rvp5/battery.jsonl
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fogen.probes.rvp import BOYS_V2, GIRLS_V2  # noqa: E402

RESCUE_GIRLS = ["Mary", "Nora", "Clara", "Eva", "Ivy", "Hazel",
                "Stella", "Nina", "Tessa", "Wendy", "Flora", "Carla"]
RESCUE_BOYS = ["David", "Noah", "Liam", "Owen", "Felix", "Oscar",
               "Henry", "Carl", "Eric", "Hugo", "Brian", "Dean"]

# ---- pronoun_gender_ref_ood: {n} -> prefix ending right before she/he
PRON_FRAMES = {
    "until":      "{n} sat by the window until",
    "but":        "{n} liked the song, but",
    "even_though": "{n} kept smiling even though",
    # frames with a second animate referent use plural/neutral ones
    # ("the kids") so the distractor pronoun has no licensed antecedent
    "yes_answer": "Did {n} find the key? Yes,",
    "think_say":  '"Where is {n}?" the kids asked. "I think',
    "maybe_say":  'The kids could not find {n} and said, "Maybe',
    "if_comma":   "If {n} wins the game,",
    "since":      "{n} stayed inside since",
    "that_comp":  "{n} promised that",
    "soon":       "{n} got on the bus, and soon",
    "once_comma": "Once {n} found the map,",
    "later_sent": "{n} hid the toy. Later",
}

# ---- reflexive_gender_ood: verbs/preps not in the rvp1-4 reflexive set
REFL_FRAMES = {
    "by_self":    "{n} built the tower all by",
    "proud_of":   "{n} finished the race and felt proud of",
    "talked_to":  "{n} talked quietly to",
    "picture_of": "{n} drew a funny picture of",
    "wrapped":    "{n} wrapped the warm blanket around",
    "told_self":  "{n} took a deep breath and told",
    "looked_after": "While Mom was away, {n} looked after",
    "made_for":   "{n} made a sandwich just for",
}

# ---- plural_was_were_ood: new subject-NP structures; agree side uses
# singular heads with plural attractors (the classic agreement trap).
PLURAL_CONFLICT = {   # plural subject -> " were"
    "conj_np":   "The {a} and the {b}",
    "pp_plural": "The {a}s on the {b}",
    "rel_bought": "The {a}s that Dad bought",
    "num_three": "Three little {a}s",
    "both_of":   "Both of the {a}s",
    "twins":     "The twins near the {b}",
}
PLURAL_AGREE = {      # singular head (often + plural attractor) -> " was"
    "of_attractor":  "The box of {a}s",
    "in_attractor":  "The girl with the {a}s",
    "rel_attractor": "The book that the boys read",
    "one_of":        "One of the {a}s",
    "with_attractor": "The teacher with the {a}s",
    "nobody":        "Nobody in the {b}",
}
NOUN_A = ["cat", "dog", "cup", "frog", "duck", "star", "shoe", "apple",
          "sock", "block", "crayon", "balloon"]
NOUN_B = ["table", "garden", "kitchen", "park", "shelf", "yard",
          "school", "house", "wall", "window", "store", "barn"]


def items():
    out = []

    def add(family, condition, tid, prefix, correct, distractor, idx):
        out.append({
            "family": family, "condition": condition,
            "probe": f"{family}.{condition}",
            "category": "rule_vs_prior", "template_id": tid,
            "split": "heldout", "prefix": prefix,
            "correct": correct, "distractor": distractor, "chance": 0.5,
            "item_id": f"{family}.{condition}/heldout/{idx:04d}"})

    i = {"pc": 0, "pa": 0, "rc": 0, "ra": 0, "wc": 0, "wa": 0}
    for tid, frame in PRON_FRAMES.items():
        for n in GIRLS_V2:
            add("pronoun_gender_ref_ood", "conflict", tid,
                frame.format(n=n), " she", " he", i["pc"]); i["pc"] += 1
        for n in BOYS_V2:
            add("pronoun_gender_ref_ood", "agree", tid,
                frame.format(n=n), " he", " she", i["pa"]); i["pa"] += 1
    for tid, frame in REFL_FRAMES.items():
        for n in GIRLS_V2:
            add("reflexive_gender_ood", "conflict", tid,
                frame.format(n=n), " herself", " himself", i["rc"]); i["rc"] += 1
        for n in BOYS_V2:
            add("reflexive_gender_ood", "agree", tid,
                frame.format(n=n), " himself", " herself", i["ra"]); i["ra"] += 1
    for tid, frame in PLURAL_CONFLICT.items():
        for k in range(12):
            a, b = NOUN_A[k], NOUN_B[(k + 3) % 12]
            pre = frame.format(a=a, b=b)
            if tid == "conj_np":
                pre = frame.format(a=a, b=NOUN_A[(k + 5) % 12])
            add("plural_was_were_ood", "conflict", tid,
                pre, " were", " was", i["wc"]); i["wc"] += 1
    for tid, frame in PLURAL_AGREE.items():
        for k in range(12):
            a, b = NOUN_A[k], NOUN_B[(k + 7) % 12]
            add("plural_was_were_ood", "agree", tid,
                frame.format(a=a, b=b), " was", " were", i["wa"]); i["wa"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/probes/rvp5/battery.jsonl")
    args = ap.parse_args()

    rows = items()
    blob = "".join(json.dumps(r) + "\n" for r in rows)
    for r in rows:  # D1-D3: no injected name anywhere
        for name in RESCUE_GIRLS + RESCUE_BOYS:
            assert name not in r["prefix"], (name, r["item_id"])
    seen = set()
    for r in rows:
        assert r["item_id"] not in seen
        seen.add(r["item_id"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(blob)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    fams = {}
    for r in rows:
        fams[r["probe"]] = fams.get(r["probe"], 0) + 1
    print(f"wrote {args.out}: {len(rows)} items, sha256={sha}")
    for k, v in sorted(fams.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
