"""Frozen M4 margin scalars over a run's checkpoints (Step-5 rebuild).

Registered 2026-06-10 (RESEARCH_LOG "STEP 5 REBUILD part 2") BEFORE any
computation. Frozen prompt sets:
  FEMCUE  = rvp3.1 battery heldout conflict prefixes for
            pronoun_gender_ref (name cue; names need not be single-token
            — margins are tokenizer-blind, unlike S1).
  NEUTRAL = the same four frames with the name cue replaced by each of
            eight frozen gender-neutral subjects (rule frozen here, in
            code, before any forward pass).
Per checkpoint:
  PM = mean[ logp(" he") - logp(" she") ] over NEUTRAL  (bare prior pull)
  CM = mean[ logp(" she") - logp(" he") ] over FEMCUE   (cued rule margin)
Writes mech_margins.jsonl: {"step", "PM", "CM", "n_neutral", "n_femcue"}.

  python scripts/mech_margins.py runs/web_packed_v2_s42 \
      [--battery data/probes/rvp3/battery.jsonl] [--out ...]
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.data import load_tokenizer
from fogen.evals.scoring import load_battery
from fogen.model import GPT, ModelConfig

NEUTRAL_SUBJECTS = ["The child", "The friend", "The teacher", "The student",
                    "The neighbor", "The baby", "The doctor", "The farmer"]
FRAMES = ["{n} smiled because", "{n} was happy because",
          "{n} laughed when", "{n} cried because"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--battery", default="data/probes/rvp3/battery.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = yaml.safe_load(open(run / "config_used.yaml"))
    tok = load_tokenizer(cfg["data"]["tokenizer_dir"])
    model = GPT(ModelConfig(**cfg["model"]))
    model.eval()

    she, he = tok.encode(" she").ids, tok.encode(" he").ids
    assert len(she) == 1 and len(he) == 1, "pronouns must be single-token"
    she, he = she[0], he[0]

    femcue = sorted({it["prefix"] for it in load_battery(args.battery)
                     if it["probe"] == "pronoun_gender_ref.conflict"
                     and it["split"] == "heldout"})
    neutral = [fr.format(n=s) for fr in FRAMES for s in NEUTRAL_SUBJECTS]
    assert femcue, "no femcue prefixes found in battery"
    enc = {p: tok.encode(p).ids for p in femcue + neutral}

    out_path = Path(args.out or run / "mech_margins.jsonl")
    rows = []
    with torch.no_grad():
        for ck in sorted(run.glob("ckpts/step*.safetensors")):
            step = int(re.search(r"step(\d+)", ck.name).group(1))
            state = {k: v.float() for k, v in load_file(str(ck)).items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert not unexpected, unexpected
            assert all(k.startswith("rope_") for k in missing), missing

            def margin(prompt, a, b):
                idx = torch.tensor([enc[prompt]])
                lp = F.log_softmax(model(idx)[0, -1], dim=-1)
                return (lp[a] - lp[b]).item()

            cm = sum(margin(p, she, he) for p in femcue) / len(femcue)
            pm = sum(margin(p, he, she) for p in neutral) / len(neutral)
            rows.append({"step": step, "PM": pm, "CM": cm,
                         "n_neutral": len(neutral), "n_femcue": len(femcue)})
            print(f"step {step:6d}  PM={pm:+.3f}  CM={cm:+.3f}", flush=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
