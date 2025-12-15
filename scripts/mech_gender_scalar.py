"""Gender-coupling scalar S1 across a run's checkpoints (Step-5 candidate).

S1(ckpt) = cos( mean_f wte[f] - mean_m wte[m],  u[" she"] - u[" he"] )

where f/m range over GIRLS/BOYS names that are single-token in the run's
tokenizer and u is the lm_head (unembedding) row. Layer- and head-free:
a pure embedding-geometry scalar, comparable across seeds and corpora.
Also writes S2 = ||u_she|| / ||u_he|| (prior-asymmetry control scalar).

Registered predictions in RESEARCH_LOG (2026-06-10) BEFORE computation on
any governed checkpoints.

AMENDMENT 2026-06-10 (still BEFORE any governed-checkpoint computation;
discovered by tokenizer inspection only): the ClimbMix bpe8192 tokenizer
has ZERO single-token girl names and only "Ben" for boys, so the
name-based scalar is undefined on web runs. The registered M1/M2 scalar
is now S1_cue: identical formula, cue classes = GIRLS+FEM_NOUNS vs
BOYS+MASC_NOUNS restricted to single-token variants in the run's
tokenizer (the same cue-class definition as the f*-v2 window counter).
Web inventory: fem {mom, woman}, masc {Ben, king, man} — imbalance
disclosed. S1_name (names only) is still logged where defined (TS) as a
secondary descriptive scalar.

  python scripts/mech_gender_scalar.py runs/web_packed_v2_s42 [--out ...]
"""

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from fogen.data import load_tokenizer

GIRLS = ["Lily", "Mia", "Anna", "Emma", "Sue", "Jane"]
BOYS = ["Tom", "Ben", "Max", "Sam", "Jack", "Tim"]
FEM_NOUNS = ["girl", "woman", "mom", "queen", "princess", "sister",
             "grandma", "aunt"]
MASC_NOUNS = ["boy", "man", "dad", "king", "prince", "brother",
              "grandpa", "uncle"]


def single_token_ids(tok, words):
    """Amended 2026-06-10 BEFORE any computation on governed checkpoints:
    include the leading-space variant (" Lily"), which is how names occur
    mid-text and hence the wte row training actually updates; the bare
    variant only appears sentence-initially."""
    out = {}
    for w in words:
        ids = [tok.encode(v).ids for v in (w, " " + w)]
        toks = [i[0] for i in ids if len(i) == 1]
        if toks:
            out[w] = toks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = yaml.safe_load(open(run / "config_used.yaml"))
    tok = load_tokenizer(cfg["data"]["tokenizer_dir"])

    g_ids = single_token_ids(tok, GIRLS)
    b_ids = single_token_ids(tok, BOYS)
    f_ids = single_token_ids(tok, GIRLS + FEM_NOUNS)
    m_ids = single_token_ids(tok, BOYS + MASC_NOUNS)
    she, he = tok.encode(" she").ids, tok.encode(" he").ids
    assert len(she) == 1 and len(he) == 1, "pronouns must be single-token"
    assert f_ids and m_ids, "no single-token gender cues in tokenizer"
    print(f"single-token names: girls={sorted(g_ids)} boys={sorted(b_ids)}")
    print(f"single-token cues:  fem={sorted(f_ids)} masc={sorted(m_ids)}")

    out_path = Path(args.out or run / "mech_gender_scalar.jsonl")
    rows = []
    for ck in sorted(run.glob("ckpts/step*.safetensors")):
        step = int(re.search(r"step(\d+)", ck.name).group(1))
        sd = load_file(str(ck))
        wte = sd["wte.weight"].float()
        u = sd["lm_head.weight"].float()
        pron_dir = u[she[0]] - u[he[0]]
        f_rows = [t for v in f_ids.values() for t in v]
        m_rows = [t for v in m_ids.values() for t in v]
        cue_dir = wte[f_rows].mean(0) - wte[m_rows].mean(0)
        s1_cue = torch.cosine_similarity(cue_dir, pron_dir, dim=0).item()
        s1_name = None
        if g_ids and b_ids:
            g_rows = [t for v in g_ids.values() for t in v]
            b_rows = [t for v in b_ids.values() for t in v]
            name_dir = wte[g_rows].mean(0) - wte[b_rows].mean(0)
            s1_name = torch.cosine_similarity(name_dir, pron_dir, dim=0).item()
        s2 = (u[she[0]].norm() / (u[he[0]].norm() + 1e-8)).item()
        rows.append({"step": step, "S1_cue_gender_coupling": s1_cue,
                     "S1_name_gender_coupling": s1_name,
                     "S2_unembed_norm_ratio": s2,
                     "n_fem_cues": len(f_ids), "n_masc_cues": len(m_ids),
                     "n_girl_names": len(g_ids), "n_boy_names": len(b_ids)})
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")
    for r in rows[:: max(1, len(rows) // 12)]:
        print(f"  step {r['step']:6d}  S1_cue={r['S1_cue_gender_coupling']:+.3f}"
              f"  S2={r['S2_unembed_norm_ratio']:.3f}")


if __name__ == "__main__":
    main()
