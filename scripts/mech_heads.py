"""POST-HOC (descriptive, not registered) per-head mechanism measures.

Registered as a post-hoc design in DECISIONS.md 2026-06-11 ("borrowed
mechanism analyses") BEFORE any checkpoint was scored; defined AFTER the
registered CM/behavioral results were known, so corroboration only.

Per checkpoint, three per-head views of the she-he machinery:

A. Per-head DLA: mech_decomp's exact residual decomposition refined from
   per-layer attention to per-head. Head (l,h) contributes
   y_lh @ wo[:, h*Dh:(h+1)*Dh].T @ (w_she - w_he) / rms(x_final), the
   same realized-rms pre-softcap convention; ve mixing is part of v and
   is attributed to the head that reads it. Heads sum to the layer's
   attention contribution exactly (asserted).

B. Per-head zero-ablation: zero head (l,h)'s pre-wo output at ALL
   positions, full nonlinear forward, report the ablated CM (post-
   softcap log-softmax margin on FEMCUE, the mech_margins definition).
   Causal necessity, complementing A's attribution.

C. Static OV-cosine: cos( mean_cue wo_slice @ v_h(cue), w_she - w_he )
   per head, where cues are the single-token members of
   GIRLS_V2 u FEM_NOUNS in the run's own tokenizer (battery girl names
   have no single-token web-vocab encoding, so the web-side cue set is
   fem_nouns plus any single-token names; cue ids are recorded in every
   output row). Tracks whether the cue->she OV alignment decays or
   flips when the margin does.

Prompt sets are the frozen battery slices of mech_margins/mech_decomp.
Checkpoints stream from S3 as in mech_decomp (local ckpts/ preferred).

  python scripts/mech_heads.py runs/web_packed_v2_s42 [runs/...]
      [--battery data/probes/rvp3/battery.jsonl]

Writes runs/<run>/mech_heads.jsonl per run.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from mech_decomp import iter_checkpoints  # noqa: E402

from fogen.data import load_tokenizer  # noqa: E402
from fogen.evals.scoring import load_battery  # noqa: E402
from fogen.model import GPT, ModelConfig, _rmsnorm, _rotary  # noqa: E402
from fogen.probes.rvp import FEM_NOUNS, GIRLS_V2  # noqa: E402


def forward_heads(model, idx, ablate=None):
    """Full forward mirroring GPT.forward, additionally returning each
    head's wo-projected last-position contribution. ablate=(layer, head)
    zeroes that head's pre-wo output at all positions.

    Returns (logits_last, e0, head_contrib, mlp_out, x_final) where
    head_contrib[l][h] is a (d_model,) vector."""
    cfg = model.cfg
    T = idx.size(1)
    cos, sin = model.rope_cos[:, :, :T], model.rope_sin[:, :, :T]
    H, Dh = cfg.n_head, cfg.head_dim
    x = _rmsnorm(model.wte(idx))
    e0 = x[0, -1].clone()
    head_contrib, mlp_out = [], []
    for i, b in enumerate(model.blocks):
        ve = (model.value_embeds[str(i)](idx)
              if str(i) in model.value_embeds else None)
        at = b.attn
        xn = _rmsnorm(x)
        B, T2, C = xn.shape
        q = at.wq(xn).view(B, T2, H, Dh).transpose(1, 2)
        k = at.wk(xn).view(B, T2, H, Dh).transpose(1, 2)
        v = at.wv(xn).view(B, T2, H, Dh).transpose(1, 2)
        if at.ve_lambdas is not None:
            vee = ve.view(B, T2, H, Dh).transpose(1, 2)
            v = at.ve_lambdas[0] * v + at.ve_lambdas[1] * vee
        q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
        q, k = _rmsnorm(q), _rmsnorm(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if ablate is not None and ablate[0] == i:
            y = y.clone()
            y[:, ablate[1]] = 0.0
        head_contrib.append(
            [(y[0, h, -1] @ at.wo.weight[:, h * Dh:(h + 1) * Dh].T).clone()
             for h in range(H)])
        a = at.wo(y.transpose(1, 2).reshape(B, T2, C))
        x = x + a
        m = b.mlp(_rmsnorm(x))
        x = x + m
        mlp_out.append(m[0, -1].clone())
    xf = x[0, -1]
    logits = 15.0 * torch.tanh(model.lm_head(_rmsnorm(xf)).float() / 15.0)
    return logits, e0, head_contrib, mlp_out, xf


def ov_cosines(model, cue_ids, she, he):
    """Static OV alignment: per layer, per head, cosine between the
    mean cue-token OV output and the she-he unembedding direction.
    Zero-norm safe (returns 0.0, e.g. at init where wo is zero)."""
    cfg = model.cfg
    H, Dh = cfg.n_head, cfg.head_dim
    dw = model.lm_head.weight[she] - model.lm_head.weight[he]
    e = _rmsnorm(model.wte.weight[cue_ids])
    out = []
    for i, b in enumerate(model.blocks):
        at = b.attn
        v = at.wv(e)
        if at.ve_lambdas is not None:
            vee = model.value_embeds[str(i)].weight[cue_ids]
            v = at.ve_lambdas[0] * v + at.ve_lambdas[1] * vee
        per = []
        for h in range(H):
            sl = slice(h * Dh, (h + 1) * Dh)
            ov = (v[:, sl] @ at.wo.weight[:, sl].T).mean(0)
            if ov.norm() == 0 or dw.norm() == 0:
                per.append(0.0)
            else:
                per.append(F.cosine_similarity(ov, dw, dim=0).item())
        out.append(per)
    return out


def cue_token_ids(tok, vocab_size):
    """Single-token ids of GIRLS_V2 u FEM_NOUNS, with and without the
    leading space, deduplicated and sorted."""
    ids = set()
    for w in list(GIRLS_V2) + list(FEM_NOUNS):
        for v in (w, " " + w):
            enc = tok.encode(v).ids
            if len(enc) == 1:
                ids.add(enc[0])
    assert ids, "no single-token cues in this vocab"
    assert all(i < vocab_size for i in ids)
    return sorted(ids)


def cm_margin(logits, she, he):
    lp = F.log_softmax(logits, dim=-1)
    return (lp[she] - lp[he]).item()


def score_checkpoint(model, enc, femcue, she, he, cue_ids):
    cfg = model.cfg
    L, H = cfg.n_layer, cfg.n_head
    n = len(femcue)
    cm = gap_pre = c_dir = 0.0
    c_heads = [[0.0] * H for _ in range(L)]
    c_mlp = [0.0] * L
    abl_cm = [[0.0] * H for _ in range(L)]
    with torch.no_grad():
        for p in femcue:
            idx = torch.tensor([enc[p]])
            logits, e0, hc, mlp, xf = forward_heads(model, idx)
            rms = xf.pow(2).mean().sqrt()
            dw = model.lm_head.weight[she] - model.lm_head.weight[he]
            cm += cm_margin(logits, she, he) / n
            gp = (xf @ dw / rms).item()
            comp = (e0 @ dw / rms).item()
            c_dir += comp / n
            tot = comp
            for i in range(L):
                for h in range(H):
                    c = (hc[i][h] @ dw / rms).item()
                    c_heads[i][h] += c / n
                    tot += c
                cmlp = (mlp[i] @ dw / rms).item()
                c_mlp[i] += cmlp / n
                tot += cmlp
            assert abs(tot - gp) < 1e-3, f"decomposition not exact: {tot - gp}"
            gap_pre += gp / n
            for i in range(L):
                for h in range(H):
                    lg, *_ = forward_heads(model, idx, ablate=(i, h))
                    abl_cm[i][h] += cm_margin(lg, she, he) / n
        ov = ov_cosines(model, torch.tensor(cue_ids), she, he)
    return {"CM": cm, "gap_pre": gap_pre, "C_dir": c_dir,
            "C_heads": c_heads, "C_mlp": c_mlp, "abl_CM": abl_cm,
            "ov_cos": ov, "cue_ids": cue_ids, "n_femcue": n}


def score_run(run, battery):
    import re
    run = Path(run)
    assert not re.search(r"s10\d\d$", run.name), \
        f"{run} looks like a quarantined replication seed; refusing"
    cfg = yaml.safe_load(open(run / "config_used.yaml"))
    tok = load_tokenizer(cfg["data"]["tokenizer_dir"])
    model = GPT(ModelConfig(**cfg["model"]))
    model.eval()

    she, he = tok.encode(" she").ids, tok.encode(" he").ids
    assert len(she) == 1 and len(he) == 1
    she, he = she[0], he[0]
    cue_ids = cue_token_ids(tok, model.cfg.vocab_size)

    femcue = sorted({it["prefix"] for it in load_battery(battery)
                     if it["probe"] == "pronoun_gender_ref.conflict"
                     and it["split"] == "heldout"})
    assert femcue
    enc = {p: tok.encode(p).ids for p in femcue}

    rows = []
    for step, path, cleanup in iter_checkpoints(str(run)):
        state = {k: v.float() for k, v in load_file(str(path)).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("rope_") for k in missing), missing
        row = {"step": step,
               **score_checkpoint(model, enc, femcue, she, he, cue_ids)}
        rows.append(row)
        flat = [c for layer in row["C_heads"] for c in layer]
        top = max(range(len(flat)), key=lambda j: abs(flat[j]))
        print(f"{run.name} step {step:6d}  CM={row['CM']:+.3f}  "
              f"top-head L{top // model.cfg.n_head}H{top % model.cfg.n_head}"
              f"={flat[top]:+.3f}", flush=True)
        if cleanup:
            cleanup()
    rows.sort(key=lambda r: r["step"])
    out = run / "mech_heads.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--battery", default="data/probes/rvp3/battery.jsonl")
    args = ap.parse_args()
    for run in args.run_dirs:
        score_run(run, args.battery)


if __name__ == "__main__":
    main()
