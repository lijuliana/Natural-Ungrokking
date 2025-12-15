"""POST-HOC (exploratory, not registered) second mechanism measure.

Two scalars per checkpoint, computed by offline rescoring of frozen
checkpoints; defined and committed 2026-06-11 BEFORE any checkpoint was
scored with them (see DECISIONS.md), but AFTER the registered CM results
were known — so they are corroboration, never confirmatory evidence.

1. Direct-logit decomposition (DLA) of the she-he gap on FEMCUE.
   The final hidden state is an exact sum of residual increments:
       x_final = rmsnorm(wte) + sum_i attn_i + sum_i mlp_i
   and the pre-softcap logit gap is x_final . (w_she - w_he) / rms(x_final),
   so each component's contribution is exact given the realized rms.
     C_dir = embedding-stream (direct path) contribution
     C_ctx = sum of all attention + MLP contributions
   C_ctx is the *contextual* (cue-driven) part of the contrast margin; it
   does not depend on the CM instrument's peak-height validity gate, so it
   is defined on all seeds including the gate-invalid ones.

2. Cue-gender decodability in the residual stream.
   FEMCUE (girl-name conflict/heldout prefixes) vs MASCCUE (boy-name
   agree/heldout prefixes, same frames, same battery). Representation:
   rmsnorm of the residual stream at the last token after each block
   (layer 0 = embedding stream only). Classifier: cross-frame
   nearest-class-mean — class means fit on one frame's prompts, tested
   on the other frame's, both directions — no trained parameters, no
   hyperparameters, and unbiased at chance 0.5 under no signal (a
   leave-one-out variant is chance-biased downward). dec_acc[k] in
   [0,1]; dec_sep = mean signed projection margin at the final layer
   (positive = test point on its own class's side).

Prompt sets are the same frozen battery slices as scripts/mech_margins.py
(data/probes/rvp3/battery.jsonl): FEMCUE = pronoun_gender_ref.conflict
heldout prefixes; MASCCUE = pronoun_gender_ref.agree heldout prefixes.

Checkpoints stream from s3://fogen-phase/runs/<run>/ckpts/ (download,
score, delete — never modifies S3). Local ckpts/ dirs are used if present.

  python scripts/mech_decomp.py runs/web_packed_v2_s42 [runs/...]
      [--battery data/probes/rvp3/battery.jsonl]

Writes runs/<run>/mech_decomp.jsonl per run.
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from fogen.data import load_tokenizer
from fogen.evals.scoring import load_battery
from fogen.model import GPT, ModelConfig, _rmsnorm

BUCKET = "fogen-phase"


def decompose(model, idx, she, he):
    """Exact residual decomposition of the pre-softcap she-he logit gap
    at the last position. Returns (c_dir, c_attn[4], c_mlp[4], gap_pre,
    gap_post, hiddens) where hiddens[k] is the normalized residual after
    block k (k=0: embedding stream only)."""
    cfg = model.cfg
    T = idx.size(1)
    cos = model.rope_cos[:, :, :T]
    sin = model.rope_sin[:, :, :T]
    x = _rmsnorm(model.wte(idx))
    e0 = x[0, -1].clone()
    hiddens = [_rmsnorm(x[0, -1]).clone()]
    attn_out, mlp_out = [], []
    for i, b in enumerate(model.blocks):
        ve = (model.value_embeds[str(i)](idx)
              if str(i) in model.value_embeds else None)
        a = b.attn(_rmsnorm(x), ve, cos, sin)
        x = x + a
        m = b.mlp(_rmsnorm(x))
        x = x + m
        attn_out.append(a[0, -1].clone())
        mlp_out.append(m[0, -1].clone())
        hiddens.append(_rmsnorm(x[0, -1]).clone())
    xf = x[0, -1]
    rms = xf.pow(2).mean().sqrt()
    dw = model.lm_head.weight[she] - model.lm_head.weight[he]
    c_dir = (e0 @ dw / rms).item()
    c_attn = [(a @ dw / rms).item() for a in attn_out]
    c_mlp = [(m @ dw / rms).item() for m in mlp_out]
    logits = 15.0 * torch.tanh(model.lm_head(_rmsnorm(xf)).float() / 15.0)
    gap_pre = (xf @ dw / rms).item()
    gap_post = (logits[she] - logits[he]).item()
    return c_dir, c_attn, c_mlp, gap_pre, gap_post, hiddens


def crossframe_ncm(fem, masc, fem_frames, masc_frames):
    """Cross-frame nearest-class-mean: class means fit on the prompts of
    every frame except the test point's, both directions. Inputs are
    lists of 1-D tensors and parallel frame labels. Returns (accuracy,
    mean signed projection margin). Unbiased at 0.5 under no signal."""
    fem, masc = torch.stack(fem), torch.stack(masc)
    frames = sorted(set(fem_frames) | set(masc_frames))
    assert len(frames) >= 2, "need >=2 frames for cross-frame validation"
    correct, total, margins = 0, 0, []
    for test_f in frames:
        mu_f = fem[[i for i, f in enumerate(fem_frames)
                    if f != test_f]].mean(0)
        mu_m = masc[[i for i, f in enumerate(masc_frames)
                     if f != test_f]].mean(0)
        d = mu_f - mu_m
        dn = d.norm()
        for own, mu_own, mu_other in ((fem, mu_f, mu_m),
                                      (masc, mu_m, mu_f)):
            own_frames = fem_frames if own is fem else masc_frames
            for i, f in enumerate(own_frames):
                if f != test_f:
                    continue
                h = own[i]
                if (h - mu_own).norm() < (h - mu_other).norm():
                    correct += 1
                total += 1
                sgn = 1.0 if own is fem else -1.0
                if dn > 0:
                    margins.append(
                        (sgn * (h - (mu_f + mu_m) / 2) @ d / dn).item())
                else:
                    margins.append(0.0)
    return correct / total, sum(margins) / total


def score_checkpoint(model, enc, femcue, masccue, fem_frames,
                     masc_frames, she, he):
    n_layer = model.cfg.n_layer
    c_dir = c_ctx = gap_pre = gap_post = 0.0
    c_attn = [0.0] * n_layer
    c_mlp = [0.0] * n_layer
    fem_h = [[] for _ in range(n_layer + 1)]
    masc_h = [[] for _ in range(n_layer + 1)]
    with torch.no_grad():
        for p in femcue:
            idx = torch.tensor([enc[p]])
            d, ca, cm_, gp, gq, hs = decompose(model, idx, she, he)
            c_dir += d / len(femcue)
            gap_pre += gp / len(femcue)
            gap_post += gq / len(femcue)
            for i in range(n_layer):
                c_attn[i] += ca[i] / len(femcue)
                c_mlp[i] += cm_[i] / len(femcue)
            for k, h in enumerate(hs):
                fem_h[k].append(h)
        for p in masccue:
            idx = torch.tensor([enc[p]])
            *_, hs = decompose(model, idx, she, he)
            for k, h in enumerate(hs):
                masc_h[k].append(h)
    c_ctx = sum(c_attn) + sum(c_mlp)
    recon_err = abs(c_dir + c_ctx - gap_pre)
    assert recon_err < 1e-3, f"decomposition not exact: {recon_err}"
    dec = [crossframe_ncm(fem_h[k], masc_h[k], fem_frames, masc_frames)
           for k in range(n_layer + 1)]
    return {"C_dir": c_dir, "C_ctx": c_ctx, "C_attn": c_attn,
            "C_mlp": c_mlp, "gap_pre": gap_pre, "gap_post": gap_post,
            "dec_acc": [a for a, _ in dec], "dec_sep": dec[-1][1],
            "n_fem": len(femcue), "n_masc": len(masccue)}


def iter_checkpoints(run):
    """Yield (step, local_path, cleanup_fn) for each checkpoint, local if
    present, else streamed from S3 one at a time."""
    local = sorted(Path(run).glob("ckpts/step*.safetensors"))
    if local:
        for p in local:
            yield int(re.search(r"step(\d+)", p.name).group(1)), p, None
        return
    import boto3
    s3 = boto3.client("s3")
    keys = []
    kw = {"Bucket": BUCKET, "Prefix": f"{run}/ckpts/"}
    while True:
        r = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", [])
                 if o["Key"].endswith(".safetensors")]
        if not r.get("IsTruncated"):
            break
        kw["ContinuationToken"] = r["NextContinuationToken"]
    assert keys, f"no checkpoints under s3://{BUCKET}/{run}/ckpts/"
    for key in sorted(keys):
        step = int(re.search(r"step(\d+)", key).group(1))
        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors",
                                          delete=False)
        tmp.close()
        s3.download_file(BUCKET, key, tmp.name)
        yield step, Path(tmp.name), lambda p=tmp.name: Path(p).unlink()


def score_run(run, battery):
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

    items = load_battery(battery)
    fem_items = sorted({(it["prefix"], it["template_id"]) for it in items
                        if it["probe"] == "pronoun_gender_ref.conflict"
                        and it["split"] == "heldout"})
    masc_items = sorted({(it["prefix"], it["template_id"]) for it in items
                         if it["probe"] == "pronoun_gender_ref.agree"
                         and it["split"] == "heldout"})
    assert fem_items and masc_items
    femcue = [p for p, _ in fem_items]
    masccue = [p for p, _ in masc_items]
    fem_frames = [f for _, f in fem_items]
    masc_frames = [f for _, f in masc_items]
    enc = {p: tok.encode(p).ids for p in femcue + masccue}

    rows = []
    for step, path, cleanup in iter_checkpoints(str(run)):
        state = {k: v.float() for k, v in load_file(str(path)).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("rope_") for k in missing), missing
        row = {"step": step,
               **score_checkpoint(model, enc, femcue, masccue,
                                  fem_frames, masc_frames, she, he)}
        rows.append(row)
        print(f"{run.name} step {step:6d}  C_ctx={row['C_ctx']:+.3f}  "
              f"C_dir={row['C_dir']:+.3f}  dec_acc={row['dec_acc'][-1]:.2f}",
              flush=True)
        if cleanup:
            cleanup()
    rows.sort(key=lambda r: r["step"])
    out = run / "mech_decomp.jsonl"
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
