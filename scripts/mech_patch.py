"""POST-HOC (descriptive, not registered) patching + specificity control.

Registered as a post-hoc design in DECISIONS.md 2026-06-11 ("borrowed
mechanism analyses") BEFORE any checkpoint was scored; defined AFTER the
registered CM/behavioral results were known, so corroboration only.

D. Same-trajectory activation patching. Cache component outputs (each
   head pre-wo, each MLP residual increment, the embedding stream) at
   the run's PEAK-CM checkpoint on the frozen FEMCUE prompts; rerun the
   FINAL checkpoint with one component's output replaced by its peak
   cache at all positions; report the patched CM and the recovery
   fraction (CM_patch - CM_final) / (CM_peak - CM_final). Also the
   reverse direction (final cache into peak model). Peak/final steps
   come from the run's mech_margins.jsonl (argmax CM / last step) —
   artifacts, not hand-picked.

E. Direction-specificity control. d_k = mean(FEMCUE) - mean(MASCCUE)
   of the NORMALIZED last-position residual after block k (k=0 is the
   embedding stream; class means over all frames). Project the unit
   direction out of the raw residual at the same stream point, at all
   positions, and report at peak and final checkpoints: delta-CM vs two
   non-target effects — mean |delta| of the other five families'
   heldout-conflict margins (length-normalized logprob_diff, frozen
   Scorer), and delta val-bpb on the first 16 deterministic ctx-length
   windows of the run's own shard stream. Specific = large CM hit,
   small control hits.

  python scripts/mech_patch.py runs/web_packed_v2_s42 [runs/...]
      [--battery data/probes/rvp3/battery.jsonl]

Writes runs/<run>/mech_patch.json per run.
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from mech_decomp import BUCKET, decompose  # noqa: E402

from fogen.data import load_tokenizer  # noqa: E402
from fogen.evals.bpb import evaluate_bpb, token_byte_table, val_stream  # noqa: E402
from fogen.evals.scoring import Scorer, load_battery  # noqa: E402
from fogen.model import GPT, ModelConfig, _rmsnorm, _rotary  # noqa: E402


def component_keys(cfg):
    keys = ["emb"]
    for i in range(cfg.n_layer):
        keys += [f"head:{i}:{h}" for h in range(cfg.n_head)]
        keys.append(f"mlp:{i}")
    return keys


def forward_components(model, idx, patch=None, record=False):
    """Mirror GPT.forward with per-component capture/substitution.
    patch: dict key -> tensor replacing that component's output
      emb      x0 = rmsnorm(wte(idx))           (B, T, C)
      head:l:h pre-wo head output y[:, h]       (B, T, Dh)
      mlp:l    residual increment m             (B, T, C)
    Returns (logits (B,T,V), cache or None)."""
    cfg = model.cfg
    patch = patch or {}
    cache = {} if record else None
    T = idx.size(1)
    cos, sin = model.rope_cos[:, :, :T], model.rope_sin[:, :, :T]
    H, Dh = cfg.n_head, cfg.head_dim
    x = _rmsnorm(model.wte(idx))
    if "emb" in patch:
        x = patch["emb"]
    if record:
        cache["emb"] = x.clone()
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
        for h in range(H):
            key = f"head:{i}:{h}"
            if key in patch:
                y = y.clone()
                y[:, h] = patch[key]
            if record:
                cache[key] = y[:, h].clone()
        x = x + at.wo(y.transpose(1, 2).reshape(B, T2, C))
        m = b.mlp(_rmsnorm(x))
        key = f"mlp:{i}"
        if key in patch:
            m = patch[key]
        if record:
            cache[key] = m.clone()
        x = x + m
    logits = model.lm_head(_rmsnorm(x)).float()
    return 15.0 * torch.tanh(logits / 15.0), cache


def forward_project(model, idx, u=None, at_k=None):
    """GPT.forward with the unit direction u projected out of the raw
    residual stream at point at_k (0 = after embedding, k = after block
    k-1's full block) at all positions. at_k=None: plain forward."""
    T = idx.size(1)
    cos, sin = model.rope_cos[:, :, :T], model.rope_sin[:, :, :T]

    def proj(x):
        return x - (x @ u)[..., None] * u

    x = _rmsnorm(model.wte(idx))
    if at_k == 0:
        x = proj(x)
    for i, b in enumerate(model.blocks):
        ve = (model.value_embeds[str(i)](idx)
              if str(i) in model.value_embeds else None)
        x = b(x, ve, cos, sin)
        if at_k == i + 1:
            x = proj(x)
    logits = model.lm_head(_rmsnorm(x)).float()
    return 15.0 * torch.tanh(logits / 15.0)


def load_state_into(model, path):
    state = {k: v.float() for k, v in load_file(str(path)).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("rope_") for k in missing), missing


def list_ckpt_steps(run):
    """Sorted checkpoint steps available for a run, local else S3."""
    local = sorted(Path(run).glob("ckpts/step*.safetensors"))
    if local:
        return [int(re.search(r"step(\d+)", p.name).group(1))
                for p in local]
    import boto3
    s3 = boto3.client("s3")
    steps = []
    kw = {"Bucket": BUCKET, "Prefix": f"{run}/ckpts/"}
    while True:
        r = s3.list_objects_v2(**kw)
        steps += [int(m.group(1)) for o in r.get("Contents", [])
                  if (m := re.search(r"step(\d+)\.safetensors$", o["Key"]))]
        if not r.get("IsTruncated"):
            break
        kw["ContinuationToken"] = r["NextContinuationToken"]
    assert steps, f"no checkpoints for {run}"
    return sorted(steps)


def get_ckpt(run, step):
    """Path to the checkpoint at `step`, local if present else a temp
    download from S3 (returns (path, cleanup_or_None))."""
    for p in sorted(Path(run).glob("ckpts/step*.safetensors")):
        if int(re.search(r"step(\d+)", p.name).group(1)) == step:
            return p, None
    import boto3
    s3 = boto3.client("s3")
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{run}/ckpts/")
    for o in r.get("Contents", []):
        m = re.search(r"step(\d+)\.safetensors$", o["Key"])
        if m and int(m.group(1)) == step:
            tmp = tempfile.NamedTemporaryFile(suffix=".safetensors",
                                              delete=False)
            tmp.close()
            s3.download_file(BUCKET, o["Key"], tmp.name)
            return Path(tmp.name), (lambda p=tmp.name: Path(p).unlink())
    raise FileNotFoundError(f"step {step} not found for {run}")


def cm_on(model, enc, prompts, she, he, u=None, at_k=None):
    tot = 0.0
    with torch.no_grad():
        for p in prompts:
            idx = torch.tensor([enc[p]])
            if at_k is None:
                lg, _ = forward_components(model, idx)
            else:
                lg = forward_project(model, idx, u=u, at_k=at_k)
            lp = F.log_softmax(lg[0, -1], dim=-1)
            tot += (lp[she] - lp[he]).item()
    return tot / len(prompts)


def family_margins(model, items, tok, u=None, at_k=None):
    """Mean length-normalized logprob_diff per family on heldout
    conflict items, via the frozen Scorer (optionally projected)."""
    def logits_fn(ids):
        if at_k is None:
            lg, _ = forward_components(model, ids)
        else:
            lg = forward_project(model, ids, u=u, at_k=at_k)
        return lg
    sc = Scorer(encode=lambda s: tok.encode(s).ids, logits_fn=logits_fn)
    with torch.no_grad():
        rows = sc.score_items(items)
    fams = {}
    for r in rows:
        fams.setdefault(r["probe"].rsplit(".", 1)[0], []).append(
            r["logprob_diff"])
    return {f: sum(v) / len(v) for f, v in sorted(fams.items())}


class _ProjModel:
    def __init__(self, model, u, at_k):
        self.model, self.u, self.at_k = model, u, at_k

    def __call__(self, x):
        return forward_project(self.model, x, u=self.u, at_k=self.at_k)


def fit_directions(model, enc, femcue, masccue, she, he):
    """d_k (k = 0..n_layer) from normalized last-position residuals."""
    L = model.cfg.n_layer
    fem = [[] for _ in range(L + 1)]
    masc = [[] for _ in range(L + 1)]
    with torch.no_grad():
        for prompts, store in ((femcue, fem), (masccue, masc)):
            for p in prompts:
                *_, hs = decompose(model, torch.tensor([enc[p]]), she, he)
                for k, h in enumerate(hs):
                    store[k].append(h)
    return [torch.stack(fem[k]).mean(0) - torch.stack(masc[k]).mean(0)
            for k in range(L + 1)]


def score_run(run, battery):
    run = Path(run)
    assert not re.search(r"s10\d\d$", run.name), \
        f"{run} looks like a quarantined replication seed; refusing"
    cfg = yaml.safe_load(open(run / "config_used.yaml"))
    tok = load_tokenizer(cfg["data"]["tokenizer_dir"])

    margins = [json.loads(l) for l in open(run / "mech_margins.jsonl")]
    peak_step = max(margins, key=lambda r: r["CM"])["step"]
    final_step = max(r["step"] for r in margins)
    assert peak_step != final_step, "peak == final; nothing to patch"

    she, he = tok.encode(" she").ids, tok.encode(" he").ids
    assert len(she) == 1 and len(he) == 1
    she, he = she[0], he[0]

    items = load_battery(battery)
    femcue = sorted({it["prefix"] for it in items
                     if it["probe"] == "pronoun_gender_ref.conflict"
                     and it["split"] == "heldout"})
    masccue = sorted({it["prefix"] for it in items
                      if it["probe"] == "pronoun_gender_ref.agree"
                      and it["split"] == "heldout"})
    other = [it for it in items
             if it["probe"].endswith(".conflict")
             and it["split"] == "heldout"
             and not it["probe"].startswith("pronoun_gender_ref")]
    assert femcue and masccue and other
    enc = {p: tok.encode(p).ids for p in femcue + masccue}

    mc = ModelConfig(**cfg["model"])
    models = {}
    for name, step in (("peak", peak_step), ("final", final_step)):
        m = GPT(mc)
        m.eval()
        path, cleanup = get_ckpt(str(run), step)
        load_state_into(m, path)
        if cleanup:
            cleanup()
        models[name] = m

    # ---- D. patching, both directions
    caches = {}
    base_cm = {}
    for name, m in models.items():
        caches[name] = {}
        with torch.no_grad():
            for p in femcue:
                _, caches[name][p] = forward_components(
                    m, torch.tensor([enc[p]]), record=True)
        base_cm[name] = cm_on(m, enc, femcue, she, he)
    print(f"{run.name}: peak step {peak_step} CM={base_cm['peak']:+.3f}, "
          f"final step {final_step} CM={base_cm['final']:+.3f}", flush=True)

    patch_out = {}
    for direction, src, dst in (("peak_into_final", "peak", "final"),
                                ("final_into_peak", "final", "peak")):
        res = {}
        denom = base_cm[src] - base_cm[dst]
        for key in component_keys(mc):
            tot = 0.0
            with torch.no_grad():
                for p in femcue:
                    lg, _ = forward_components(
                        models[dst], torch.tensor([enc[p]]),
                        patch={key: caches[src][p][key]})
                    lp = F.log_softmax(lg[0, -1], dim=-1)
                    tot += (lp[she] - lp[he]).item()
            cm_p = tot / len(femcue)
            res[key] = {"CM_patch": cm_p,
                        "recovery": (cm_p - base_cm[dst]) / denom
                        if denom != 0 else None}
        patch_out[direction] = res
        top = max(res, key=lambda k: res[k]["recovery"] or -9e9)
        print(f"  {direction}: top component {top} "
              f"recovery={res[top]['recovery']:+.3f}", flush=True)

    # ---- E. direction-specificity at peak and final
    stream = val_stream(cfg["data"]["shard_dir"])
    tb = token_byte_table(tok)
    spec_out = {}
    for name, m in models.items():
        dirs = fit_directions(m, enc, femcue, masccue, she, he)
        base_f = family_margins(m, other, tok)
        base_bpb = evaluate_bpb(m, stream, tb, mc.ctx_len,
                                max_windows=16)["val_bpb"]
        rows = []
        for k, d in enumerate(dirs):
            if d.norm() < 1e-8:
                # k=0 is the embedding stream: the last-position state is
                # the shared frame-final token, so the class difference
                # vanishes there and no direction is defined
                rows.append({"k": k, "d_norm": d.norm().item(),
                             "CM_proj": None, "dCM": None,
                             "fam_delta": None,
                             "other_fam_mean_absdelta": None,
                             "val_bpb_proj": None, "d_val_bpb": None})
                print(f"  spec[{name}] k={k} d_norm=0; skipped", flush=True)
                continue
            u = d / d.norm()
            cm_proj = cm_on(m, enc, femcue, she, he, u=u, at_k=k)
            fam = family_margins(m, other, tok, u=u, at_k=k)
            bpb = evaluate_bpb(_ProjModel(m, u, k), stream, tb,
                               mc.ctx_len, max_windows=16)["val_bpb"]
            dfam = {f: fam[f] - base_f[f] for f in fam}
            rows.append({
                "k": k, "d_norm": d.norm().item(),
                "CM_proj": cm_proj, "dCM": cm_proj - base_cm[name],
                "fam_delta": dfam,
                "other_fam_mean_absdelta":
                    sum(abs(v) for v in dfam.values()) / len(dfam),
                "val_bpb_proj": bpb, "d_val_bpb": bpb - base_bpb})
            print(f"  spec[{name}] k={k} dCM={rows[-1]['dCM']:+.3f} "
                  f"fam|d|={rows[-1]['other_fam_mean_absdelta']:.3f} "
                  f"dbpb={rows[-1]['d_val_bpb']:+.4f}", flush=True)
        spec_out[name] = {"base_fam_margins": base_f,
                          "base_val_bpb": base_bpb, "per_k": rows}

    out = {"peak_step": peak_step, "final_step": final_step,
           "CM_peak": base_cm["peak"], "CM_final": base_cm["final"],
           "n_femcue": len(femcue), "n_other_items": len(other),
           "patch": patch_out, "specificity": spec_out}
    out_path = run / "mech_patch.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--battery", default="data/probes/rvp3/battery.jsonl")
    args = ap.parse_args()
    for run in args.run_dirs:
        score_run(run, args.battery)


if __name__ == "__main__":
    main()
