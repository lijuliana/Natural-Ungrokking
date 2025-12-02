# Prereg draft inputs (working doc — NOT the frozen prereg)

Drafted 2026-06-10 during harness construction, before any v2 sweep analysis.
These go into PREREGISTRATION.md at the Week-2 freeze after pilot validation.

## Proposed transience metric parameters (from v1 paper conventions)
- Smoothing: k=3 rolling mean; skip first 10 evals for peak search.
- final = mean accuracy over last 25% of eval steps.
- transience = (peak − final) / (peak − chance); chance = 0.5 (binary pairs).
- Classification threshold T = 0.15 (v1's; ~1.7 binomial SEs at largest probe) —
  revisit against pilot noise floor before freezing.
- Non-monotonicity: peak strictly before the final-window start.
- Bootstrap: 2000 resamples over items, 95% CI; transient iff CI excludes T.
- Reproducibility: σ/μ ≤ 0.20 of peak step across seeds (|σ| ≤ 25 steps for near-zero peaks).

## Capability categorization, axis A (linguistic type) — as in v1
- lexical/associative: determiner_a_an, comparative_than, proper_noun_completion,
  pronoun_gender, numeric_sequence, common_idiom
- structural: end_of_sentence, modal_continuation, adjective_order,
  past_tense_consistency, subj_verb_agreement, relative_clause_agreement,
  close_quote, reflexive_pronoun
- Axis B (support frequency) comes from fogen.theory.frequency on each corpus.
  DONE for TinyStories (full corpus, 364,237,317 words):
  analysis/frequency/tinystories/{frequency.json,capability_table.csv}.
  Support spans 3.5 orders of magnitude (numeric_sequence 28/M …
  end_of_sentence 89,928/M). Exploratory observation (NOT a confirmatory
  claim): the three v1 transients sit at contested/intermediate support —
  end_of_sentence ratio 0.66, modal_continuation 0.83, adjective_order
  202/M support — while stable lexical probes have high support and/or
  ratio ≈ 1.0, and relative_clause (30/M), which never emerges in v1, is
  rarest among structural probes. Consistent with the phase-diagram
  hypothesis; to be tested properly in the v2 sweep.

## v1 expected outcomes (sanity anchors for the reproduction)
- Transient in all 5 v1 seeds: end_of_sentence (~0.74→0.24), modal_continuation
  (~0.87→0.66), adjective_order (~0.89→0.64); peaks between steps 180–2130.
- Lexical probes: stable, emergence-time σ/μ < 0.30.

## Proposed seeds
- Behavioral/development seeds: 42, 123, 7, 5, 17 (v1's, for comparability).
- Mechanistic seeds: 17, 99 (v1's).
- QUARANTINED replication seeds (proposal): 1001, 2002, 3003 — never evaluated
  until Week 10. To be confirmed at freeze.

## Exclusion rules (draft)
- Exclude a run only for: NaN/divergence, corrupted checkpoint (hash mismatch),
  incomplete training. Never for outcome. All exclusions logged in DECISIONS.md.

## Known weaknesses — status after battery_rev=2 (2026-06-10)
battery_rev=2 generated (1,270 items, data/probes/v2/battery.jsonl,
sha256 17710a834d5f883b955c8f2ba25a0ae83262210321f7fa0d9ef6d5cadd2f84b9).
rev=1 remains byte-identical to the frozen v1 file (regression-tested);
the v1 reproduction runs continue to use rev=1 for comparability.
- FIXED: reflexive_pronoun held-out split 4 → 32 items (new verbs +
  proud_of/mirror/kids frames).
- FIXED: probes with < 40 train items expanded; rev=2 guarantees ≥ 40
  train and ≥ 16 heldout items per probe (tested in test_core.py).
- FIXED: common_idiom heldout prefixes de-duplicated.
- FIXED: close_quote heldout now minimal pairs (correct vs distractor
  differ only in punctuation order, len ≤ 2); the old "said {name}"
  frame is rev=1-only.
- OPEN: probe items are reconstructions from the v1 paper, not the
  original files; if original v1 item files are recovered, reconcile
  and log differences.
- Decision for freeze: v2 sweep uses battery_rev=2; v1-comparison
  figures use rev=1.

## v1 reproduction status (2026-06-10 overnight) — input to prereg framing
The v1 fidelity gate is UNCLOSABLE from the paper spec alone (see
RESEARCH_LOG 2026-06-10 entries): spec-faithful packed repro on real
TinyStories gives val_bpb 0.595 vs v1's claimed 1.149-1.152 with all probes
saturated; both one-doc regimes fail pathologically at v1's hparams. v1's
corpus ("TinyStories-like") and val protocol are unstated. Implications for
prereg:
- Do NOT anchor prereg quantitative predictions to v1's Table 2/9 numbers;
  treat v1 as qualitative motivation (transience exists) located in an
  unknown phase-diagram cell.
- The controlled-sweep D/N axis (first cell: databudget_dn5) replaces the
  v1 cell as the transience proof-of-life for the Week-2 GO/NO-GO gate
  (PLAN.md already allowed this fallback).
- If v1 artifacts (code/corpus/ckpts — paper §6 says "released") surface,
  reconcile and log; do not hold the freeze for them.

## Battery design requirements learned from rev1/rev2/rev3 (2026-06-10)
- Splits within a probe MUST be exchangeable samples of one construction;
  rev1's relative_clause mixed object-RC (train) with PP-attractor (heldout)
  and produced uninterpretable cross-split divergence. Prereg battery:
  separate probes per construction (rc_attractor, pp_attractor), each with
  balanced sing/plur heads and >= 30 items per (template, split).
- Agreement probes need bias controls: any singular-only or plural-only
  item set confounds syntax with a blanket verb-form preference.
- Power floor: >= 30 items per reported trajectory; n=4/n=12 splits in rev1
  generated spurious transience flags in every run.
- Step-0 tie artifact: zero-init lm_head -> all-zero logits -> argmax ties
  scored incorrect; exclude step 0 from trajectory fits or score ties at
  chance.
