# Natural Ungrokking

Code for the paper **"Natural Ungrokking: Asymmetric Control of Which
Rules Survive Pretraining"** (Li & Sreedhar, 2026).

Paper: [arXiv link forthcoming]

Midway through pretraining, small language models learn linguistic
rules and then lose them, with no trace in the loss curve. This repo
contains everything needed to reproduce the experiments: training,
probe batteries, mechanism instruments, corpus-edit interventions
(kill and rescue), the public-checkpoint suite, and the figure and
table generators. Every threshold and prediction was pre-registered
(`prereg/PREREGISTRATION.md`) before the outcome data existed.

## Layout

```
src/fogen/          installable package
  training/         4-layer decoder training loop (python -m fogen.training.train)
  probes/           rule-vs-prior forced-choice probe batteries (rvp)
  evals/            bits-per-byte, probe scoring, public-checkpoint suite
  analysis/         support-frequency counting, phase classification
  theory/           critical-frequency model
  crosscoders/ llc/ mechanism instruments (model diffing, refined LLC)
scripts/            every experiment, evaluator, figure, and table;
                    eval_*.py scripts apply the frozen registered checks
configs/            single source of truth for all hyperparameters
prereg/             the frozen pre-registration document
data/               probe items, tokenizer assets, smoke-test corpus
infra/skypilot/     job templates used for the training grid
analysis/           corpus frequency counts
```

## Install

```
pip install -e .
pytest src scripts          # unit tests live next to the code
```

## Reproduce

Each stage reads configs and prior-stage artifacts; nothing is
hand-entered downstream.

1. **Corpora** — `scripts/prepare_tinystories.py`,
   `scripts/prepare_climbmix.py` (deterministic, seed 0).
2. **Train a grid cell** —
   `python -m fogen.training.train --config configs/web_packed.yaml --seed 42`
   (cells: `{v1_repro, ts_packed_armB, databudget_*, web_*}`, seeds 42-44).
3. **Probes and margins** — checkpoint scoring with the frozen rvp3.1
   battery (`scripts/score_ckpts.py`) and the contrast-margin
   instrument (`scripts/mech_margins.py`).
4. **Interventions** — kill: `scripts/build_kill_shards.py`,
   `scripts/build_an_kill_shards.py`; rescue:
   `scripts/gen_rescue_docs.py`; orchestrated by
   `scripts/step6_run_node.sh` and `scripts/step6t_run_node.sh`.
5. **Registered verdicts** — `scripts/eval_m4.py`,
   `scripts/eval_step6.py`, `scripts/eval_step6t.py`,
   `scripts/eval_predictions.py` emit the pass/fail/void scoreboard.
6. **Public checkpoints** — `scripts/eval_public_suite.py` scores
   Pythia and OLMo with the same frozen probes.
7. **Figures and tables** — `scripts/fig_*.py` and
   `scripts/make_paper_tables.py` read only the artifacts above.

Training run artifacts (checkpoints, probe logs, evaluator outputs)
are not stored in this repository; an archived bundle is linked from
the paper.

## Citation

```bibtex
@article{li2026naturalungrokking,
  title  = {Natural Ungrokking: Asymmetric Control of Which Rules
            Survive Pretraining},
  author = {Li, Juliana and Sreedhar, Diya},
  journal= {arXiv preprint},
  year   = {2026}
}
```
