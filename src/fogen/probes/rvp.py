"""Rule-vs-prior probe families (Gate A battery, pivot of 2026-06-10).

Each family tests whether a low-frequency RULE survives competition with a
high-frequency surface PRIOR. Every family ships two conditions:

  conflict — the rule's answer opposes the prior (the diagnostic)
  agree    — the rule's answer coincides with the prior (the control)

A model applying the rule scores high on both; a model running a global
surface preference scores high on exactly one (and a preference FLIP moves
the two in opposite directions). Family verdicts are only valid at
checkpoints where the agree control holds (threshold registered in
RESEARCH_LOG before scoring).

On-disk schema matches v1_probes: probe name carries the condition as
"<family>.<condition>" so all existing scoring/aggregation tooling works
unchanged. train/heldout = disjoint template frames, same construction.
"""

import itertools
import json
import random
from pathlib import Path

GIRLS = ["Lily", "Mia", "Anna", "Emma", "Sue", "Jane"]
BOYS = ["Tom", "Ben", "Max", "Sam", "Jack", "Tim"]
NAMES = GIRLS + BOYS
VOWEL_NOUNS = ["apple", "orange", "egg", "ant", "owl", "umbrella", "elephant", "acorn"]
CONS_NOUNS = ["ball", "dog", "cat", "book", "toy", "cake", "bird", "frog"]
PLUR_NOUNS = ["dogs", "cats", "birds", "frogs", "boys", "girls", "toys", "ducks"]
SING_NOUNS = ["dog", "cat", "bird", "frog", "boy", "girl", "toy", "duck"]

# (base, irregular_past, overregularized) — no forms that collide with real
# words (dropped see->sawed/seed)
IRREGULARS = [("go", "went", "goed"), ("run", "ran", "runned"),
              ("eat", "ate", "eated"), ("fall", "fell", "falled"),
              ("come", "came", "comed"), ("make", "made", "maked"),
              ("take", "took", "taked"), ("give", "gave", "gived"),
              ("tell", "told", "telled")]
REGULARS = ["play", "jump", "walk", "look", "want", "help", "laugh", "smile"]


def _mk(family, condition, split, template_id, prefix, correct, distractor):
    return {"family": family, "condition": condition,
            "probe": f"{family}.{condition}",
            "category": "rule_vs_prior",
            "template_id": template_id, "split": split,
            "prefix": prefix, "correct": correct, "distractor": distractor,
            "chance": 0.5}


def det_an_choice():
    """Rule: 'an' before vowel onset. Prior: 'a' (~10x more frequent).
    Determiner-contrast continuations share the noun, isolating the rule."""
    frames_tr = ["{n} ate", "{n} saw", "{n} found", "{n} had"]
    frames_ho = ["{n} got", "{n} held", "{n} wanted", "One day, {n} saw"]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for fr, name in itertools.product(frames, NAMES[:6]):
            p = fr.format(n=name)
            tid = fr.replace("{n} ", "").replace("One day, ", "oneday_")
            for vn in VOWEL_NOUNS[:4]:
                out.append(_mk("det_an_choice", "conflict", split, tid,
                               p, f" an {vn}", f" a {vn}"))
            for cn in CONS_NOUNS[:4]:
                out.append(_mk("det_an_choice", "agree", split, tid,
                               p, f" a {cn}", f" an {cn}"))
    return out


def a_an_adjective():
    """Compositional a/an: the determiner agrees with the ADJECTIVE onset,
    not the noun. Lower support than bare det+noun."""
    frames_tr = ["{n} saw", "{n} found"]
    frames_ho = ["{n} had", "{n} got"]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for fr, name in itertools.product(frames, NAMES[:6]):
            p = fr.format(n=name)
            tid = fr.replace("{n} ", "")
            for cn in CONS_NOUNS[:4]:
                out.append(_mk("a_an_adjective", "conflict", split, tid,
                               p, f" an old {cn}", f" a old {cn}"))
            for vn in VOWEL_NOUNS[:4]:
                out.append(_mk("a_an_adjective", "agree", split, tid,
                               p, f" a big {vn}", f" an big {vn}"))
    return out


def irregular_past():
    """Rule: irregular past forms. Prior: the dominant -ed pattern."""
    frames_tr = ["Yesterday, {n}"]
    frames_ho = ["One day, {n}", "Last night, {n}"]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for fr, name in itertools.product(frames, NAMES[:6]):
            p = fr.format(n=name)
            tid = fr.split(",")[0].lower().replace(" ", "")
            for base, irr, overreg in IRREGULARS:
                out.append(_mk("irregular_past", "conflict", split,
                               f"{tid}_{base}", p, f" {irr}", f" {overreg}"))
            for reg in REGULARS[:6]:
                out.append(_mk("irregular_past", "agree", split, tid,
                               p, f" {reg}ed", f" {reg}"))
    return out


def plural_was_were():
    """Rule: adjacent number agreement. Prior: 'was' (more frequent).
    Replaces the retired relative_clause probe; NO attractors."""
    frames_tr = ["The {x}", "All the {x}", "The little {x}"]
    frames_ho = ["The two {x}", "{name}'s {x}", "The big {x}"]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for fr in frames:
            tid = {"The {x}": "the", "All the {x}": "all",
                   "The little {x}": "little", "The big {x}": "big",
                   "The two {x}": "two", "{name}'s {x}": "poss"}[fr]
            for pl in PLUR_NOUNS:
                p = fr.format(x=pl, name="Tom")
                out.append(_mk("plural_was_were", "conflict", split, tid,
                               p, " were", " was"))
            if "two" in fr or "All" in fr:
                continue  # frames that force plural get no agree items
            for sg in SING_NOUNS:
                p = fr.format(x=sg, name="Tom")
                out.append(_mk("plural_was_were", "agree", split, tid,
                               p, " was", " were"))
    return out


def negation_bare_verb():
    """Rule: bare verb after do-support. Prior: -ed past in narrative."""
    frames_tr = [("{n} did not", "want"), ("{n} did not", "play"),
                 ("{n} did not", "jump"), ("{n} did not", "help")]
    frames_ho = [("{n} could not", "find"), ("{n} could not", "open"),
                 ("{n} would not", "stop"), ("{n} could not", "reach")]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for (fr, verb), name in itertools.product(frames, NAMES):
            p = fr.format(n=name)
            aux = fr.split()[1]  # did/could/would
            past = {"find": "found"}.get(verb, verb + "ed")
            out.append(_mk("negation_bare_verb", "conflict", split,
                           f"{aux}_{verb}", p, f" {verb}", f" {past}"))
    # agree: simple narrative past, -ed prior and rule coincide
    for split, frames in (("train", ["Yesterday, {n}"]),
                          ("heldout", ["After lunch, {n}"])):
        for fr, name, reg in itertools.product(frames, NAMES[:6], REGULARS[:4]):
            p = fr.format(n=name)
            out.append(_mk("negation_bare_verb", "agree", split,
                           fr.split(",")[0].lower().replace(" ", ""),
                           p, f" {reg}ed", f" {reg}"))
    return out


def reflexive_gender():
    """Rule: reflexive agrees with antecedent gender. Prior direction
    (himself vs herself) measured from corpus; conditions assigned per
    antecedent so the corpus-dominant form defines 'agree'."""
    verbs_tr = ["hurt", "enjoyed", "dressed", "splashed"]
    verbs_ho = ["washed", "surprised", "taught", "saw"]
    out = []
    for split, verbs in (("train", verbs_tr), ("heldout", verbs_ho)):
        for v in verbs:
            for g in GIRLS:
                out.append(_mk("reflexive_gender", "conflict", split, v,
                               f"{g} {v}", " herself", " himself"))
            for b in BOYS:
                out.append(_mk("reflexive_gender", "agree", split, v,
                               f"{b} {v}", " himself", " herself"))
    return out


def pronoun_gender_ref():
    """High-support reference family (anchor near the top of the frequency
    axis; expected RECOVERED in every cell)."""
    frames_tr = ["{n} smiled because", "{n} was happy because"]
    frames_ho = ["{n} laughed when", "{n} cried because"]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for fr in frames:
            tid = fr.split()[1]
            for g in GIRLS:
                out.append(_mk("pronoun_gender_ref", "conflict", split, tid,
                               fr.format(n=g), " she", " he"))
            for b in BOYS:
                out.append(_mk("pronoun_gender_ref", "agree", split, tid,
                               fr.format(n=b), " he", " she"))
    return out


def comparative_er():
    """Rule: short adjectives take -er, not 'more X'. EXPLORATORY: the
    continuation pair is length-asymmetric; scored on raw sequence logprob
    like all items, but flagged in analysis."""
    adjs = ["big", "small", "fast", "tall"]
    long_adjs = ["beautiful", "careful"]
    frames_tr = ["The {x} was even"]
    frames_ho = ["It got even"]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for fr in frames:
            tid = "was_even" if "was" in fr else "got_even"
            for a, n in itertools.product(adjs, SING_NOUNS[:6]):
                er = {"big": "bigger", "small": "smaller",
                      "fast": "faster", "tall": "taller"}[a]
                p = fr.format(x=n)
                out.append(_mk("comparative_er", "conflict", split,
                               f"{tid}_{a}", p, f" {er}", f" more {a}"))
            for la, n in itertools.product(long_adjs, SING_NOUNS[:6]):
                p = fr.format(x=n)
                out.append(_mk("comparative_er", "agree", split,
                               f"{tid}_{la}", p, f" more {la}", f" {la}er"))
    return out


def modal_agreement():
    """rev2 (registered 2026-06-10, before any web ckpt existed): controlled
    analog of v1's modal_continuation — the family with the strongest
    final-state displacement on web (stuck 0.63-0.69 in 6/6 seeds).
    Rule: bare verb after a modal. Prior: 3sg -s after a singular subject.
    Continuations are length-matched (verb vs verb+s), unlike the
    to-infinitive contrast, so init-time length bias does not confound
    emergence."""
    verbs = ["play", "jump", "run", "sleep", "sing", "swim", "eat", "walk"]
    modals_tr = ["can", "must"]
    modals_ho = ["should", "might"]
    advs_tr = ["always", "often"]
    advs_ho = ["usually", "never"]
    out = []
    for split, modals, advs in (("train", modals_tr, advs_tr),
                                ("heldout", modals_ho, advs_ho)):
        for m, name, v in itertools.product(modals, NAMES[:6], verbs):
            out.append(_mk("modal_agreement", "conflict", split, m,
                           f"{name} {m}", f" {v}", f" {v}s"))
        for adv, name, v in itertools.product(advs, NAMES[:6], verbs):
            out.append(_mk("modal_agreement", "agree", split, adv,
                           f"{name} {adv}", f" {v}s", f" {v}"))
    return out


def modal_agreement_v2():
    """rev3: modal_agreement with a corpus-neutral agree control. rev2's
    agree frame ("{name} always" -> " plays") failed in ALL cells (per-
    template 0.04-0.69 at final ckpts): adverb-adjacent position does not
    elicit the 3sg -s prior. Habitual temporal frames do. Conflict items
    re-issued under the v2 family name so the classifier pairs them with
    the fixed control."""
    verbs = ["play", "jump", "run", "sleep", "sing", "swim", "eat", "walk"]
    modals_tr = ["can", "must"]
    modals_ho = ["should", "might"]
    # rvp3.1 (2026-06-10, amended BEFORE any governed-cell scoring): the
    # initial "Most days/evenings" frames sat at 0.60/0.62 per template on
    # pythia-70m@final (public smoke) vs 0.88/0.81 for the "Every" frames.
    frames_tr = ["Every day {n}", "Every afternoon {n}"]
    frames_ho = ["Every morning {n}", "Every night {n}"]
    out = []
    for split, modals, frames in (("train", modals_tr, frames_tr),
                                  ("heldout", modals_ho, frames_ho)):
        for m, name, v in itertools.product(modals, NAMES[:6], verbs):
            out.append(_mk("modal_agreement_v2", "conflict", split, m,
                           f"{name} {m}", f" {v}", f" {v}s"))
        for fr, name, v in itertools.product(frames, NAMES[:6], verbs):
            tid = fr.split(" {")[0].lower().replace(" ", "")
            out.append(_mk("modal_agreement_v2", "agree", split, tid,
                           fr.format(n=name), f" {v}s", f" {v}"))
    return out


def irregular_past_v2():
    """rev3: irregular_past with a corpus-neutral agree control. The rev1
    agree frame ("Yesterday, {n}" -> " {reg}ed") sits at 0.67/template on
    web (TinyStories-narrative framing). Coordination after a past verb
    predicts past tense in both corpora. Conflict items re-issued."""
    frames_tr = ["Yesterday, {n}"]
    frames_ho = ["One day, {n}", "Last night, {n}"]
    agree_tr = ["{n} smiled and"]
    agree_ho = ["{n} stopped and", "{n} turned and"]
    out = []
    for split, frames, agrees in (("train", frames_tr, agree_tr),
                                  ("heldout", frames_ho, agree_ho)):
        for fr, name in itertools.product(frames, NAMES[:6]):
            p = fr.format(n=name)
            tid = fr.split(",")[0].lower().replace(" ", "")
            for base, irr, overreg in IRREGULARS:
                out.append(_mk("irregular_past_v2", "conflict", split,
                               f"{tid}_{base}", p, f" {irr}", f" {overreg}"))
        for fr, name, reg in itertools.product(agrees, NAMES[:6], REGULARS[:6]):
            stem = fr.split()[1].rstrip("d")  # smiled -> smile etc.
            if reg.startswith(stem[:4]):
                continue  # avoid "smiled and smiled"
            tid = "and_" + fr.split()[1]
            out.append(_mk("irregular_past_v2", "agree", split, tid,
                           fr.format(n=name), f" {reg}ed", f" {reg}"))
    return out


def negation_bare_verb_v2():
    """rev3: negation_bare_verb with a corpus-neutral agree control. The
    rev1 agree frame ("Yesterday, {n}"/"After lunch, {n}") sits at 0.50/
    template on web. Perfect-tense 'had' demands the -ed form in both
    corpora. Conflict items re-issued."""
    frames_tr = [("{n} did not", "want"), ("{n} did not", "play"),
                 ("{n} did not", "jump"), ("{n} did not", "help")]
    frames_ho = [("{n} could not", "find"), ("{n} could not", "open"),
                 ("{n} would not", "stop"), ("{n} could not", "reach")]
    agree_tr = ["{n} had", "{n} had just"]
    agree_ho = ["{n} had already", "By then {n} had"]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for (fr, verb), name in itertools.product(frames, NAMES):
            p = fr.format(n=name)
            aux = fr.split()[1]
            past = {"find": "found"}.get(verb, verb + "ed")
            out.append(_mk("negation_bare_verb_v2", "conflict", split,
                           f"{aux}_{verb}", p, f" {verb}", f" {past}"))
    for split, agrees in (("train", agree_tr), ("heldout", agree_ho)):
        for fr, name, reg in itertools.product(agrees, NAMES[:6], REGULARS[:4]):
            tid = fr.replace("{n} ", "").replace(" {n}", "").lower().replace(" ", "")
            out.append(_mk("negation_bare_verb_v2", "agree", split, tid,
                           fr.format(n=name), f" {reg}ed", f" {reg}"))
    return out


# rvp4.1 (2026-06-10, amended BEFORE any governed-cell scoring): Nora ->
# Zoe, Owen -> Billy. Original picks were the only two names multi-token
# in the TinyStories bpe8192 vocab (i.e. corpus-rare; weak gender prior
# risks agree-control noise). Replacements are single-token there and
# correctly gendered by pythia-70m@final.
GIRLS_V2 = GIRLS + ["Lucy", "Ella", "Rose", "Grace", "Amy", "Kate",
                    "Sara", "Beth", "Daisy", "Molly", "Ruby", "Zoe"]
BOYS_V2 = BOYS + ["Leo", "Jake", "Adam", "Mark", "Paul", "Pete",
                  "Nick", "John", "Finn", "Billy", "Fred", "Joe"]
FEM_NOUNS = ["girl", "woman", "mom", "queen", "princess", "sister",
             "grandma", "aunt"]
MASC_NOUNS = ["boy", "man", "dad", "king", "prince", "brother",
              "grandpa", "uncle"]


def pronoun_gender_ref_v2():
    """rev4: pronoun_gender_ref widened from 12 to 36 names and 4 to 8
    frames (critique C1: per-template conflict cells averaged 6 items,
    too thin to separate displacement from item noise)."""
    frames_tr = ["{n} smiled because", "{n} was happy because",
                 "{n} jumped because", "{n} ran home because"]
    frames_ho = ["{n} laughed when", "{n} cried because",
                 "{n} was sad because", "{n} stopped when"]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for fr in frames:
            tid = "_".join(fr.split()[1:3]).rstrip("_")
            for g in GIRLS_V2:
                out.append(_mk("pronoun_gender_ref_v2", "conflict", split,
                               tid, fr.format(n=g), " she", " he"))
            for b in BOYS_V2:
                out.append(_mk("pronoun_gender_ref_v2", "agree", split, tid,
                               fr.format(n=b), " he", " she"))
    return out


def reflexive_gender_v2():
    """rev4: reflexive_gender widened to the 36-name set (critique C1)."""
    verbs_tr = ["hurt", "enjoyed", "dressed", "splashed"]
    verbs_ho = ["washed", "surprised", "taught", "saw"]
    out = []
    for split, verbs in (("train", verbs_tr), ("heldout", verbs_ho)):
        for v in verbs:
            for g in GIRLS_V2:
                out.append(_mk("reflexive_gender_v2", "conflict", split, v,
                               f"{g} {v}", " herself", " himself"))
            for b in BOYS_V2:
                out.append(_mk("reflexive_gender_v2", "agree", split, v,
                               f"{b} {v}", " himself", " herself"))
    return out


def pronoun_gender_noun():
    """rev4 discriminator (critique C2): gender cue is a frequent COMMON
    NOUN, not a proper name. Frames mirror pronoun_gender_ref exactly.
    If name-cued conflict collapses while noun-cued conflict survives,
    the mechanism is name-embedding forgetting; if both collapse, the
    female-cue->she mapping itself is displaced by the 'he' prior."""
    frames_tr = ["The {x} smiled because", "The {x} was happy because"]
    frames_ho = ["The {x} laughed when", "The {x} cried because"]
    out = []
    for split, frames in (("train", frames_tr), ("heldout", frames_ho)):
        for fr in frames:
            tid = "_".join(fr.split()[2:4])
            for fn_ in FEM_NOUNS:
                out.append(_mk("pronoun_gender_noun", "conflict", split,
                               tid, fr.format(x=fn_), " she", " he"))
            for mn in MASC_NOUNS:
                out.append(_mk("pronoun_gender_noun", "agree", split, tid,
                               fr.format(x=mn), " he", " she"))
    return out


ALL_FAMILIES = [det_an_choice, a_an_adjective, irregular_past,
                plural_was_were, negation_bare_verb, reflexive_gender,
                pronoun_gender_ref, comparative_er]
REV2_FAMILIES = ALL_FAMILIES + [modal_agreement]
REV3_FAMILIES = REV2_FAMILIES + [modal_agreement_v2, irregular_past_v2,
                                 negation_bare_verb_v2]
REV4_FAMILIES = REV3_FAMILIES + [pronoun_gender_ref_v2, reflexive_gender_v2,
                                 pronoun_gender_noun]
_REV = {1: ALL_FAMILIES, 2: REV2_FAMILIES, 3: REV3_FAMILIES,
        4: REV4_FAMILIES}


def build_rvp_battery(seed: int = 0, rev: int = 1) -> list[dict]:
    items = []
    for fn in _REV[rev]:
        fam = fn()
        rng = random.Random(f"{fn.__name__}-{seed}")
        rng.shuffle(fam)
        items.extend(fam)
    for i, it in enumerate(items):
        it["item_id"] = f"{it['probe']}/{it['split']}/{i:04d}"
    return items


def write_rvp_battery(out_path: str | Path, seed: int = 0, rev: int = 1) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = build_rvp_battery(seed, rev)
    with out_path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return len(items)
