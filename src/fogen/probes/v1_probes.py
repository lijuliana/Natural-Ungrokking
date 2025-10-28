"""The 14 v1 probes as minimal-pair item generators.

Item = (prefix, correct, distractor); scored by comparing continuation
logprobs. Each probe has TRAIN templates and HELD-OUT paraphrase templates
(disjoint frames, same construction) per the v2 design. Items are
deterministic given the seed; JSONL is the on-disk format.

Reconstructed from the v1 paper's probe names/descriptions (original item
files not yet recovered); 6 lexical/associative + 8 structural.

battery_rev: rev=1 reproduces data/probes/v1/battery.jsonl exactly (frozen).
rev=2 expands the weak probes flagged in prereg/draft_inputs.md (reflexive
held-out, <40-train-item probes, duplicated common_idiom held-out prefixes,
close_quote held-out long-suffix pairs) without touching rev-1 output.
"""

import itertools
import json
import random
from pathlib import Path

GIRLS = ["Lily", "Mia", "Anna", "Emma", "Sue", "Jane"]
BOYS = ["Tom", "Ben", "Max", "Sam", "Jack", "Tim"]
VOWEL_NOUNS = ["apple", "orange", "egg", "ant", "owl", "umbrella", "elephant", "acorn"]
CONS_NOUNS = ["ball", "dog", "cat", "book", "toy", "cake", "bird", "frog"]
PLACES = ["park", "store", "garden", "beach", "forest", "kitchen"]
ADJ_SIZE = ["big", "small", "little", "huge", "tiny"]
ADJ_COLOR = ["red", "blue", "green", "yellow", "brown"]
SING_NOUNS = ["dog", "cat", "bird", "girl", "boy", "frog"]
PAST_PAIRS = [("went", "goes"), ("ran", "runs"), ("saw", "sees"),
              ("ate", "eats"), ("found", "finds"), ("made", "makes")]
COMPLETE_CLAUSES_TRAIN = [
    "The cat sat on the mat", "Tom played in the park", "Lily ate the cake",
    "The dog ran to the tree", "Ben found a shiny stone", "The sun was warm",
    "Mia hugged her mom", "The bird flew over the house", "Sam opened the door",
]
COMPLETE_CLAUSES_HELDOUT = [
    "The frog jumped into the pond", "Anna painted a pretty picture",
    "The children laughed together", "Max climbed the tall hill",
    "The rain stopped at last",
]

CATEGORY = {
    "determiner_a_an": "lexical", "comparative_than": "lexical",
    "proper_noun_completion": "lexical", "pronoun_gender": "lexical",
    "numeric_sequence": "lexical", "common_idiom": "lexical",
    "end_of_sentence": "structural", "modal_continuation": "structural",
    "adjective_order": "structural", "past_tense_consistency": "structural",
    "subj_verb_agreement": "structural", "relative_clause_agreement": "structural",
    "close_quote": "structural", "reflexive_pronoun": "structural",
}


def _items(probe, rows, n_train=64, n_heldout=32, seed=0):
    rng = random.Random(f"{probe}-{seed}")
    train = [r for r in rows if r[3] == "train"]
    held = [r for r in rows if r[3] == "heldout"]
    rng.shuffle(train), rng.shuffle(held)
    out = []
    for split, pool, cap in (("train", train, n_train), ("heldout", held, n_heldout)):
        for i, (prefix, correct, distractor, _s, tid) in enumerate(pool[:cap]):
            out.append({
                "probe": probe, "category": CATEGORY[probe],
                "item_id": f"{probe}/{split}/{i:03d}", "template_id": tid,
                "split": split, "prefix": prefix,
                "correct": correct, "distractor": distractor, "chance": 0.5,
            })
    return out


def determiner_a_an(seed=0, rev=1):
    rows = []
    for name, vn, cn in itertools.product(GIRLS + BOYS, VOWEL_NOUNS, CONS_NOUNS):
        rows.append((f"{name} ate an", f" {vn}", f" {cn}", "train", "ate_an"))
        rows.append((f"{name} ate a", f" {cn}", f" {vn}", "train", "ate_a"))
        rows.append((f"One day, {name} saw an", f" {vn}", f" {cn}", "heldout", "saw_an"))
        rows.append((f"One day, {name} saw a", f" {cn}", f" {vn}", "heldout", "saw_a"))
    return _items("determiner_a_an", rows, seed=seed)


def comparative_than(seed=0, rev=1):
    adjs = [("bigger", "train"), ("smaller", "train"), ("faster", "train"),
            ("taller", "heldout"), ("stronger", "heldout")]
    rows = []
    for (adj, split), n1, n2 in itertools.product(adjs, SING_NOUNS, SING_NOUNS):
        if n1 == n2:
            continue
        tid = f"cmp_{adj}"
        rows.append((f"The {n1} was {adj}", " than", " then", split, tid))
    return _items("comparative_than", rows, seed=seed)


def proper_noun_completion(seed=0, rev=1):
    rows = []
    for g in GIRLS:
        rows.append(("Once upon a time, there was a little girl named",
                     f" {g}", " table", "train", "girl_named"))
    for b in BOYS:
        rows.append(("Once upon a time, there was a little boy named",
                     f" {b}", " chair", "train", "boy_named"))
    for g in GIRLS:
        rows.append(("There once lived a kind girl called",
                     f" {g}", " spoon", "heldout", "girl_called"))
    for b in BOYS:
        rows.append(("There once lived a brave boy called",
                     f" {b}", " window", "heldout", "boy_called"))
    if rev >= 2:
        for g in GIRLS:
            rows.append(("In a small village there lived a girl named",
                         f" {g}", " basket", "train", "village_named"))
            rows.append(("This is a story about a happy girl named",
                         f" {g}", " garden", "train", "story_named"))
            rows.append(("Everyone in town loved the little girl named",
                         f" {g}", " wagon", "train", "town_named"))
            rows.append(("Down the road lived a little girl called",
                         f" {g}", " bucket", "heldout", "road_called"))
        for b in BOYS:
            rows.append(("In a small village there lived a boy named",
                         f" {b}", " basket", "train", "village_named"))
            rows.append(("This is a story about a happy boy named",
                         f" {b}", " garden", "train", "story_named"))
            rows.append(("Everyone in town loved the little boy named",
                         f" {b}", " wagon", "train", "town_named"))
            rows.append(("Down the road lived a little boy called",
                         f" {b}", " bucket", "heldout", "road_called"))
    n_tr, n_h = (12, 12) if rev == 1 else (64, 32)
    return _items("proper_noun_completion", rows, n_train=n_tr, n_heldout=n_h, seed=seed)


def pronoun_gender(seed=0, rev=1):
    rows = []
    for g, n in itertools.product(GIRLS, CONS_NOUNS):
        rows.append((f"{g} lost", " her", " his", "train", "lost"))
        rows.append((f"{g} smiled because", " she", " he", "heldout", "because"))
    for b, n in itertools.product(BOYS, CONS_NOUNS):
        rows.append((f"{b} lost", " his", " her", "train", "lost"))
        rows.append((f"{b} smiled because", " he", " she", "heldout", "because"))
    return _items("pronoun_gender", rows, seed=seed)


def numeric_sequence(seed=0, rev=1):
    seqs = [("one, two, three,", " four", " six", "train"),
            ("two, three, four,", " five", " seven", "train"),
            ("three, four, five,", " six", " nine", "train"),
            ("four, five, six,", " seven", " ten", "heldout"),
            ("five, six, seven,", " eight", " three", "heldout")]
    if rev >= 2:
        seqs += [("six, seven, eight,", " nine", " four", "train"),
                 ("seven, eight, nine,", " ten", " two", "heldout")]
    rows = []
    for name in GIRLS + BOYS:
        for pre, c, d, split in seqs:
            rows.append((f"{name} counted: {pre}", c, d, split, f"count_{c.strip()}"))
            if rev >= 2:
                rows.append((f"{name} counted the stars: {pre}", c, d, split,
                             f"stars_{c.strip()}"))
    return _items("numeric_sequence", rows, seed=seed)


def common_idiom(seed=0, rev=1):
    idioms = [("and they lived happily ever", " after", " before", "train", "ever_after"),
              ("It was raining cats and", " dogs", " cups", "train", "cats_dogs"),
              ("They were safe and", " sound", " round", "train", "safe_sound"),
              ("Once upon a", " time", " tree", "heldout", "once_upon"),
              ("The end of the", " story", " spoon", "heldout", "end_story")]
    if rev >= 2:
        idioms += [("They were best friends forever and", " ever", " never",
                    "train", "forever_ever"),
                   ("They all shouted: hip hip", " hooray", " hello",
                    "heldout", "hip_hooray")]
    rows = []
    for name in GIRLS + BOYS:
        for pre, c, d, split, tid in idioms:
            if split == "train":
                rows.append((f"{name} read the words: {pre}", c, d, split, tid))
            elif rev == 1:
                rows.append((pre, c, d, split, tid))
            else:
                # rev 1 held-out items were bare idiom prefixes duplicated per
                # name; vary the lead-in so bootstrap items are not identical
                rows.append((f"{name} read aloud: {pre}", c, d, split, tid))
    return _items("common_idiom", rows, seed=seed)


def end_of_sentence(seed=0, rev=1):
    rows = []
    for cl in COMPLETE_CLAUSES_TRAIN:
        rows.append((cl, ".", ",", "train", "clause_period"))
    for cl in COMPLETE_CLAUSES_HELDOUT:
        rows.append((cl, ".", ",", "heldout", "clause_period_h"))
    # expand with subject/place variation
    for name, place in itertools.product(GIRLS + BOYS, PLACES):
        rows.append((f"{name} walked to the {place}", ".", ",", "train", "walked"))
        rows.append((f"After lunch, {name} napped under the tree", ".", ",",
                     "heldout", "napped"))
    return _items("end_of_sentence", rows, seed=seed)


def modal_continuation(seed=0, rev=1):
    verbs = ["play", "go", "run", "sing", "help", "jump"]
    rows = []
    for name, v in itertools.product(GIRLS + BOYS, verbs):
        rows.append((f"Tomorrow, {name}", f" will {v}", f" is {v}", "train", "tomorrow"))
        rows.append((f"{name} said, \"Soon I", " will", " is", "heldout", "soon_I"))
    return _items("modal_continuation", rows, seed=seed)


def adjective_order(seed=0, rev=1):
    rows = []
    for size, color, noun in itertools.product(ADJ_SIZE, ADJ_COLOR, CONS_NOUNS):
        rows.append(("She had a", f" {size} {color} {noun}",
                     f" {color} {size} {noun}", "train", "she_had"))
        rows.append(("In the box there was a", f" {size} {color} {noun}",
                     f" {color} {size} {noun}", "heldout", "in_box"))
    return _items("adjective_order", rows, seed=seed)


def past_tense_consistency(seed=0, rev=1):
    rows = []
    for name, (past, pres) in itertools.product(GIRLS + BOYS, PAST_PAIRS):
        rows.append((f"Yesterday, {name}", f" {past}", f" {pres}", "train", "yesterday"))
        rows.append((f"Last week, {name}", f" {past}", f" {pres}", "heldout", "last_week"))
    return _items("past_tense_consistency", rows, seed=seed)


def subj_verb_agreement(seed=0, rev=1):
    rows = []
    for n in SING_NOUNS:
        rows.append((f"The {n}s", " were", " was", "train", "plural_were"))
        rows.append((f"The {n}", " was", " were", "train", "sing_was"))
        rows.append((f"All of the {n}s", " were", " was", "heldout", "all_were"))
        rows.append((f"Every {n}", " was", " were", "heldout", "every_was"))
    if rev >= 2:
        extra = ["duck", "bunny", "pig", "cow", "bear", "fox"]
        for n in extra:
            rows.append((f"The {n}s", " were", " was", "train", "plural_were"))
            rows.append((f"The {n}", " was", " were", "train", "sing_was"))
            rows.append((f"All of the {n}s", " were", " was", "heldout", "all_were"))
            rows.append((f"Every {n}", " was", " were", "heldout", "every_was"))
        for n in SING_NOUNS + extra:
            rows.append((f"The {n}s in the garden", " were", " was",
                         "train", "garden_were"))
            rows.append((f"The {n} in the garden", " was", " were",
                         "train", "garden_was"))
            rows.append((f"Both of the {n}s", " were", " was", "heldout", "both_were"))
            rows.append((f"Each {n}", " was", " were", "heldout", "each_was"))
    n_tr, n_h = (12, 12) if rev == 1 else (64, 32)
    return _items("subj_verb_agreement", rows, n_train=n_tr, n_heldout=n_h, seed=seed)


def relative_clause_agreement(seed=0, rev=1):
    rows = []
    for n1, n2 in itertools.product(SING_NOUNS, SING_NOUNS):
        if n1 == n2:
            continue
        rows.append((f"The {n1} that the {n2}s chased", " was", " were",
                     "train", "rc_sing_head"))
        rows.append((f"The {n1}s that the {n2} chased", " were", " was",
                     "train", "rc_plur_head"))
        rows.append((f"The {n1} near the {n2}s", " was", " were",
                     "heldout", "pp_sing_head"))
        if rev >= 3:
            # plural-head PP control: disambiguates a blanket "was" bias
            # (pp_sing 1 / pp_plur 0) from head agreement (1 / 1) and
            # nearest-noun attraction (0 / 0); without it the pp_sing_head
            # peak is uninterpretable
            rows.append((f"The {n1}s near the {n2}", " were", " was",
                         "heldout", "pp_plur_head"))
    n_h = 64 if rev >= 3 else 32
    return _items("relative_clause_agreement", rows, n_heldout=n_h, seed=seed)


def close_quote(seed=0, rev=1):
    quotes = ["I am happy", "I want to play", "Let's go home",
              "This is fun", "I like cake"]
    rows = []
    for name, q in itertools.product(GIRLS + BOYS, quotes):
        rows.append((f"{name} said, \"{q}", ".\"", "\".", "train", "said_quote"))
        if rev == 1:
            rows.append((f"\"{q}", ",\" said {0}.".format(name),
                         "\", said {0}.".format(name), "heldout", "quote_said"))
    if rev >= 2:
        # rev 1 held-out pairs buried the contrast under a long shared
        # " said {name}." suffix; use minimal two-char continuations instead
        quotes2 = ["We did it", "Come back soon", "That was yummy",
                   "Time for bed", "Look at that"]
        for name, q in itertools.product(GIRLS + BOYS, quotes2):
            rows.append((f"{name} shouted, \"{q}", "!\"", "\"!", "train",
                         "shout_quote"))
        for q in quotes + quotes2:
            rows.append((f"\"{q}", ",\"", "\",", "heldout", "quote_comma"))
            rows.append((f"\"{q}", "!\"", "\"!", "heldout", "quote_exclaim"))
    return _items("close_quote", rows, seed=seed)


def reflexive_pronoun(seed=0, rev=1):
    verbs = ["hurt", "enjoyed", "dressed", "splashed"]
    rows = []
    for b, v in itertools.product(BOYS, verbs):
        rows.append((f"{b} {v}", " himself", " herself", "train", "boy_refl"))
    for g, v in itertools.product(GIRLS, verbs):
        rows.append((f"{g} {v}", " herself", " himself", "train", "girl_refl"))
    for v in verbs:
        rows.append((f"The children {v}", " themselves", " himself",
                     "heldout", "plural_refl"))
    if rev >= 2:
        verbs2 = ["washed", "surprised", "taught", "reminded"]
        for b, v in itertools.product(BOYS, verbs2):
            rows.append((f"{b} {v}", " himself", " herself", "train", "boy_refl2"))
        for g, v in itertools.product(GIRLS, verbs2):
            rows.append((f"{g} {v}", " herself", " himself", "train", "girl_refl2"))
        for b in BOYS:  # rev 1 had only 4 held-out items; new disjoint frames
            rows.append((f"After the race, {b} was proud of",
                         " himself", " herself", "heldout", "proud_of"))
            rows.append((f"{b} looked in the mirror and saw",
                         " himself", " herself", "heldout", "mirror"))
        for g in GIRLS:
            rows.append((f"After the race, {g} was proud of",
                         " herself", " himself", "heldout", "proud_of"))
            rows.append((f"{g} looked in the mirror and saw",
                         " herself", " himself", "heldout", "mirror"))
        for v in verbs + verbs2:
            rows.append((f"The kids {v}", " themselves", " herself",
                         "heldout", "kids_refl"))
    return _items("reflexive_pronoun", rows, seed=seed)


ALL_PROBES = [
    determiner_a_an, comparative_than, proper_noun_completion, pronoun_gender,
    numeric_sequence, common_idiom, end_of_sentence, modal_continuation,
    adjective_order, past_tense_consistency, subj_verb_agreement,
    relative_clause_agreement, close_quote, reflexive_pronoun,
]


def build_battery(seed: int = 0, rev: int = 1) -> list[dict]:
    items = []
    for fn in ALL_PROBES:
        items.extend(fn(seed=seed, rev=rev))
    return items


def write_battery(out_path: str | Path, seed: int = 0, rev: int = 1) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = build_battery(seed, rev=rev)
    with out_path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return len(items)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/probes/v1/battery.jsonl"
    rev = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n = write_battery(path, rev=rev)
    print(f"wrote {n} items (rev={rev})")
