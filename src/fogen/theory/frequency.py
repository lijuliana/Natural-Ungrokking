"""Support-frequency estimation: how often does the licensing construction
for each capability occur in the training corpus?

This is the theory-relevant axis of the phase diagram (proposal §1) and the
input to the critical-frequency f* model (§3). v0 uses regex/string counts
per million tokens on raw text; refinements (parsed counts, competing-pattern
pressure) come after the pilot.
"""

import re
from collections import Counter

VOWEL_RE = r"[aeiouAEIOU]"

# capability -> (supporting_pattern, competing_pattern) regexes on raw text
PATTERNS = {
    "end_of_sentence": (r"[a-z]\.\s", r"[a-z],\s"),
    "comparative_than": (r"\b\w+er than\b", r"\b\w+er then\b"),
    "determiner_a_an": (rf"\ban {VOWEL_RE}", rf"\ba {VOWEL_RE}"),
    "modal_continuation": (r"\bwill [a-z]+\b", r"\bis [a-z]+ing\b"),
    "adjective_order_size_color": (
        r"\b(big|small|little|huge|tiny) (red|blue|green|yellow|brown)\b",
        r"\b(red|blue|green|yellow|brown) (big|small|little|huge|tiny)\b"),
    "past_tense": (r"\b(went|ran|saw|ate|found|made)\b",
                   r"\b(goes|runs|sees|eats|finds|makes)\b"),
    "subj_verb_agreement": (r"\b\w+s (were|are)\b", r"\b\w+s (was|is)\b"),
    "reflexive": (r"\b(himself|herself|themselves)\b", r"\b(hisself|theirselves)\b"),
    "close_quote": (r'[.!?]"', r'",|",'),
    "numeric_sequence": (r"\b(one, two|two, three|three, four|four, five)\b", None),
    "idiom_ever_after": (r"\bhappily ever after\b", r"\bever before\b"),
    "pronoun_gender": (r"\b(she|her|he|his|him)\b", None),
    "proper_noun_intro": (r"\bnamed [A-Z][a-z]+\b", r"\bcalled [A-Z][a-z]+\b"),
    "relative_clause": (r"\bthat the \w+s? [a-z]+ed\b", None),
}


# PATTERNS key -> probe name in data/probes/v1/battery.jsonl
PATTERN_TO_PROBE = {
    "end_of_sentence": "end_of_sentence",
    "comparative_than": "comparative_than",
    "determiner_a_an": "determiner_a_an",
    "modal_continuation": "modal_continuation",
    "adjective_order_size_color": "adjective_order",
    "past_tense": "past_tense_consistency",
    "subj_verb_agreement": "subj_verb_agreement",
    "reflexive": "reflexive_pronoun",
    "close_quote": "close_quote",
    "numeric_sequence": "numeric_sequence",
    "idiom_ever_after": "common_idiom",
    "pronoun_gender": "pronoun_gender",
    "proper_noun_intro": "proper_noun_completion",
    "relative_clause": "relative_clause_agreement",
}


def count_patterns(doc_iter, max_docs: int | None = None) -> dict:
    counts, total_words = Counter(), 0
    compiled = {k: (re.compile(s), re.compile(c) if c else None)
                for k, (s, c) in PATTERNS.items()}
    for i, doc in enumerate(doc_iter):
        if max_docs and i >= max_docs:
            break
        total_words += doc.count(" ") + 1
        for k, (sup, comp) in compiled.items():
            counts[f"{k}/support"] += len(sup.findall(doc))
            if comp:
                counts[f"{k}/compete"] += len(comp.findall(doc))
    out = {"total_words": total_words}
    for k in PATTERNS:
        s = counts[f"{k}/support"]
        c = counts.get(f"{k}/compete", 0)
        out[k] = {
            "support_per_million": 1e6 * s / max(total_words, 1),
            "compete_per_million": 1e6 * c / max(total_words, 1),
            "support_ratio": s / max(s + c, 1),
        }
    return out
