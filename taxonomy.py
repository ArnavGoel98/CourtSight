"""Shared DDL schema maps and regex taxonomies (pandas/numpy only, so it runs on any laptop)."""
from __future__ import annotations

from typing import Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

# DDL judicial-data column names -> canonical names. Verify against the release README.
DDL_CASE_COLS: Final[Mapping[str, str]] = {
    "ddl_case_id": "case_id",
    "state_code": "state_code",
    "dist_code": "dist_code",
    "court_no": "court_no",
    "type_name": "type_name",
    "purpose_name": "purpose_name",
    "date_of_filing": "filed",
    "date_of_decision": "decided",
    "date_first_list": "first_list",
    "date_last_list": "last_list",
    "date_next_list": "next_list",
}
DDL_JUDGE_COLS: Final[Mapping[str, str]] = {
    "ddl_judge_id": "judge_id",
    "state_code": "state_code",
    "dist_code": "dist_code",
    "court_no": "court_no",
    "judge_position": "judge_position",
    "start_date": "start",
    "end_date": "end",
}

STAGES: Final[tuple[str, ...]] = ("service", "charge", "interlocutory", "evidence", "arguments", "judgment")

# First match wins. Audit against the label-audit files written by compress_ddl.py.
CASE_CATEGORIES: Final[tuple[tuple[str, str], ...]] = (
    ("crim_bail", r"bail|^b\.?\s?a\.?$|\bb\.?a\.? ?(?:no|case)"),
    ("civ_mact", r"m\.?\s?a\.?\s?c\.?\s?[tpcm]|motor accident|accident claim|\bm\.?\s?v\.?\s?c\b|mcop|\bop\.?\s?\(?mv|mvop|\bmacc|macma|\bmact|claim cases?"),
    ("crim_ni138", r"n\.?\s?i\.?\s?act|\b138\b|cheque|\bnact\b|negotiable"),
    ("crim_sessions", r"session|\bs\.?\s?c\.?\b|ndps|pocso|spl\.?\s?case|special (?:case|trial)|\bs\.?\s?t\.?\s?\(?sc"),
    ("civ_family", r"matrimonial|marriage|divorce|h\.?\s?m\.?\s?a|hmop|guardian|family|maintenance|\bmnt|\b125\b|^m\.?\s?c\.?$"),
    ("civ_execution", r"execu?i?tion|^exe?\b|^ex\.|\bex\.?\s?p|\be\.?\s?p\.?\b|\bexe\s?r"),
    ("civ_suit", r"suit|\bo\.?\s?s\.?\b|original|\bc\.?\s?s\.?\b|\br\.?\s?c\.?\s?s\b|\bs\.?\s?c\.?\s?s\b"),
    ("crim_magisterial", r"crim|\bcri\b|\bcri\.|\bcr\b|\bcr\.|crl|\bcr[moa]|\bcrr|\bcrma|\bcrm\b|\br\.?\s?c\.?\s?t\b|\bstc\b|\bs\.?\s?t\.?\b"
     r"|\bsumm?\b|summon|\bg\.?\s?r\b|\bn?gr\s?case|p\.?\s?c\.?\s?r|final report|^f\.?r\.?$|\bct\.?\s?cases?|warrant|excise"
     r"|traffic|motor veh|\bm\.?\s?v\.?\s?act|\bc\.?\s?c\.?\b|complaint|petty|police report|\bucr\b|\bipc\b|r\.?\s?c\.?\s?c\b"
     r"|summary|challan|police cases?|electricity act|mot[ao]r vehicle|\bs\.?\s?s\.?\b|ss cases|\bprc\b"),
    ("civ_misc", r"misc|arbitration|arbtn|succession|land acq|\bl\.?\s?a\.?\s?r?\b|civil|appeal|petition|^o\.?p\.?\b"
     r"|^c\.?a\.?$|^a\.?s\.?$|^r\.?\s?a\.?$|\br\.?\s?c\.?\s?a\b|^c\.?m\.?a?\.?$|\btrf\b|transfer|caveat|insolvency|rent"),
)
STAGE_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("judgment", r"judg|verdict|pronounce|orders?\b|final form|^f\.\s?o\.?$|disposal|^for disposal"),
    ("arguments", r"argu|final hearing"),
    ("evidence", r"evidence|witness|\bp\.?\s?e\.?\b|\bd\.?\s?e\.?\b|examination|cross|\b313\b|statement of accused|^(?:for )?trial$"),
    ("charge", r"charge|plea|framing|consider?ation|^issues?$|police paper|\b207\b"),
    ("service", r"summon|notice|service|app[ea]a?re?a?nce|appearence|appereance|warrant|process|\bn\.?\s?b\.?\s?w|\bb\.?\s?w\.?_|abscond"
     r"|attendance|copy|^steps|for steps|steps_|compliance|awaiting|await|report|enquiry|production of accused|accused presence"),
    ("interlocutory", r"\bi\.?\s?a\.?\b|interim|stay|misc|application|reply|written statement|^w\.\s?s\.?$|objection|hearing|bail|counter"
     r"|show cause|verification|lok.?adalat|lok.?nyayalaya|admission|addmission|filing|say on"),
)


def classify(labels: pd.Series, rules: Sequence[tuple[str, str]], default: str) -> npt.NDArray[np.str_]:
    """Regex taxonomy evaluated on unique strings only (|uniques| << |rows|)."""
    codes, uniques = pd.factorize(labels.astype("string").fillna(""), sort=False)
    u = pd.Series(np.asarray(uniques, dtype=object), dtype="string").str.lower()
    conds = [u.str.contains(p, regex=True).fillna(False).to_numpy(dtype=bool) for _, p in rules]
    out = np.select(conds, [n for n, _ in rules], default=default)
    return out[codes]
