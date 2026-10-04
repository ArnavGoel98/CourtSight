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
    ("crim_bail", r"bail"),
    ("crim_ni138", r"n\.?\s?i\.?\s?act|\b138\b|cheque"),
    ("crim_sessions", r"sessions|\bs\.?\s?c\.?\b|ndps|pocso|spl\.?\s?case|special case"),
    ("civ_mact", r"m\.?\s?a\.?\s?c\.?\s?[tp]|motor accident"),
    ("civ_family", r"matrimonial|divorce|h\.?\s?m\.?\s?a|guardian|family|maintenance|\b125\b"),
    ("civ_execution", r"execution|\bex\.?\s?p|\be\.?\s?p\.?\b"),
    ("civ_suit", r"suit|\bo\.?\s?s\.?\b|original|\bc\.?\s?s\.?\b"),
    ("crim_magisterial", r"crim|crl|\bc\.?\s?c\.?\b|complaint|summary|petty|\bs\.?\s?t\.?\b|police report"),
    ("civ_misc", r"misc|arbitration|succession|land acq|civil|appeal|petition"),
)
STAGE_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("judgment", r"judg|verdict|pronounce|orders?\b"),
    ("arguments", r"argu|final hearing"),
    ("evidence", r"evidence|witness|\bp\.?\s?e\.?\b|\bd\.?\s?e\.?\b|examination|cross|\b313\b|statement of accused"),
    ("charge", r"charge|plea|framing|consideration"),
    ("service", r"summon|notice|service|appearance|warrant|process|\bnbw\b|abscond|attendance|copy"),
    ("interlocutory", r"\bi\.?\s?a\.?\b|interim|stay|misc|application|reply|written statement|objection|hearing"),
)


def classify(labels: pd.Series, rules: Sequence[tuple[str, str]], default: str) -> npt.NDArray[np.str_]:
    """Regex taxonomy evaluated on unique strings only (|uniques| << |rows|)."""
    codes, uniques = pd.factorize(labels.astype("string").fillna(""), sort=False)
    u = pd.Series(np.asarray(uniques, dtype=object), dtype="string").str.lower()
    conds = [u.str.contains(p, regex=True).fillna(False).to_numpy(dtype=bool) for _, p in rules]
    out = np.select(conds, [n for n, _ in rules], default=default)
    return out[codes]
