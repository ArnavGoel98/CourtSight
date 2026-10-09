"""Compare regenerated result files with the committed ones, numerically (to ~1e-9 relative), not byte for byte.

Floating-point results can differ in the last digits between machines, so a byte diff fails without any real
change. Usage: python tests/compare_outputs.py <committed_dir> <regenerated_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-9, 1e-9


def same_json(a: object, b: object) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(a, b, rtol=RTOL, atol=ATOL))
    return a == b


def same_csv(a: Path, b: Path) -> bool:
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return False
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            if not np.allclose(x[c], y[c], rtol=RTOL, atol=ATOL, equal_nan=True):
                return False
        elif not x[c].astype(str).equals(y[c].astype(str)):
            return False
    return True


def main() -> None:
    old, new = Path(sys.argv[1]), Path(sys.argv[2])
    bad = []
    for f in sorted(old.rglob("*")):
        if f.suffix not in (".csv", ".json"):
            continue
        g = new / f.relative_to(old)
        ok = g.exists() and (same_csv(f, g) if f.suffix == ".csv" else same_json(json.loads(f.read_text()),
                                                                                   json.loads(g.read_text())))
        if not ok:
            bad.append(str(f.relative_to(old)))
    if bad:
        print(f"::error::results differ: {bad}")
        raise SystemExit(f"results differ from the committed ones: {bad}")
    print("regenerated results match the committed ones")


if __name__ == "__main__":
    main()
