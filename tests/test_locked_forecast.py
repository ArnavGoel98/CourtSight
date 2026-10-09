"""The 31.12.2026 forecast is locked: these hashes are the files as first committed (commit 84732fc, originally 391dca6)."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "forecast_2026"
LOCKED = {
    "forecast_states_2026.csv": "adf62cd20085e545730b10e62a7ffa66dae5e3f90207d197c5207f2aa5c96d2d",
    "forecast_meta.json": "1715bda1eed20622d7c43f89ea813ccf8b4d294f69a4380c40d39578174f4c13",
}


def test_locked_forecast_unchanged() -> None:
    for name, digest in LOCKED.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"{name} changed after locking"


if __name__ == "__main__":
    test_locked_forecast_unchanged()
    print("locked forecast unchanged")
