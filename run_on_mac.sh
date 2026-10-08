#!/bin/bash
# One-shot local runner: district key -> push, then 2019 grading if Dataful files exist.
# Usage:  bash court_pendency/run_on_mac.sh [DDL_FOLDER] [DATAFUL_FOLDER]
set -euo pipefail
DDL="${1:-/Users/arnav/Documents/Uploads/justice_data}"
DATAFUL="${2:-$HOME/Downloads/dataful}"
cd "$(dirname "$0")/.."
git pull --ff-only origin main

KEY_OUT=court_pendency/ddl_compact/district_key.csv
if [ ! -s "$KEY_OUT" ]; then
  KEYS=$(find "$DDL" -name 'keys.tar.gz' 2>/dev/null | head -1)
  if [ -z "$KEYS" ]; then
    echo "No keys.tar.gz under $DDL. Find it with: find ~ -name keys.tar.gz 2>/dev/null"
    echo "then re-run: bash court_pendency/run_on_mac.sh /path/to/ddl_folder"; exit 1
  fi
  MEMBER=$(tar -tzf "$KEYS" | grep -i 'district' | grep -i '\.csv$' | head -1 || true)
  if [ -z "$MEMBER" ]; then echo "No district CSV inside $KEYS. Contents:"; tar -tzf "$KEYS"; exit 1; fi
  echo "Extracting $MEMBER from $KEYS"
  tar -xzf "$KEYS" -O "$MEMBER" > "$KEY_OUT"
  head -3 "$KEY_OUT"
  git add "$KEY_OUT"
  git commit -m "Add DDL district name key"
  git push origin main
else
  echo "district_key.csv already present"
fi

if ls "$DATAFUL"/* >/dev/null 2>&1; then
  python3 court_pendency/prepare_dataful.py --src "$DATAFUL" --out court_pendency/dataful
  python3 court_pendency/grade_2019.py --dataful court_pendency/dataful --district-key "$KEY_OUT"
  echo "Review unmatched rows in court_pendency/results/ddl_2010_2018/grading_2019/crosswalk_review.csv"
  open court_pendency/results/ddl_2010_2018/grading_2019/crosswalk_review.csv || true
else
  echo "No Dataful files in $DATAFUL yet; grading skipped. Download 21265 + 21282 there and re-run."
fi
