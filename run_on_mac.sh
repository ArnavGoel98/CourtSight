#!/bin/bash
# Rebuild ddl_compact/ with the extra detail (case types, courtroom-months, district names) from the DDL download,
# then commit and push it. Only aggregated counts leave the computer: no case-level rows, no names of people.
# Usage:  bash run_on_mac.sh [DDL_FOLDER]
set -euo pipefail
DDL="${1:-/Users/arnav/Documents/Uploads/justice_data}"
cd "$(dirname "$0")"
git pull --ff-only origin main
python3 -m pip install --user -q pandas pyarrow numpy
if [ ! -d "$DDL" ]; then
  echo "DDL folder not found: $DDL. Find it with: find ~ -name 'cases*.tar.gz' 2>/dev/null"; exit 1
fi
python3 compress_ddl.py --src "$DDL" --out ddl_compact --detail
du -sh ddl_compact ddl_compact/detail
git add ddl_compact
git commit -m "Add case-type, courtroom and district-name detail to the compact DDL data"
git push origin main
echo "Done. Pushed ddl_compact/detail and ddl_compact/district_key.csv."
