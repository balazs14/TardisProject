#!/usr/bin/env bash
set -euo pipefail

for day in $(python3 - << 'EOF'
from datetime import date, timedelta

d = date(2025, 1, 1)
end = date(2026, 1, 1)

while d < end:
    print(d.isoformat())
    d += timedelta(days=1)
EOF
            ); do
    if ! python okex_polars.py "$day"; then
        echo $day failed, continuing >& 2
    fi
    
done
