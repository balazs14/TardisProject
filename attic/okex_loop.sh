#!/bin/bash
set -euo pipefail

for day in $(python3 - << 'EOF'
from datetime import date, timedelta

start = date(2022, 1, 1)
d = date(2026, 2, 23)

while d > start:
    print(d.isoformat())
    d -= timedelta(days=1)
EOF
            ); do
    if ! python okex.py "$day"; then
        echo $day failed, continuing >& 2
    fi

done
