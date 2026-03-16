#!/bin/bash
set -euo pipefail

for day in $(python3 - << 'EOF'
from datetime import date, timedelta
import pandas as pd

start = date(2020, 1, 1)
d = date(2026, 3, 14)
d = (pd.Timestamp.now() - pd.Timedelta("1day")).date() 
while d > start:
    print(d.isoformat())
    d -= timedelta(days=1)
EOF
            ); do
    for ex in deribit okex; do
        cmd="python -m tardis.main --days $day --exchanges $ex --cleanup_csv true --cleanup_intermediate_parquet false --force_reload false --loglevel INFO --align_only true"
        echo $cmd
        if ! $cmd ; then
            echo $day failed, continuing >& 2
        fi
        #exit
    done
done
