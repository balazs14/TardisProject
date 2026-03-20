#!/bin/bash
set -euo pipefail

cd /my/TardisProject
source venv/bin/activate

for day in $(python3 - << 'EOF'
from datetime import date, timedelta
import pandas as pd

start = date(2024, 1, 1)
d = date(2026, 3, 16)
#d = (pd.Timestamp.now() - pd.Timedelta("1day")).date() 
while d > start:
    print(d.isoformat())
    d -= timedelta(days=1)
EOF
            ); do
         cmd="venv/bin/python -m tardis.download_and_convert  --data-type options_chain  --exchange deribit   --symbol OPTIONS "
         cmd=$cmd"  --start-date $day  --end-date $day   --loglevel INFO   --data-dir datasets/deribit_chain --resample-freq 5min "
        echo $cmd
        if ! $cmd ; then
            echo $day failed, continuing >& 2
        fi
        #exit
done
