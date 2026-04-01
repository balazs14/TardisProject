import logging
from pathlib import Path
import numpy as np
from zmq import log
from tardis.zipjoin_files import inspect_inputs, zip_join_parquets
import sys
from tardis.utils import _configure_logging

logger = logging.getLogger(__name__)

# Example: merge 5min parquets by key and keep only selected columns.
SOURCES = {
    "options": "datasets/deribit/deribit_options_chain_2026-03-26_OPTIONS_5min.parquet",
    "quotes": "datasets/deribit/deribit_quotes_2026-03-26_OPTIONS_5min.parquet",
    "trades": "datasets/deribit/deribit_trades_2026-03-26_OPTIONS_5min.parquet",
}

# output_col -> (source_alias, source_column)
COLUMN_MAP = {
    "opt_mark_price": ("options", "mark_price"),
    "opt_bid": ("options", "bid_price"),
    "opt_ask": ("options", "ask_price"),
    "q_bid": ("quotes", "bid_price"),
    "q_ask": ("quotes", "ask_price"),
    "trade_amount": ('trades', "trade_amount"),
}



OUT = Path("./deribit_joined_2026-03-26_5min.parquet")

def testme():
    _configure_logging("DEBUG")
    inspect_inputs(SOURCES, symbol_limit=2, ts_limit=2)
    zip_join_parquets(
        sources=SOURCES,
        column_map=COLUMN_MAP,
        output_path=OUT,
        join_keys=("timestamp", "symbol"),
        join_type="inner",
    )

def main():
    _loglevel = "INFO"
    if "--loglevel" in sys.argv:
        _i = sys.argv.index("--loglevel")
        if _i + 1 >= len(sys.argv):
            raise SystemExit("Expected value after --loglevel")
        _loglevel = sys.argv[_i + 1]
        del sys.argv[_i:_i + 2]
    _configure_logging(_loglevel)

    inspect_inputs(SOURCES, symbol_limit=2, ts_limit=2)
    zip_join_parquets(
        sources=SOURCES,
        column_map=COLUMN_MAP,
        output_path=OUT,
        join_keys=("timestamp", "symbol"),
        join_type="inner",
    )


if __name__ == "__main__":
    main()

