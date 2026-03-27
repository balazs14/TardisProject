import logging
from pathlib import Path

from tardis.lib.parquet_zip_join import inspect_inputs, zip_join_parquets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Example: merge 5min parquets by key and keep only selected columns.
SOURCES = {
    "options": "datasets/deribit_chain/deribit_options_chain_2025-11-21_OPTIONS_5min.parquet",
    "quotes": "datasets/deribit/deribit_quotes_2025-11-21_OPTIONS_5min.parquet",
    "trades": "datasets/deribit/deribit_trades_2025-11-21_OPTIONS_5min.parquet",
}

# output_col -> (source_alias, source_column)
COLUMN_MAP = {
    "opt_mark_price": ("options", "mark_price"),
    "opt_bid": ("options", "bid_price"),
    "opt_ask": ("options", "ask_price"),
    "q_bid": ("quotes", "bid_price"),
    "q_ask": ("quotes", "ask_price"),
    "trade_amount": ("trades", "trade_amount"),
}

OUT = Path("datasets/deribit_chain/deribit_joined_2025-11-21_5min.parquet")


if __name__ == "__main__":
    inspect_inputs(SOURCES, symbol_limit=20, ts_limit=20)
    zip_join_parquets(
        sources=SOURCES,
        column_map=COLUMN_MAP,
        output_path=OUT,
        join_keys=("timestamp", "symbol", "exchange"),
        join_type="inner",
    )
