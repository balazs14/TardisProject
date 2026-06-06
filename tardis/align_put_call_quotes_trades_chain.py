import argparse
import logging
from pathlib import Path

import polars as pl

from tardis import _CliHelpFormatter
from tardis import align_put_call_data as apc
from tardis.markup_files import access_files, mark_up
from tardis.tardis_universe import universe_symbols


logger = logging.getLogger(__name__)

JOIN_SYMBOL_KEYS = ["timestamp", "symbol"]
TRADE_VALUE_COLS = ["trade_amount", "trade_signed_amount", "trade_price_amount"]


def _assert_has_columns(df: pl.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"{name} missing columns: {missing}"


def _assert_unique_keys(df: pl.DataFrame, keys: list[str], name: str) -> None:
    duplicates = df.group_by(keys).len().filter(pl.col("len") > 1)
    assert duplicates.is_empty(), f"{name} has duplicate rows for keys {keys}"


def align_put_call_quotes_trades_chain(
    df_opt_quotes: pl.DataFrame,
    df_opt_trades: pl.DataFrame,
    df_option_chain: pl.DataFrame,
    df_fut: pl.DataFrame,
    df_spot: pl.DataFrame,
) -> pl.DataFrame:
    _assert_has_columns(df_opt_trades, JOIN_SYMBOL_KEYS + TRADE_VALUE_COLS, "df_opt_trades")
    _assert_has_columns(df_option_chain, JOIN_SYMBOL_KEYS + ["open_interest"], "df_option_chain")

    _assert_unique_keys(df_opt_trades, JOIN_SYMBOL_KEYS, "df_opt_trades")
    _assert_unique_keys(df_option_chain, JOIN_SYMBOL_KEYS, "df_option_chain")

    # Start from existing put/call + future + spot quote alignment logic.
    out = apc.align_data(df_opt_quotes, df_fut, df_spot)

    trade_frame = df_opt_trades.select(JOIN_SYMBOL_KEYS + TRADE_VALUE_COLS)
    call_trade = trade_frame.rename(
        {
            "symbol": "call_symbol",
            "trade_amount": "call_trade_amount",
            "trade_signed_amount": "call_trade_signed_amount",
            "trade_price_amount": "call_trade_price_amount",
        }
    )
    put_trade = trade_frame.rename(
        {
            "symbol": "put_symbol",
            "trade_amount": "put_trade_amount",
            "trade_signed_amount": "put_trade_signed_amount",
            "trade_price_amount": "put_trade_price_amount",
        }
    )

    oi_frame = df_option_chain.select(JOIN_SYMBOL_KEYS + ["open_interest"])
    call_oi = oi_frame.rename({"symbol": "call_symbol", "open_interest": "call_open_interest"})
    put_oi = oi_frame.rename({"symbol": "put_symbol", "open_interest": "put_open_interest"})

    out = (
        out.join(call_trade, on=["timestamp", "call_symbol"], how="left")
        .join(put_trade, on=["timestamp", "put_symbol"], how="left")
        .join(call_oi, on=["timestamp", "call_symbol"], how="left")
        .join(put_oi, on=["timestamp", "put_symbol"], how="left")
    )

    return out.drop(["local_timestamp", "funding_timestamp"], strict=False)


def create_okex_aligned_put_call_quotes_trades_chain(
    date: str,
    sample_freq: str = "5min",
    raw_data_dir: str = "datasets/{exchange}_raw/",
) -> pl.DataFrame:
    df_opt_quotes, _ = access_files("okex-options", "quotes", date, "OPTIONS", sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_opt_quotes = mark_up(df_opt_quotes, "okex-options", "quotes")

    df_opt_trades, _ = access_files("okex-options", "trades", date, "OPTIONS", sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_opt_trades = mark_up(df_opt_trades, "okex-options", "trades")

    df_option_chain, _ = access_files("okex-options", "options_chain", date, "OPTIONS", sample_freq=sample_freq, raw_data_dir=raw_data_dir)

    assert not df_opt_quotes.is_empty(), f"Missing options quotes data for symbol=OPTIONS date={date}"
    assert not df_opt_trades.is_empty(), f"Missing options trades data for symbol=OPTIONS date={date}"
    assert not df_option_chain.is_empty(), f"Missing options_chain data for symbol=OPTIONS date={date}"

    fut_symbols = (
        df_opt_quotes.select(pl.col("fut_sym").drop_nulls().unique().sort()).to_series().to_list()
    )

    available_fut_symbols = {
        s["id"]
        for s in universe_symbols(
            exchange="okex-futures",
            stream_type_regex="quotes",
            date=date,
        )
    }
    fut_symbols = [s for s in fut_symbols if s in available_fut_symbols]

    df_fut, _ = access_files("okex-futures", "quotes", date, fut_symbols, sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_fut = mark_up(df_fut, "okex-futures", "quotes")

    df_spot, _ = access_files("okex", "quotes", date, ["BTC-USDT", "ETH-USDT"], sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_spot = mark_up(df_spot, "okex", "quotes")

    df_result = align_put_call_quotes_trades_chain(df_opt_quotes, df_opt_trades, df_option_chain, df_fut, df_spot)
    return df_result.with_columns(pl.lit("okex").alias("exchange"))


def create_deribit_aligned_put_call_quotes_trades_chain(
    date: str,
    sample_freq: str = "5min",
    raw_data_dir: str = "datasets/{exchange}_raw/",
) -> pl.DataFrame:
    df_opt_quotes, _ = access_files("deribit", "quotes", date, "OPTIONS", sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_opt_quotes = mark_up(df_opt_quotes, "deribit", "quotes", symbol_type="option")

    df_opt_trades, _ = access_files("deribit", "trades", date, "OPTIONS", sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_opt_trades = mark_up(df_opt_trades, "deribit", "trades", symbol_type="option")

    df_option_chain, _ = access_files("deribit", "options_chain", date, "OPTIONS", sample_freq=sample_freq, raw_data_dir=raw_data_dir)

    assert not df_opt_quotes.is_empty(), f"Missing options quotes data for symbol=OPTIONS date={date}"
    assert not df_opt_trades.is_empty(), f"Missing options trades data for symbol=OPTIONS date={date}"
    assert not df_option_chain.is_empty(), f"Missing options_chain data for symbol=OPTIONS date={date}"

    df_fut, _ = access_files("deribit", "derivative_ticker", date, "FUTURES", sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_fut = mark_up(df_fut, "deribit", "derivative_ticker", symbol_type="future")

    df_spot, _ = access_files("deribit", "quotes", date, ["ETH_USDC", "BTC_USDC"], sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_spot = mark_up(df_spot, "deribit", "quotes", symbol_type="spot")

    df_result = align_put_call_quotes_trades_chain(df_opt_quotes, df_opt_trades, df_option_chain, df_fut, df_spot)
    return df_result.with_columns(pl.lit("deribit").alias("exchange"))


def create_aligned_put_call_quotes_trades_chain(
    exchange: str,
    date: str,
    sample_freq: str = "5min",
    raw_data_dir: str | None = None,
) -> pl.DataFrame:
    if exchange == "okex":
        resolved_raw_data_dir = raw_data_dir or "datasets/okex_raw/"
        return create_okex_aligned_put_call_quotes_trades_chain(
            date,
            sample_freq=sample_freq,
            raw_data_dir=resolved_raw_data_dir,
        )
    if exchange == "deribit":
        resolved_raw_data_dir = raw_data_dir or "datasets/deribit_raw/"
        return create_deribit_aligned_put_call_quotes_trades_chain(
            date,
            sample_freq=sample_freq,
            raw_data_dir=resolved_raw_data_dir,
        )

    print(f"Unsupported exchange: {exchange}. Expected okex or deribit.")
    assert False, f"unsupported exchange: {exchange}"


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build aligned put/call/future/spot quotes with option trades and open interest.",
        formatter_class=_CliHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--exchange", required=True, choices=["okex", "deribit"], help="Exchange to process")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--sample-freq", default="5min", help="Sampling frequency label for output naming")
    recreate_group = parser.add_mutually_exclusive_group()
    recreate_group.add_argument(
        "--recreate-existing",
        dest="recreate_existing",
        action="store_true",
        help="Recreate output parquet even when it already exists.",
    )
    recreate_group.add_argument(
        "--no-recreate-existing",
        dest="recreate_existing",
        action="store_false",
        help="Skip processing when the output parquet path already exists (default).",
    )
    parser.set_defaults(recreate_existing=False)
    parser.add_argument(
        "--raw-data-dir",
        default="datasets/{exchange}_raw/",
        help="Directory for download_resample parquet outputs",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path. Derived from exchange/date/sample-freq when omitted.",
    )
    return parser


def run_cli_args(args: argparse.Namespace) -> None:
    logger.debug(
        "Parsed align-put-call-quotes-trades-chain args: exchange=%s date=%s sample_freq=%s recreate_existing=%s raw_data_dir=%s output=%s",
        args.exchange,
        args.date,
        args.sample_freq,
        args.recreate_existing,
        args.raw_data_dir,
        args.output,
    )

    output_path = args.output
    if output_path is None:
        output_path = (
            f"datasets/{args.exchange}/"
            f"{args.exchange}_aligned_put_call_quotes_trades_chain_{args.date}_{args.sample_freq}.parquet"
        )

    out = Path(output_path)
    if out.exists() and not args.recreate_existing:
        logger.info("Output already exists and recreate_existing is false, skipping: %s", out)
        print(f"skipped_existing={out}")
        return

    raw_data_dir = args.raw_data_dir.format(exchange=args.exchange)

    df = create_aligned_put_call_quotes_trades_chain(
        args.exchange,
        args.date,
        sample_freq=args.sample_freq,
        raw_data_dir=raw_data_dir,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    logger.info("Aligned dataframe written: path=%s rows=%d cols=%d", out, df.height, df.width)
    print(f"wrote={out} rows={df.height} cols={df.width}")


def run_cli(argv: list[str] | None = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    run_cli_args(args)
