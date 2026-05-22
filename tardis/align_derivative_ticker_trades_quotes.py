import argparse
import logging
from pathlib import Path

import polars as pl

from tardis import _CliHelpFormatter
from tardis.markup_files import access_files


logger = logging.getLogger(__name__)

JOIN_KEYS = ["timestamp", "symbol"]
TRADE_VALUE_COLS = ["trade_amount", "trade_signed_amount", "trade_price_amount"]
QUOTE_VALUE_COLS = ["bid_price", "ask_price", "bid_amount", "ask_amount", "stale", "local_timestamp"]


def _assert_has_columns(df: pl.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"{name} missing columns: {missing}"


def _assert_unique_keys(df: pl.DataFrame, keys: list[str], name: str) -> None:
    duplicates = df.group_by(keys).len().filter(pl.col("len") > 1)
    assert duplicates.is_empty(), f"{name} has duplicate rows for keys {keys}"


def align_derivative_ticker_trades_quotes(
    df_derivative_ticker: pl.DataFrame,
    df_trades: pl.DataFrame,
    df_quotes: pl.DataFrame,
) -> pl.DataFrame:
    _assert_has_columns(df_derivative_ticker, ["timestamp", "symbol", "mark_price"], "df_derivative_ticker")
    _assert_has_columns(df_trades, JOIN_KEYS + TRADE_VALUE_COLS, "df_trades")
    _assert_has_columns(df_quotes, JOIN_KEYS + ["bid_price", "ask_price", "bid_amount", "ask_amount", "stale"], "df_quotes")

    _assert_unique_keys(df_derivative_ticker, JOIN_KEYS, "df_derivative_ticker")
    _assert_unique_keys(df_trades, JOIN_KEYS, "df_trades")
    _assert_unique_keys(df_quotes, JOIN_KEYS, "df_quotes")

    trade_frame = df_trades.select(JOIN_KEYS + TRADE_VALUE_COLS)
    quote_cols = [col for col in QUOTE_VALUE_COLS if col in df_quotes.columns and col != "local_timestamp"]
    quote_frame = df_quotes.select(JOIN_KEYS + quote_cols).rename({col: f"quote_{col}" for col in quote_cols})

    out = (
        df_derivative_ticker
        .join(trade_frame, on=JOIN_KEYS, how="left")
        .join(quote_frame, on=JOIN_KEYS, how="left")
    )

    return out.drop(["local_timestamp", "funding_timestamp"], strict=False)


def create_deribit_aligned_derivative_ticker_trades_quotes(
    date: str,
    symbol: str,
    sample_freq: str = "5min",
    raw_data_dir: str = "datasets/deribit_raw/",
) -> pl.DataFrame:
    df_derivative_ticker, _ = access_files("deribit", "derivative_ticker", date, symbol, sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_trades, _ = access_files("deribit", "trades", date, symbol, sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_quotes, _ = access_files("deribit", "quotes", date, symbol, sample_freq=sample_freq, raw_data_dir=raw_data_dir)

    assert not df_derivative_ticker.is_empty(), f"Missing derivative_ticker data for symbol={symbol} date={date}"
    assert not df_trades.is_empty(), f"Missing trades data for symbol={symbol} date={date}"
    assert not df_quotes.is_empty(), f"Missing quotes data for symbol={symbol} date={date}"

    df_result = align_derivative_ticker_trades_quotes(df_derivative_ticker, df_trades, df_quotes)
    return df_result.with_columns(pl.lit("deribit").alias("exchange"))


def create_okex_aligned_derivative_ticker_trades_quotes(
    date: str,
    symbol: str,
    sample_freq: str = "5min",
    raw_data_dir: str = "datasets/okex_raw/",
) -> pl.DataFrame:
    df_derivative_ticker, _ = access_files("okex-futures", "derivative_ticker", date, symbol, sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_trades, _ = access_files("okex-futures", "trades", date, symbol, sample_freq=sample_freq, raw_data_dir=raw_data_dir)
    df_quotes, _ = access_files("okex-futures", "quotes", date, symbol, sample_freq=sample_freq, raw_data_dir=raw_data_dir)

    assert not df_derivative_ticker.is_empty(), f"Missing derivative_ticker data for symbol={symbol} date={date}"
    assert not df_trades.is_empty(), f"Missing trades data for symbol={symbol} date={date}"
    assert not df_quotes.is_empty(), f"Missing quotes data for symbol={symbol} date={date}"

    df_result = align_derivative_ticker_trades_quotes(df_derivative_ticker, df_trades, df_quotes)
    return df_result.with_columns(pl.lit("okex").alias("exchange"))


def create_aligned_derivative_ticker_trades_quotes(
    exchange: str,
    date: str,
    symbol: str,
    sample_freq: str = "5min",
    raw_data_dir: str | None = None,
) -> pl.DataFrame:
    if exchange == "okex":
        resolved_raw_data_dir = raw_data_dir or "datasets/okex_raw/"
        return create_okex_aligned_derivative_ticker_trades_quotes(
            date,
            symbol,
            sample_freq=sample_freq,
            raw_data_dir=resolved_raw_data_dir,
        )
    if exchange == "deribit":
        resolved_raw_data_dir = raw_data_dir or "datasets/deribit_raw/"
        return create_deribit_aligned_derivative_ticker_trades_quotes(
            date,
            symbol,
            sample_freq=sample_freq,
            raw_data_dir=resolved_raw_data_dir,
        )

    print(f"Unsupported exchange: {exchange}. Expected okex or deribit.")
    assert False, f"unsupported exchange: {exchange}"


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build aligned derivative-ticker/trades/quotes dataframe for an exchange/day.",
        formatter_class=_CliHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--exchange", required=True, choices=["okex", "deribit"], help="Exchange to process")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--symbol", required=True, help="Instrument symbol to align (e.g. BTC-24APR26)")
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
        "Parsed align-derivative-ticker args: exchange=%s date=%s symbol=%s sample_freq=%s recreate_existing=%s raw_data_dir=%s output=%s",
        args.exchange,
        args.date,
        args.symbol,
        args.sample_freq,
        args.recreate_existing,
        args.raw_data_dir,
        args.output,
    )

    output_path = args.output
    if output_path is None:
        symbol_slug = args.symbol.replace("/", "_")
        output_path = (
            f"datasets/{args.exchange}/"
            f"{args.exchange}_aligned_derivative_ticker_trades_quotes_{args.date}_{symbol_slug}_{args.sample_freq}.parquet"
        )

    out = Path(output_path)
    if out.exists() and not args.recreate_existing:
        logger.info("Output already exists and recreate_existing is false, skipping: %s", out)
        print(f"skipped_existing={out}")
        return

    raw_data_dir = args.raw_data_dir.format(exchange=args.exchange)

    df = create_aligned_derivative_ticker_trades_quotes(
        args.exchange,
        args.date,
        args.symbol,
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