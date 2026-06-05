import argparse
import logging
from pathlib import Path

from tardis import _CliHelpFormatter

import polars as pl

from tardis.markup_files import access_files, mark_up
from tardis.tardis_universe import universe_symbols


logger = logging.getLogger(__name__)


def create_okex_aligned_options(
        date: str,
        sample_freq: str = "5min",
        raw_data_dir: str = "datasets/{exchange}_raw/",
) -> pl.DataFrame:
        df_opt, _ = access_files("okex-options", "quotes", date, "OPTIONS", sample_freq=sample_freq, raw_data_dir=raw_data_dir)
        df_opt = mark_up(df_opt, "okex-options", "quotes")

        fut_symbols = (
                df_opt.select(pl.col("fut_sym").drop_nulls().unique().sort())
                .to_series()
                .to_list()
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

        df_result = align_data(df_opt, df_fut, df_spot)
        df_result = df_result.with_columns(pl.lit("okex").alias("exchange"))

        return df_result


def create_deribit_aligned_options(
        date: str,
        sample_freq: str = "5min",
        raw_data_dir: str = "datasets/{exchange}_raw/",
) -> pl.DataFrame:
        df_opt, _ = access_files("deribit", "quotes", date, "OPTIONS", sample_freq=sample_freq, raw_data_dir=raw_data_dir)
        df_opt = mark_up(df_opt, "deribit", "quotes", symbol_type="option")

        df_fut, _ = access_files("deribit", "derivative_ticker", date, "FUTURES", sample_freq=sample_freq, raw_data_dir=raw_data_dir)
        df_fut = mark_up(df_fut, "deribit", "derivative_ticker", symbol_type="future")

        df_spot, _ = access_files("deribit", "quotes", date, ["ETH_USDC", "BTC_USDC"], sample_freq=sample_freq, raw_data_dir=raw_data_dir)
        df_spot = mark_up(df_spot, "deribit", "quotes", symbol_type="spot")

        df_result = align_data(df_opt, df_fut, df_spot)
        df_result = df_result.with_columns(pl.lit("deribit").alias("exchange"))

        return df_result


def align_data(
        df_opt: pl.DataFrame,
        df_fut: pl.DataFrame,
        df_spot: pl.DataFrame,
) -> pl.DataFrame:
        """Join calls, puts, futures and spot quotes into one PCP-ready dataframe.

        Returns a frame keyed by (timestamp, call_symbol) with call_/put_/fut_/spot_
        prefixed bid/ask/amount/stale columns.
        """
        assert "stale" in df_opt.columns, "df_opt must have a stale column"
        assert "stale" in df_fut.columns, "df_fut must have a stale column"
        assert "stale" in df_spot.columns, "df_spot must have a stale column"

        call_put_keys = ["timestamp", "exp", "strike", "exchange", "inverse", "fut_sym", "ref_sym", "spot_sym"]
        opt_val_cols = ["bid_price", "ask_price", "bid_amount", "ask_amount", "stale"]
        leg_val_cols = ["bid_price", "ask_price", "bid_amount", "ask_amount", "stale"]

        # ── calls ────────────────────────────────────────────────────────────────
        df_call = (
                df_opt
                .filter(pl.col("pc") == "C")
                .rename({"symbol": "call_symbol", **{c: f"call_{c}" for c in opt_val_cols}})
                .select(call_put_keys + ["call_symbol"] + [f"call_{c}" for c in opt_val_cols])
        )

        # ── puts ─────────────────────────────────────────────────────────────────
        df_put = (
                df_opt
                .filter(pl.col("pc") == "P")
                .rename({"symbol": "put_symbol", **{c: f"put_{c}" for c in opt_val_cols}})
                .select(call_put_keys + ["put_symbol"] + [f"put_{c}" for c in opt_val_cols])
        )

        # ── call/put join ─────────────────────────────────────────────────────────
        df = df_call.join(df_put, on=call_put_keys, how="inner")

        # ── futures — inner join on (timestamp, fut_sym) ──────────────────────────
        df_fut_sel = (
                df_fut
                .select(["timestamp", "symbol"] + leg_val_cols)
                .rename({"symbol": "fut_sym", **{c: f"fut_{c}" for c in leg_val_cols}})
        )
        df = df.join(df_fut_sel, on=["timestamp", "fut_sym"], how="inner")

        # ── spot — left join on (timestamp, spot_sym) ─────────────────────────────
        df_spot_sel = (
                df_spot
                .select(["timestamp", "symbol"] + leg_val_cols)
                .rename({"symbol": "spot_sym", **{c: f"spot_{c}" for c in leg_val_cols}})
        )
        df = df.join(df_spot_sel, on=["timestamp", "spot_sym"], how="left")

        return df

def testme():
        df = create_okex_aligned_options("2026-01-01")
        print(df.schema)
        print(df.head())

def build_cli_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Build aligned options dataframe for an exchange/day.", formatter_class=_CliHelpFormatter, allow_abbrev=False)
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
                "Parsed align-put-call-data args: exchange=%s date=%s sample_freq=%s recreate_existing=%s raw_data_dir=%s output=%s",
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
                        f"{args.exchange}_aligned_options_{args.date}_{args.sample_freq}.parquet"
                )

        out = Path(output_path)
        if out.exists() and not args.recreate_existing:
                logger.info("Output already exists and recreate_existing is false, skipping: %s", out)
                print(f"skipped_existing={out}")
                return

        raw_data_dir = args.raw_data_dir.format(exchange=args.exchange)
        logger.debug("Resolved raw_data_dir=%s", raw_data_dir)

        if args.exchange == "okex":
                logger.debug("Running OKEX alignment pipeline")
                df = create_okex_aligned_options(args.date, sample_freq=args.sample_freq, raw_data_dir=raw_data_dir)
        else:
                logger.debug("Running Deribit alignment pipeline")
                df = create_deribit_aligned_options(args.date, sample_freq=args.sample_freq, raw_data_dir=raw_data_dir)

        out.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Writing aligned parquet to %s", out)
        df.write_parquet(out)
        logger.info("Aligned dataframe written: path=%s rows=%d cols=%d", out, df.height, df.width)

        print(f"wrote={out} rows={df.height} cols={df.width}")


def run_cli(argv: list[str] | None = None) -> None:
        parser = build_cli_parser()
        args = parser.parse_args(argv)
        run_cli_args(args)
