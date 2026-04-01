import argparse
from pathlib import Path

import polars as pl

from tardis.markup_files import access_files, mark_up


def create_okex_aligned_options(date: str, sample_freq: str = "5min") -> pl.DataFrame:
	df_opt, _ = access_files("okex-options", "quotes", date, "OPTIONS", sample_freq=sample_freq)
	df_opt = mark_up(df_opt, "okex-options", "quotes")

	fut_symbols = (
		df_opt.select(pl.col("fut_sym").drop_nulls().unique().sort())
		.to_series()
		.to_list()
	)

	df_fut, _ = access_files("okex-futures", "quotes", date, fut_symbols, sample_freq=sample_freq)
	df_fut = mark_up(df_fut, "okex-futures", "quotes")

	df_spot, _ = access_files("okex", "quotes", date, ["BTC-USDT", "ETH-USDT"], sample_freq=sample_freq)
	df_spot = mark_up(df_spot, "okex", "quotes")

	df_result = merge_pcp_data(df_opt, df_fut, df_spot)
	df_result = df_result.with_columns(pl.lit("okex").alias("exchange"))

	return df_result


def create_deribit_aligned_options(date: str, sample_freq: str = "5min") -> pl.DataFrame:
	df_opt, _ = access_files("deribit", "quotes", date, "OPTIONS", sample_freq=sample_freq)
	df_opt = mark_up(df_opt, "deribit", "quotes", symbol_type="option")

	df_fut, _ = access_files("deribit", "derivative_ticker", date, "FUTURES", sample_freq=sample_freq)
	assert "mark_price" in df_fut.columns, "df_fut must have a mark_price column"
	df_fut = df_fut.with_columns(
		[
			pl.col("mark_price").alias("bid_price"),
			pl.col("mark_price").alias("ask_price"),
			pl.lit(None, dtype=pl.Float64).alias("bid_amount"),
			pl.lit(None, dtype=pl.Float64).alias("ask_amount"),
		]
	)

	df_spot, _ = access_files("deribit", "quotes", date, ["ETH_USDC", "BTC_USDC"], sample_freq=sample_freq)
	df_spot = mark_up(df_spot, "deribit", "quotes", symbol_type="spot")

	df_result = merge_pcp_data(df_opt, df_fut, df_spot)
	df_result = df_result.with_columns(pl.lit("deribit").alias("exchange"))

	return df_result


def merge_pcp_data(
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


def _main() -> None:
	parser = argparse.ArgumentParser(description="Build aligned options dataframe for an exchange/day.")
	parser.add_argument("--exchange", required=True, choices=["okex", "deribit"], help="Exchange to process")
	parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
	parser.add_argument("--sample-freq", default="5min", help="Sampling frequency label for output naming")
	parser.add_argument(
		"--output",
		default=None,
		help="Output parquet path. Default: datasets/{exchange}/{exchange}_aligned_options_{date}_{sample_freq}.parquet",
	)
	args = parser.parse_args()

	if args.exchange == "okex":
		df = create_okex_aligned_options(args.date, sample_freq=args.sample_freq)
	else:
		df = create_deribit_aligned_options(args.date, sample_freq=args.sample_freq)

	output_path = args.output
	if output_path is None:
		output_path = (
			f"datasets/{args.exchange}/"
			f"{args.exchange}_aligned_options_{args.date}_{args.sample_freq}.parquet"
		)

	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	df.write_parquet(out)

	print(f"wrote={out} rows={df.height} cols={df.width}")


if __name__ == "__main__":
	_main()
