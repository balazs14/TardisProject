import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl

from tardis.download_and_convert import download_and_convert


logger = logging.getLogger(__name__)
TIMESTAMP_UNIT = "us"

@dataclass(frozen=True)
class SampleConfig:
    data_dir: str = "datasets/daily_parquets"
    output_dir: str = "datasets/daily_parquets"
    freq: str = "5min"
    post_resample_freq: str | None = None
    force_reload: bool = False
    cleanup_csv: bool = True
    cleanup_intermediate_parquet: bool = False
    timestamp_unit: str = TIMESTAMP_UNIT


# Manually maintained recipes per exchange.
EXCHANGE_DOWNLOADS: dict[str, dict[str, object]] = {
    "okex": {
        "static": [
            ("okex-options", "quotes", "OPTIONS"),
            ("okex", "quotes", "BTC-USDT"),
            ("okex", "quotes", "ETH-USDT"),
            ("okex-options","trades", "OPTIONS")
        ],
        "futures_quotes_are_in_separate_files": True,
        "futures_ticker": ("okex-futures", "derivative_ticker", "FUTURES"),
        "futures_quotes": ("okex-futures", "quotes"),
    },
    "deribit": {
        "static": [
            ("deribit", "derivative_ticker", "FUTURES"),
            ("deribit", "quotes", "OPTIONS"),
            ("deribit", "quotes", "BTC_USDC"),
            ("deribit", "quotes", "ETH_USDC"),
            ("deribit", "trades", "OPTIONS")
        ],
    }
}


def _to_ts(df: pl.DataFrame, source_col: str, out_col: str, timestamp_unit: str) -> pl.DataFrame:
    dtype = df.schema[source_col]
    if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        return df.with_columns(pl.from_epoch(pl.col(source_col), time_unit=timestamp_unit).alias(out_col))
    return df.with_columns(pl.col(source_col).cast(pl.Datetime(timestamp_unit)).alias(out_col))


def _polars_freq(freq: str) -> str:
    f = freq.strip().lower()
    if f.endswith("min"):
        return f[:-3] + "m"
    return f

def _normalize_trades(exchange, df: pl.DataFrame, timestamp_unit: str) -> pl.DataFrame:
    if df.is_empty():
        logger.debug("_normalize_trades empty input")
        return df

    if "timestamp" not in df.columns:
        raise ValueError("No exchange timestamp column found in trades")
    logger.debug(
        "_normalize_trades start rows=%d columns=%s",
        df.height,
        df.columns,
    )
    df = _to_ts(df, "timestamp", "ts", timestamp_unit)

    cols = ['symbol','ts','trade_price_amount','trade_amount','trade_signed_amount']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.debug("_normalize_trades filling missing columns=%s", missing)
        df = df.with_columns([pl.lit(False if c == "stale" else None).alias(c) for c in missing])
    logger.debug("_normalize_trades done rows=%d selected_cols=%d", df.height, len(cols))
    return df.select(cols)


def _normalize_okex_quotes(df: pl.DataFrame, timestamp_unit: str) -> pl.DataFrame:
    if df.is_empty():
        logger.debug("_normalize_okex_quotes empty input")
        return df

    if "timestamp" not in df.columns:
        raise ValueError("No exchange timestamp column found in okex quotes")
    logger.debug(
        "_normalize_okex_quotes start rows=%d columns=%s",
        df.height,
        df.columns,
    )
    df = _to_ts(df, "timestamp", "ts", timestamp_unit)
    parts = pl.col("symbol").str.split("-")

    df = df.with_columns(
        [
            parts.list.get(0).alias("_A"),
            parts.list.get(1).alias("_B"),
            parts.list.get(2, null_on_oob=True).alias("exp_str"),
            parts.list.get(3, null_on_oob=True).alias("_D"),
            parts.list.get(4, null_on_oob=True).alias("_E"),
        ]
    )
    logger.debug("_normalize_okex_quotes parsed symbol parts rows=%d", df.height)

    df = df.with_columns(
        [
            pl.when(pl.col("_D").is_not_null())
            .then(pl.col("_D").cast(pl.Float64, strict=False))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("strike"),
            pl.when(pl.col("exp_str").is_null())
            .then(pl.lit("S"))
            .when(pl.col("_E").is_null())
            .then(pl.lit("F"))
            .otherwise(pl.col("_E"))
            .alias("pc"),
            pl.when((pl.col("_B").is_in(["USD", "USDT"]) & pl.col("_E").is_in(["C", "P"])))
            .then(pl.lit("USDT"))
            .otherwise(pl.col("_B"))
            .alias("_BT"),
            pl.lit("USD").alias("_BB"),
            pl.col("exp_str").str.strptime(pl.Date, format="%y%m%d", strict=False).cast(pl.Datetime(timestamp_unit)).alias("exp"),
        ]
    )
    logger.debug(
        "_normalize_okex_quotes derived fields rows=%d columns=%s",
        df.height,
        df.columns,
    )

    df = df.with_columns(pl.col("exp").dt.strftime("%Y-%m-%d").alias("exp_str"))

    df = df.with_columns(
        [
            pl.concat_str([pl.col("_A"), pl.col("_BT"), pl.col("exp_str")], separator="-").alias("fut_sym"),
            pl.concat_str([pl.col("_A"), pl.col("_BT")], separator="-").alias("spot_sym"),
            pl.concat_str([pl.col("_A"), pl.col("_BB")], separator="").alias("ref_sym"),
        ]
    )

    cols = [
        "ts",
        "symbol",
        "ref_sym",
        "fut_sym",
        "spot_sym",
        "exp",
        "exp_str",
        "pc",
        "strike",
        "bid_price",
        "ask_price",
        "bid_amount",
        "ask_amount",
        "bid_iv",
        "ask_iv",
        "mark_iv",
        "mark_price",
        "stale",

    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.debug("_normalize_okex_quotes filling missing columns=%s", missing)
        df = df.with_columns([pl.lit(False if c == "stale" else None).alias(c) for c in missing])
    logger.debug("_normalize_okex_quotes done rows=%d selected_cols=%d", df.height, len(cols))
    return df.select(cols)


def _normalize_deribit_quotes(df: pl.DataFrame, timestamp_unit: str) -> pl.DataFrame:
    if df.is_empty():
        logger.debug("_normalize_deribit_quotes empty input")
        return df

    if "timestamp" not in df.columns:
        raise ValueError("No timestamp column found in deribit quotes")

    logger.debug(
        "_normalize_deribit_quotes start rows=%d columns=%s",
        df.height,
        df.columns,
    )

    df = _to_ts(df, "timestamp", "ts", timestamp_unit)
    parts = pl.col("symbol").str.split("-")

    df = df.with_columns(
        [
            parts.list.get(0).alias("_A"),
            parts.list.get(1, null_on_oob=True).alias("_B"),
            parts.list.get(2, null_on_oob=True).alias("_C"),
            parts.list.get(3, null_on_oob=True).alias("_D"),
        ]
    )

    # Transformation 1: classify instrument side in `pc` from null-pattern of _A/_B/_C.
    df = df.with_columns(
        [
            pl.when(pl.col("_A").is_not_null() & pl.col("_B").is_null() & pl.col("_C").is_null())
            .then(pl.lit("S"))
            .when(pl.col("_A").is_not_null() & pl.col("_B").is_not_null() & pl.col("_C").is_null())
            .then(pl.lit("F"))
            .when(pl.col("_A").is_not_null() & pl.col("_B").is_not_null() & pl.col("_C").is_not_null())
            .then(
                pl.when(pl.col("symbol").str.ends_with("-C"))
                .then(pl.lit("C"))
                .when(pl.col("symbol").str.ends_with("-P"))
                .then(pl.lit("P"))
                .otherwise(pl.lit(None, dtype=pl.String))
            )
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("pc"),


        ]
    )

    # Depending on `pc`, futures use mark_price while other instruments keep bid/ask fields.
    mark_price_expr = pl.col("mark_price") if "mark_price" in df.columns else pl.lit(None, dtype=pl.Float64)
    bid_price_expr = pl.col("bid_price") if "bid_price" in df.columns else mark_price_expr
    ask_price_expr = pl.col("ask_price") if "ask_price" in df.columns else mark_price_expr
    bid_amount_expr = pl.col("bid_amount") if "bid_amount" in df.columns else pl.lit(None, dtype=pl.Float64)
    ask_amount_expr = pl.col("ask_amount") if "ask_amount" in df.columns else pl.lit(None, dtype=pl.Float64)
    df = df.with_columns(
        [
            pl.when(pl.col("pc") == "F").then(mark_price_expr).otherwise(bid_price_expr).alias("bid_price"),
            pl.when(pl.col("pc") == "F").then(mark_price_expr).otherwise(ask_price_expr).alias("ask_price"),
            pl.when(pl.col("pc") == "F").then(pl.lit(0.0)).otherwise(bid_amount_expr).alias("bid_amount"),
            pl.when(pl.col("pc") == "F").then(pl.lit(0.0)).otherwise(ask_amount_expr).alias("ask_amount"),
        ]
    )

    # Transformation 2: cast parsed strike text (`_C`) into numeric strike.
    df = df.with_columns(
        [
            pl.col("_C").cast(pl.Float64, strict=False).alias("strike"),
        ]
    )

    # Transformation 3: create expiration fields when _B exists and is not PERPETUAL.
    df = df.with_columns(
        [
            pl.when(pl.col("_B").is_not_null() & (pl.col("_B") != "PERPETUAL"))
            .then(pl.col("_B").str.to_uppercase().str.strptime(pl.Date, format="%d%b%y", strict=False).cast(pl.Datetime(timestamp_unit)))
            .otherwise(pl.lit(None, dtype=pl.Datetime(timestamp_unit)))
            .alias("exp"),
            pl.when(pl.col("_B").is_not_null() & (pl.col("_B") != "PERPETUAL"))
            .then(
                pl.col("_B")
                .str.to_uppercase()
                #.str.strptime(pl.Date, format="%d%b%y", strict=False)
                #.dt.strftime("%Y-%m-%d")
            )
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("exp_str"),
        ]
    )

    # Transformation 4: derive option/future/spot join fields.
    df = df.with_columns(
        [
            pl.when(pl.col("pc").is_in(["C", "P", "F"]))
            .then(pl.concat_str([pl.col("_A"), pl.col("_B")], separator="-"))
            .otherwise(pl.col("symbol"))
            .alias("fut_sym"),
            pl.when(pl.col("_A").is_in(["BTC", "ETH"]))
            .then(pl.concat_str([pl.col("_A"), pl.lit("USDC")], separator="_"))
            .otherwise(pl.col("_A"))
            .alias("spot_sym"),
            pl.when(pl.col("_A").is_in(["BTC", "ETH"]))
            .then(pl.concat_str([pl.col("_A"), pl.lit("USD")], separator=""))
            .otherwise(pl.col("_A"))
            .alias("ref_sym"),
        ]
    )
    cols = [
        "ts",
        "symbol",
        "ref_sym",
        "fut_sym",
        "spot_sym",
        "exp",
        "exp_str",
        "pc",
        "strike",
        "bid_price",
        "ask_price",
        "bid_amount",
        "ask_amount",
        "bid_iv",
        "ask_iv",
        "mark_iv",
        "mark_price",
        "stale",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(False if c == "stale" else None).alias(c) for c in missing])
    logger.debug("_normalize_deribit_quotes done rows=%d selected_cols=%d", df.height, len(cols))
    return df.select(cols)


def _normalize_quotes(exchange: str, df: pl.DataFrame, timestamp_unit: str) -> pl.DataFrame:
    if exchange == "okex":
        return _normalize_okex_quotes(df, timestamp_unit)
    if exchange == "deribit":
        return _normalize_deribit_quotes(df, timestamp_unit)
    raise NotImplementedError(f"No normalizer for exchange={exchange}")


def _resample_normalized_quotes(df: pl.DataFrame, freq: str, timestamp_unit: str) -> pl.DataFrame:
    if df.is_empty():
        return df

    polars_freq = _polars_freq(freq)

    base_value_cols = [
        "ref_sym",
        "fut_sym",
        "spot_sym",
        "exp",
        "exp_str",
        "pc",
        "strike",
        "bid_price",
        "ask_price",
        "bid_amount",
        "ask_amount",
        "bid_iv",
        "ask_iv",
        "mark_iv",
        "mark_price",
    ]
    static_cols = ["ref_sym", "fut_sym", "spot_sym", "exp", "exp_str", "pc", "strike"]
    price_cols = ["bid_price", "ask_price", "bid_amount", "ask_amount", "bid_iv", "ask_iv", "mark_iv", "mark_price"]
    volume_cols = [c for c in df.columns if c not in {"ts", "symbol"} and "volume" in c.lower()]
    last_value_cols = base_value_cols
    value_cols = base_value_cols + volume_cols

    df = df.select(["ts", "symbol"] + value_cols)
    symbols = df.select("symbol").unique().sort("symbol").get_column("symbol").to_list()

    out = []
    for symbol in symbols:
        sdf = df.filter(pl.col("symbol") == symbol)
        raw_rows = sdf.height
        sdf = sdf.sort("ts")
        exact_ts = (
            sdf.group_by("ts", maintain_order=True)
            .agg(
                [
                    *[pl.col(c).last().alias(c) for c in last_value_cols],
                    *[pl.col(c).sum().alias(c) for c in volume_cols],
                ]
            )
            .sort("ts")
        )

        # A row is stale for this 5-min bucket when neither bid nor ask updated in the bucket.
        raw_bucket_updates = (
            sdf.with_columns(pl.col("ts").dt.truncate(polars_freq).alias("_bucket_ts"))
            .group_by("_bucket_ts", maintain_order=True)
            .agg(
                [
                    pl.col("bid_price").is_not_null().any().alias("_bid_updated"),
                    pl.col("ask_price").is_not_null().any().alias("_ask_updated"),
                ]
            )
            .rename({"_bucket_ts": "ts"})
            .sort("ts")
        )
        bucketed = exact_ts.with_columns(pl.col("ts").dt.truncate(polars_freq).alias("_bucket_ts"))
        aggregated = (
            bucketed.group_by("_bucket_ts", maintain_order=True)
            .agg(
                [
                    *[pl.col(c).last().alias(c) for c in last_value_cols],
                    *[pl.col(c).sum().alias(c) for c in volume_cols],
                ]
            )
            .rename({"_bucket_ts": "ts"})
            .sort("ts")
        )
        aggregated = aggregated.join(raw_bucket_updates, on="ts", how="left")

        t_start = pd.Timestamp(sdf.select(pl.col("ts").min()).item()).floor(freq)
        t_end = pd.Timestamp(sdf.select(pl.col("ts").max()).item()).ceil(freq)
        grid = pl.Series(pd.date_range(t_start, t_end, freq=freq)).cast(pl.Datetime(timestamp_unit))
        filled = pl.DataFrame({"ts": grid}).join(aggregated, on="ts", how="left")
        filled = filled.with_columns(
            (
                (~pl.col("_bid_updated").fill_null(False))
                & (~pl.col("_ask_updated").fill_null(False))
            ).alias("stale")
        ).drop(["_bid_updated", "_ask_updated"])
        filled = filled.with_columns([pl.col(c).fill_null(strategy="forward").alias(c) for c in price_cols])
        filled = filled.with_columns([pl.col(c).fill_null(strategy="forward").alias(c) for c in static_cols])
        filled = filled.with_columns([pl.col(c).fill_null(pl.col(c).drop_nulls().first()).alias(c) for c in static_cols])
        if volume_cols:
            filled = filled.with_columns([pl.col(c).fill_null(0).alias(c) for c in volume_cols])
        filled = filled.with_columns(pl.lit(symbol).alias("symbol"))
        out.append(filled.select(["ts", "symbol"] + value_cols + ["stale"]))

    res = pl.concat(out, rechunk=False).sort(["symbol", "ts"])
    logger.debug("_resample_normalized_quotes done rows=%d columns=%s", res.height, res.columns)
    return res

def _resample_normalized_trades(df: pl.DataFrame, freq: str, timestamp_unit: str) -> pl.DataFrame:
    if df.is_empty():
        return df

    polars_freq = _polars_freq(freq)

    df = df.select(["ts", "symbol",'trade_amount','trade_signed_amount', 'trade_price_amount'])
    symbols = df.select("symbol").unique().sort("symbol").get_column("symbol").to_list()

    out = []
    for symbol in symbols:
        sdf = df.filter(pl.col("symbol") == symbol)
        raw_rows = sdf.height
        sdf = sdf.sort("ts")
        exact_ts = (
            sdf.group_by("ts", maintain_order=True)
            .agg(
                    [pl.col(c).sum().alias(c) for c in ['trade_amount','trade_signed_amount', 'trade_price_amount']]
            )
            .sort("ts")
        )

        # A row is stale for this bucket when there are no trades in the bucket.
        raw_bucket_updates = (
            sdf.with_columns(pl.col("ts").dt.truncate(polars_freq).alias("_bucket_ts"))
            .group_by("_bucket_ts", maintain_order=True)
            .agg(
            [
                (pl.col("trade_amount").is_not_null().any()).alias("_trade_updated"),
            ]
            )
            .rename({"_bucket_ts": "ts"})
            .sort("ts")
        )

        bucketed = exact_ts.with_columns(pl.col("ts").dt.truncate(polars_freq).alias("_bucket_ts"))
        aggregated = (
            bucketed.group_by("_bucket_ts", maintain_order=True)
            .agg(
                [pl.col(c).sum().alias(c) for c in ['trade_amount','trade_signed_amount', 'trade_price_amount']],
            )
            .rename({"_bucket_ts": "ts"})
            .sort("ts")
        )
        aggregated = aggregated.join(raw_bucket_updates, on="ts", how="left")

        t_start = pd.Timestamp(sdf.select(pl.col("ts").min()).item()).floor(freq)
        t_end = pd.Timestamp(sdf.select(pl.col("ts").max()).item()).ceil(freq)
        grid = pl.Series(pd.date_range(t_start, t_end, freq=freq)).cast(pl.Datetime(timestamp_unit))
        filled = pl.DataFrame({"ts": grid}).join(aggregated, on="ts", how="left")
        filled = filled.with_columns([pl.col(c).fill_null(0).alias(c) for c in ['trade_amount','trade_signed_amount', 'trade_price_amount']])
        filled = filled.with_columns(pl.lit(symbol).alias("symbol"))
        out.append(filled.select(["ts", "symbol",'trade_amount','trade_signed_amount', 'trade_price_amount']))

    res = pl.concat(out, rechunk=False).sort(["symbol", "ts"])
    logger.debug("_resample_normalized_trades done rows=%d columns=%s", res.height, res.columns)
    return res


def _align_calls_puts_with_legs(resampled_df: pl.DataFrame, resampled_trade_df: pl.DataFrame) -> pl.DataFrame:
    logger.debug("_align_calls_puts_with_legs start rows=%d", resampled_df.height)

    # --- partition legs ---
    fut = resampled_df.filter(pl.col("pc") == "F").rename({
        "ask_amount": "fut_ask_amount",
        "ask_price": "fut_ask_price",
        "bid_amount": "fut_bid_amount",
        "bid_price": "fut_bid_price",
        "stale": "fut_stale",
        "exp": "fut_exp",
    })
    spot = resampled_df.filter(pl.col("pc") == "S").rename({
        "ask_amount": "spot_ask_amount",
        "ask_price": "spot_ask_price",
        "bid_amount": "spot_bid_amount",
        "bid_price": "spot_bid_price",
    })
    opts = resampled_df.filter(pl.col("pc").is_in(["C", "P"]))
    logger.debug(
        "_align_calls_puts_with_legs partitions fut=%d spot=%d opts=%d trades=%d",
        fut.height, spot.height, opts.height, resampled_trade_df.height,
    )
    if opts.is_empty():
        logger.debug("_align_calls_puts_with_legs empty options")
        return pl.DataFrame()

    # --- join trades to each option row by symbol+ts ---
    trade_keep = ["symbol", "ts", "trade_amount","trade_signed_amount","trade_price_amount"]
    trades = resampled_trade_df.select(trade_keep)
    opts = opts.join(trades, on=["symbol", "ts"], how="left")

    # --- only keep pairs that have at least one call and one put ---
    pair_keys = ["ref_sym", "fut_sym", "exp", "strike", "ts"]

    pair_universe = (
        opts.group_by(pair_keys)
        .agg([
            pl.col("pc").eq("C").any().alias("_has_call"),
            pl.col("pc").eq("P").any().alias("_has_put"),
        ])
        .filter(pl.col("_has_call") & pl.col("_has_put"))
        .select(pair_keys)
    )
    if pair_universe.is_empty():
        logger.debug("_align_calls_puts_with_legs empty pair universe")
        return pl.DataFrame()

    opts = opts.join(pair_universe, on=pair_keys, how="inner")

    # --- rename all non-key opt columns with call_/put_ prefix, then join on pair+ts ---
    opt_value_cols = [c for c in opts.columns if c not in set(pair_keys + ["pc"])]
    calls = opts.filter(pl.col("pc") == "C").drop("pc").rename({c: f"call_{c}" for c in opt_value_cols})
    puts  = opts.filter(pl.col("pc") == "P").drop("pc").rename({c: f"put_{c}"  for c in opt_value_cols})

    aligned = calls.join(puts, on=pair_keys, how="full", coalesce=True).sort(pair_keys)
    logger.debug("_align_calls_puts_with_legs joined call/put rows=%d", aligned.height)

    # --- forward-fill call and put columns over each pair ---
    call_cols = [c for c in aligned.columns if c.startswith("call_")]
    put_cols  = [c for c in aligned.columns if c.startswith("put_")]
    if call_cols:
        aligned = aligned.with_columns([pl.col(c).fill_null(strategy="forward").over(pair_keys) for c in call_cols])
    if put_cols:
        aligned = aligned.with_columns([pl.col(c).fill_null(strategy="forward").over(pair_keys) for c in put_cols])
    logger.debug(
        "_align_calls_puts_with_legs forward_filled call_cols=%d put_cols=%d",
        len(call_cols), len(put_cols),
    )

    # derive spot_sym from whichever option side carries it
    spot_sym_sources = [c for c in ("call_spot_sym", "put_spot_sym") if c in aligned.columns]
    if spot_sym_sources:
        aligned = aligned.with_columns(
            pl.coalesce(spot_sym_sources).alias("spot_sym")
        ).with_columns(pl.col("spot_sym").fill_null(strategy="forward").over(pair_keys))

    # --- join future and spot legs ---
    aligned = aligned.join(
        fut.select(["ts", "fut_sym", "fut_ask_amount", "fut_ask_price", "fut_bid_amount", "fut_bid_price", "fut_stale", "fut_exp"]),
        on=["ts", "fut_sym"],
        how="left",
    )
    if "spot_sym" in aligned.columns:
        aligned = aligned.join(
            spot.select(["ts", "spot_sym", "spot_ask_amount", "spot_ask_price", "spot_bid_amount", "spot_bid_price"]),
            on=["ts", "spot_sym"],
            how="left",
        )

    fut_price_cols  = ["fut_ask_amount", "fut_ask_price", "fut_bid_amount", "fut_bid_price", "fut_exp", "fut_stale"]
    spot_price_cols = ["spot_ask_amount", "spot_ask_price", "spot_bid_amount", "spot_bid_price"]
    aligned = aligned.with_columns([pl.col(c).fill_null(strategy="forward").over("fut_sym") for c in fut_price_cols if c in aligned.columns])
    if "spot_sym" in aligned.columns:
        aligned = aligned.with_columns([pl.col(c).fill_null(strategy="forward").over("spot_sym") for c in spot_price_cols if c in aligned.columns])
    logger.debug("_align_calls_puts_with_legs after leg joins rows=%d", aligned.height)

    extra_cols = [c for c in aligned.columns if c.endswith("_right") or c in {"call_spot_sym", "put_spot_sym"}]
    if extra_cols:
        aligned = aligned.drop(extra_cols)

    logger.debug("_align_calls_puts_with_legs done rows=%d columns=%s", aligned.height, aligned.columns)
    return aligned.sort(pair_keys)


def testme():
    return sample_day_options('2026-01-01','okex',SampleConfig(freq='5min'))

def sample_day_options(day: str, exchange: str, config: SampleConfig = SampleConfig()) -> pl.DataFrame:
    """Download, normalize, resample, and align call/put/future/spot legs for one exchange/day."""
    align_freq = config.post_resample_freq or config.freq
    logger.debug(
        "sample_day_options start exchange=%s day=%s freq=%s align_freq=%s force_reload=%s cleanup_csv=%s cleanup_intermediate_parquet=%s",
        exchange,
        day,
        config.freq,
        align_freq,
        config.force_reload,
        config.cleanup_csv,
        config.cleanup_intermediate_parquet,
    )
    recipe = EXCHANGE_DOWNLOADS.get(exchange)
    if recipe is None:
        raise NotImplementedError(f"No manual recipe for exchange={exchange}")
    logger.debug("sample_day_options recipe_keys exchange=%s keys=%s", exchange, list(recipe.keys()))

    output_path = Path(config.output_dir) / f"{exchange}_aligned_options_{day}_{align_freq}.parquet"
    if output_path.exists() and not config.force_reload:
        logger.debug("sample_day_options cache_hit exchange=%s day=%s path=%s", exchange, day, output_path)
        return pl.read_parquet(output_path)

    quote_paths: list[Path] = []
    trade_paths: list[Path] = []
    for ex, dt, sym in recipe["static"]:
        logger.debug("sample_day_options static_download exchange=%s data_type=%s symbol=%s day=%s", ex, dt, sym, day)
        paths = download_and_convert(
            exchange=ex,
            data_type=dt,
            symbol=sym,
            start_date=day,
            end_date=day,
            data_dir=config.data_dir,
            force_reload=config.force_reload,
            cleanup_csv=config.cleanup_csv,
            resample_freq=config.freq,
        )
        logger.debug("sample_day_options static_download_result symbol=%s files=%s", sym, [str(path) for path in paths])
        if dt in ['quotes','derivative_ticker']: quote_paths.extend(paths)
        elif dt in ['trades']: trade_paths.extend(paths)
        else: assert False, 'need to add extra datatype'
    logger.debug(
        "sample_day_options static_downloads exchange=%s day=%s quote files=%d, trade files=%d",
        exchange,
        day,
        len(quote_paths),
        len(trade_paths)
    )

    if recipe.get("futures_quotes_are_in_separate_files", False):
        ticker_ex, ticker_dt, ticker_sym = recipe["futures_ticker"]
        ticker_path = download_and_convert(
            exchange=ticker_ex,
            data_type=ticker_dt,
            symbol=ticker_sym,
            start_date=day,
            end_date=day,
            data_dir=config.data_dir,
            force_reload=config.force_reload,
            cleanup_csv=config.cleanup_csv,
            resample_freq=config.freq,
        )
        logger.debug("sample_day_options futures_ticker_path=%s", ticker_path)
        fut_symbols = (
            pl.scan_parquet(ticker_path)
            .select(pl.col("symbol"))
            .unique()
            .collect()
            .get_column("symbol")
            .to_list()
        )

        logger.debug(
            "sample_day_options futures_discovered exchange=%s day=%s symbols=%d",
            exchange,
            day,
            len(fut_symbols),
        )

        fut_ex, fut_dt = recipe["futures_quotes"]
        for fut_symbol in fut_symbols:
            logger.debug("sample_day_options futures_quote_download symbol=%s", fut_symbol)
            quote_paths.extend(
                download_and_convert(
                    exchange=fut_ex,
                    data_type=fut_dt,
                    symbol=fut_symbol,
                    start_date=day,
                    end_date=day,
                    data_dir=config.data_dir,
                    force_reload=config.force_reload,
                    cleanup_csv=config.cleanup_csv,
                    resample_freq=config.freq,
                )
            )

    logger.debug("sample_day_options quote_paths=%s", [str(path) for path in quote_paths])
    logger.debug("sample_day_options trade_paths=%s", [str(path) for path in trade_paths])

    quote_frames = [
        _normalize_quotes(exchange, pl.read_parquet(path), config.timestamp_unit)
        for path in quote_paths
        if path.exists()
    ]
    trade_frames = [
        _normalize_trades(exchange, pl.read_parquet(path), config.timestamp_unit)
        for path in trade_paths
        if path.exists()
    ]
    logger.debug(
        "sample_day_options existing_quote_paths=%d missing_quote_paths=%d",
        sum(1 for path in quote_paths if path.exists()),
        sum(1 for path in quote_paths if not path.exists()),
    )
    if not quote_frames:
        raise RuntimeError(f"No quote data found for exchange={exchange} day={day}")

    normalized_quotes = pl.concat(quote_frames)
    normalized_trades = pl.concat(trade_frames)
    logger.debug(
        "sample_day_options normalized exchange=%s day=%s quote frames=%d rows=%d",
        exchange, day, len(quote_frames), normalized_quotes.height)
    logger.debug(
        "sample_day_options normalized exchange=%s day=%s trade frames=%d rows=%d",
        exchange, day, len(trade_frames), normalized_trades.height )
    sampled_quotes = _resample_normalized_quotes(normalized_quotes, align_freq, config.timestamp_unit)
    sampled_trades = _resample_normalized_trades(normalized_trades, align_freq, config.timestamp_unit)
    aligned = _align_calls_puts_with_legs(sampled_quotes, sampled_trades)
    logger.debug(
        "sample_day_options aligned exchange=%s day=%s aligned_rows=%d",
        exchange, day, aligned.height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("sample_day_options ensuring output dir=%s", output_path.parent)
    aligned.write_parquet(output_path)
    logger.debug("sample_day_options wrote exchange=%s day=%s path=%s", exchange, day, output_path)
    if config.cleanup_intermediate_parquet:
        for pa in quote_paths:
            logger.debug("sample_day_options cleanup_intermediate_parquet unlink path=%s", pa)
            pa.unlink(missing_ok=True)
            pkl = Path(str(pa) + ".pkl")
            if pkl.exists():
                logger.debug("sample_day_options cleanup_intermediate_parquet unlink pkl=%s", pkl)
                pkl.unlink(missing_ok=True)
        logger.debug("sample_day_options cleanup_intermediate_parquet exchange=%s day=%s files=%d", exchange, day, len(quote_paths))
    return aligned
