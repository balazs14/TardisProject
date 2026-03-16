import logging
import os
import pickle
import csv
import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import requests
from tardis_dev import datasets


logger = logging.getLogger(__name__)

# Use microsecond resolution for all timestamps
TIMESTAMP_UNIT = 'us'

@dataclass
class Metadata:
    symbols: list[str]
    symbol_timestamp_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    global_first_timestamp: pd.Timestamp | None
    global_last_timestamp: pd.Timestamp | None
    extras: dict[str, object] = field(default_factory=dict)


@dataclass
class _OnlineResampleState:
    current_bucket: pd.Timestamp | None = None
    last_vals: dict[str, object] = field(default_factory=dict)
    sum_vals: dict[str, float] = field(default_factory=dict)
    bid_updated: bool = False
    ask_updated: bool = False
    last_emitted_bucket: pd.Timestamp | None = None
    last_emitted_last_vals: dict[str, object] = field(default_factory=dict)


def _build_tardis_column_types() -> dict[str, pa.DataType]:
    types: dict[str, pa.DataType] = {
        # Core/common
        "exchange": pa.string(),
        "data_type": pa.string(),
        "channel": pa.string(),
        "symbol": pa.string(),
        "instrument_name": pa.string(),
        "market": pa.string(),
        "type": pa.string(),
        "state": pa.string(),
        "side": pa.string(),
        "action": pa.string(),
        "id": pa.string(),
        "trade_id": pa.string(),
        "sequence": pa.int64(),
        "seq": pa.int64(),
        "version": pa.int64(),
        "count": pa.int64(),
        "timestamp": pa.timestamp(TIMESTAMP_UNIT),
        "local_timestamp": pa.timestamp(TIMESTAMP_UNIT),
        'mdy': pa.date32(),

        # Prices / sizes / quantities
        "price": pa.float64(),
        "mark_price": pa.float64(),
        "index_price": pa.float64(),
        "underlying_price": pa.float64(),
        "settlement_price": pa.float64(),
        "last_price": pa.float64(),
        "open_price": pa.float64(),
        "high_price": pa.float64(),
        "low_price": pa.float64(),
        "close_price": pa.float64(),
        "avg_price": pa.float64(),
        "funding_rate": pa.float64(),
        "predicted_funding_rate": pa.float64(),
        "interest_rate": pa.float64(),
        "bid_price": pa.float64(),
        "ask_price": pa.float64(),
        "bid_amount": pa.float64(),
        "ask_amount": pa.float64(),
        "quantity": pa.float64(),
        "amount": pa.float64(),
        "size": pa.float64(),
        "volume": pa.float64(),
        "volume_notional": pa.float64(),
        "notional": pa.float64(),
        "open_interest": pa.float64(),
        "open_interest_value": pa.float64(),
        "index": pa.float64(),

        # Options / derivatives
        "strike": pa.float64(),
        "expiration": pa.timestamp(TIMESTAMP_UNIT),
        "expiry": pa.timestamp(TIMESTAMP_UNIT),
        "delivery_price": pa.float64(),
        "underlying_index": pa.string(),
        "contract_size": pa.float64(),
        "contract_value": pa.float64(),
        "contract_type": pa.string(),
        "option_type": pa.string(),
        "implied_volatility": pa.float64(),
        "mark_iv": pa.float64(),
        "bid_iv": pa.float64(),
        "ask_iv": pa.float64(),
        "delta": pa.float64(),
        "gamma": pa.float64(),
        "vega": pa.float64(),
        "theta": pa.float64(),
        "rho": pa.float64(),

        # Liquidations / taker info
        "liquidation": pa.bool_(),
        "is_liquidation": pa.bool_(),
        "taker_side": pa.string(),
        "taker_buy_base_volume": pa.float64(),
        "taker_buy_quote_volume": pa.float64(),

        # Funding / basis / index constituents
        "basis": pa.float64(),
        "basis_rate": pa.float64(),
        "next_funding_time": pa.timestamp(TIMESTAMP_UNIT),
        "funding_timestamp": pa.timestamp(TIMESTAMP_UNIT),
        "index_timestamp": pa.timestamp(TIMESTAMP_UNIT),

        # Trade / exchange-specific frequently seen fields
        "tick_direction": pa.string(),
        "tickDirection": pa.string(),
        "trdMatchID": pa.string(),
        "grossValue": pa.float64(),
        "homeNotional": pa.float64(),
        "foreignNotional": pa.float64(),
        "quote_id": pa.string(),
        "order_id": pa.string(),
        "client_order_id": pa.string(),
        "match_id": pa.string(),
        "trade_seq": pa.int64(),
        "event_time": pa.timestamp(TIMESTAMP_UNIT),
        "transaction_time": pa.timestamp(TIMESTAMP_UNIT),
        "recv_window": pa.int64(),

        # OHLC-like aggregates (some feeds)
        "open": pa.float64(),
        "high": pa.float64(),
        "low": pa.float64(),
        "close": pa.float64(),
        "quote_volume": pa.float64(),
        "base_volume": pa.float64(),
        "num_trades": pa.int64(),

        # Misc string metadata
        "currency": pa.string(),
        "base_currency": pa.string(),
        "quote_currency": pa.string(),
        "settlement_currency": pa.string(),
        "underlying_symbol": pa.string(),
        "index_name": pa.string(),
        "status": pa.string(),
        "reason": pa.string(),
    }

    # Book snapshot style columns (top levels).
    for level in range(1, 101):
        types[f"bid_price_{level}"] = pa.float64()
        types[f"bid_amount_{level}"] = pa.float64()
        types[f"ask_price_{level}"] = pa.float64()
        types[f"ask_amount_{level}"] = pa.float64()

        types[f"bids_{level}_price"] = pa.float64()
        types[f"bids_{level}_amount"] = pa.float64()
        types[f"asks_{level}_price"] = pa.float64()
        types[f"asks_{level}_amount"] = pa.float64()

        types[f"bid{level}_price"] = pa.float64()
        types[f"bid{level}_amount"] = pa.float64()
        types[f"ask{level}_price"] = pa.float64()
        types[f"ask{level}_amount"] = pa.float64()

    return types


TARDIS_COLUMN_TYPES = _build_tardis_column_types()


RESAMPLE_AGGREGATION_BY_COLUMN: dict[str, str] = {
    "price": "last",
    "bid_price": "last",
    "ask_price": "last",
    "mark_price": "last",
    "index_price": "last",
    "open_interest": "last",
    "bid_amount": "last",
    "ask_amount": "last",
    "amount": "sum",
    "quantity": "sum",
    "size": "sum",
    "volume": "sum",
    "trade_amount": "sum",
    "trade_price_amount": "sum",
    "trade_signed_amount": "sum",
    "funding_timestamp": "last",
    "funding_rate": "last",
    "predicted_funding_rate": "last",
}


def _aggregation_for_column(column_name: str) -> str:
    if column_name in RESAMPLE_AGGREGATION_BY_COLUMN:
        return RESAMPLE_AGGREGATION_BY_COLUMN[column_name]
    if "amount" in column_name.lower():
        return "sum"
    return "last"


def _normalize_polars_freq(freq: str) -> str:
    f = freq.strip().lower().replace(" ", "")
    f = f.replace("minutes", "m").replace("minute", "m").replace("mins", "m").replace("min", "m")
    f = f.replace("hours", "h").replace("hour", "h")
    f = f.replace("seconds", "s").replace("second", "s").replace("secs", "s").replace("sec", "s")
    f = f.replace("days", "d").replace("day", "d")
    return f


def _temporal_fallback_column_types(column_types: dict[str, pa.DataType]) -> dict[str, pa.DataType]:
    fallback_types = dict(column_types)
    for col, dtype in column_types.items():
        if pa.types.is_timestamp(dtype) or pa.types.is_date32(dtype) or pa.types.is_date64(dtype):
            fallback_types[col] = pa.int64()
    return fallback_types


def _cast_temporal_fallback_columns(df: pl.DataFrame, column_types: dict[str, pa.DataType]) -> pl.DataFrame:
    cast_exprs: list[pl.Expr] = []
    for col, dtype in column_types.items():
        if col not in df.columns:
            continue
        if df.schema[col] not in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            continue

        if pa.types.is_timestamp(dtype):
            cast_exprs.append(pl.from_epoch(pl.col(col), time_unit=TIMESTAMP_UNIT).alias(col))
        elif pa.types.is_date32(dtype):
            cast_exprs.append(pl.col(col).cast(pl.Int32).cast(pl.Date).alias(col))
        elif pa.types.is_date64(dtype):
            cast_exprs.append(pl.from_epoch(pl.col(col), time_unit="ms").cast(pl.Date).alias(col))

    if not cast_exprs:
        return df
    return df.with_columns(cast_exprs)


def _parse_stream_timestamp(raw_ts: str) -> pd.Timestamp:
    raw_ts = (raw_ts or "").strip()
    if not raw_ts:
        raise ValueError("Empty timestamp")

    if raw_ts.isdigit():
        iv = int(raw_ts)
        n = len(raw_ts)
        if n >= 19:
            return pd.to_datetime(iv, unit="ns")
        if n >= 16:
            return pd.to_datetime(iv, unit="us")
        if n >= 13:
            return pd.to_datetime(iv, unit="ms")
        return pd.to_datetime(iv, unit="s")

    ts = pd.Timestamp(raw_ts)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts


def _parse_stream_value(col: str, raw_val: str):
    raw_val = (raw_val or "").strip()
    if raw_val in {"", "null", "None", "NULL", "NaN", "nan"}:
        return None

    dtype = TARDIS_COLUMN_TYPES.get(col)
    if dtype is None:
        return raw_val

    if pa.types.is_floating(dtype):
        try:
            return float(raw_val)
        except ValueError:
            return None
    if pa.types.is_integer(dtype):
        try:
            return int(raw_val)
        except ValueError:
            return None
    if pa.types.is_boolean(dtype):
        return raw_val.lower() in {"1", "true", "t", "yes", "y"}
    if pa.types.is_timestamp(dtype) or pa.types.is_date32(dtype) or pa.types.is_date64(dtype):
        try:
            return int(raw_val)
        except ValueError:
            return None

    return raw_val


def _tardis_csv_url(exchange: str, data_type: str, day: pd.Timestamp, symbol: str) -> str:
    return (
        f"https://datasets.tardis.dev/v1/{exchange}/{data_type}/"
        f"{day.year:04d}/{day.month:02d}/{day.day:02d}/{symbol}.csv.gz"
    )


def _iter_tardis_csv_rows_streaming(url: str, api_key: str | None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with gzip.GzipFile(fileobj=response.raw) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text_stream:
                reader = csv.DictReader(text_stream)
                for row in reader:
                    yield row


def _iter_parquet_chunks(path: Path, chunk_size: int = 500_000):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        yield pl.from_arrow(batch)


def _online_finalize_bucket_row(
    symbol: str,
    bucket: pd.Timestamp,
    state: _OnlineResampleState,
    last_cols: list[str],
    summed_cols: list[str],
    include_stale: bool,
) -> dict[str, object]:
    row_out: dict[str, object] = {"timestamp": bucket, "symbol": symbol}

    for c in summed_cols:
        row_out[c] = state.sum_vals.get(c)

    for c in last_cols:
        val = state.last_vals.get(c)
        if val is None:
            val = state.last_emitted_last_vals.get(c)
        row_out[c] = val

    if include_stale:
        row_out["stale"] = (not state.bid_updated) and (not state.ask_updated)

    state.last_emitted_bucket = bucket
    state.last_emitted_last_vals = {c: row_out.get(c) for c in last_cols}
    return row_out


def download_and_convert_streaming_resample(
    exchange: str,
    data_type: str,
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: str = ".",
    force_reload: bool = False,
    resample_freq: str = "1min",
) -> List[Path]:
    """
    Stream Tardis CSV directly over HTTP and resample online per symbol bucket.

    This function avoids persisting raw CSV files and performs bucket aggregation
    as rows are being downloaded. It assumes rows are timestamp-ordered.
    """
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if end < start:
        raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")

    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    offset = pd.tseries.frequencies.to_offset(resample_freq)
    api_key = os.environ.get("TARDIS_API_KEY", None)
    parquet_paths: list[Path] = []

    for day in pd.date_range(start, end, freq="1D"):
        ds = str(day.date())
        stem = f"{exchange}_{data_type}_{ds}_{symbol}"
        parquet_path = data_dir_path / f"{stem}_{resample_freq}.parquet"
        parquet_paths.append(parquet_path)

        if parquet_path.exists() and not force_reload:
            logger.debug("Parquet already exists and force_reload=False, skipping: %s", parquet_path)
            continue
        if parquet_path.exists() and force_reload:
            parquet_path.unlink(missing_ok=True)

        url = _tardis_csv_url(exchange=exchange, data_type=data_type, day=day, symbol=symbol)
        logger.info("Streaming download+resample from %s", url)

        emitted_rows: list[dict[str, object]] = []
        per_symbol_state: dict[str, _OnlineResampleState] = {}
        last_cols: list[str] | None = None
        summed_cols: list[str] | None = None
        include_stale = False

        try:
            for row in _iter_tardis_csv_rows_streaming(url, api_key=api_key):
                if last_cols is None or summed_cols is None:
                    cols = list(row.keys())
                    _, last_cols, summed_cols = _resample_column_sets(cols, tscol="timestamp")
                    include_stale = ("bid_price" in cols) or ("ask_price" in cols)

                ts_raw = row.get("timestamp")
                sym = row.get("symbol")
                if not ts_raw or not sym:
                    continue

                ts = _parse_stream_timestamp(ts_raw)
                bucket = ts.floor(resample_freq)

                state = per_symbol_state.setdefault(sym, _OnlineResampleState())
                if state.current_bucket is None:
                    state.current_bucket = bucket
                    state.sum_vals = {}
                    state.last_vals = {}
                    state.bid_updated = False
                    state.ask_updated = False

                if bucket < state.current_bucket:
                    logger.warning(
                        "Out-of-order row encountered for %s at %s (< %s); skipping row",
                        sym,
                        bucket,
                        state.current_bucket,
                    )
                    continue

                if bucket > state.current_bucket:
                    # Finalize current bucket.
                    emitted_rows.append(
                        _online_finalize_bucket_row(
                            symbol=sym,
                            bucket=state.current_bucket,
                            state=state,
                            last_cols=last_cols,
                            summed_cols=summed_cols,
                            include_stale=include_stale,
                        )
                    )

                    # Emit empty gap buckets with forward-filled "last" columns.
                    gap_bucket = state.current_bucket + offset
                    while gap_bucket < bucket:
                        gap_row = {"timestamp": gap_bucket, "symbol": sym}
                        for c in summed_cols:
                            gap_row[c] = None
                        for c in last_cols:
                            gap_row[c] = state.last_emitted_last_vals.get(c)
                        if include_stale:
                            gap_row["stale"] = True
                        emitted_rows.append(gap_row)
                        state.last_emitted_bucket = gap_bucket
                        gap_bucket = gap_bucket + offset

                    # Reset for next bucket.
                    state.current_bucket = bucket
                    state.sum_vals = {}
                    state.last_vals = {}
                    state.bid_updated = False
                    state.ask_updated = False

                # Accumulate this row into current bucket.
                for c in last_cols:
                    state.last_vals[c] = _parse_stream_value(c, row.get(c, ""))

                for c in summed_cols:
                    v = _parse_stream_value(c, row.get(c, ""))
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    state.sum_vals[c] = state.sum_vals.get(c, 0.0) + fv

                bid_val = _parse_stream_value("bid_price", row.get("bid_price", ""))
                ask_val = _parse_stream_value("ask_price", row.get("ask_price", ""))
                if bid_val is not None:
                    state.bid_updated = True
                if ask_val is not None:
                    state.ask_updated = True

            # Flush final open bucket per symbol.
            for sym, state in per_symbol_state.items():
                if state.current_bucket is None:
                    continue
                emitted_rows.append(
                    _online_finalize_bucket_row(
                        symbol=sym,
                        bucket=state.current_bucket,
                        state=state,
                        last_cols=last_cols or [],
                        summed_cols=summed_cols or [],
                        include_stale=include_stale,
                    )
                )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("No dataset found for %s (HTTP 404)", url)
                pl.DataFrame().write_parquet(parquet_path)
                continue
            raise

        if not emitted_rows:
            pl.DataFrame().write_parquet(parquet_path)
            logger.info("Wrote empty parquet %s", parquet_path)
            continue

        final_df = pl.DataFrame(emitted_rows, strict=False).sort(["symbol", "timestamp"])
        final_df = _cast_temporal_fallback_columns(final_df, TARDIS_COLUMN_TYPES)
        if final_df.schema.get("timestamp") != pl.Datetime(TIMESTAMP_UNIT):
            final_df = final_df.with_columns(pl.col("timestamp").cast(pl.Datetime(TIMESTAMP_UNIT)))
        final_df.write_parquet(parquet_path)
        logger.debug(f'wrote {parquet_path}')

        # Write compact metadata sidecar for parity with batch path.
        symbol_timestamp_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        grouped = final_df.group_by("symbol").agg([
            pl.col("timestamp").min().alias("min_ts"),
            pl.col("timestamp").max().alias("max_ts"),
        ])
        for r in grouped.iter_rows(named=True):
            symbol_timestamp_bounds[r["symbol"]] = (pd.Timestamp(r["min_ts"]), pd.Timestamp(r["max_ts"]))

        global_first = pd.Timestamp(final_df.select(pl.col("timestamp").min()).item())
        global_last = pd.Timestamp(final_df.select(pl.col("timestamp").max()).item())
        metadata = Metadata(
            symbols=sorted(symbol_timestamp_bounds.keys()),
            symbol_timestamp_bounds=symbol_timestamp_bounds,
            global_first_timestamp=global_first,
            global_last_timestamp=global_last,
            extras={"resample_freq": resample_freq, "streaming": True},
        )
        metadata_path = Path(str(parquet_path) + ".pkl")
        with metadata_path.open("wb") as handle:
            pickle.dump(metadata, handle)
            logger.debug(f'wrote {metadata_path}')

        logger.info("Finished streaming convert+resample %s", parquet_path)

    return parquet_paths


def _resample_column_sets(columns: list[str], tscol: str = "timestamp") -> tuple[list[str], list[str], list[str]]:
    value_cols = [c for c in columns if c not in {tscol, "symbol", "_bid_updated", "_ask_updated", "stale"}]
    summed_cols = [c for c in value_cols if _aggregation_for_column(c) == "sum"]
    last_cols = [c for c in value_cols if c not in summed_cols]
    return value_cols, last_cols, summed_cols


def _aggregate_resample_chunk(df: pl.DataFrame, freq: str, tscol: str = "timestamp") -> pl.DataFrame:
    if df.is_empty():
        return df

    polars_freq = _normalize_polars_freq(freq)
    _, last_cols, summed_cols = _resample_column_sets(df.columns, tscol)

    # Single-pass bucket aggregation: preserves last/sum semantics and computes
    # quote-update flags without an extra groupby+join.
    # Use sort_by(tscol).last() per bucket to avoid a global pre-sort.
    return (
        df.with_columns(pl.col(tscol).dt.truncate(polars_freq).alias("_bucket_ts"))
        .group_by(["symbol", "_bucket_ts"], maintain_order=True)
        .agg(
            [
                *[pl.col(c).sort_by(tscol).last().alias(c) for c in last_cols],
                *[pl.col(c).sum().alias(c) for c in summed_cols],
                (pl.col("bid_price").is_not_null().any() if "bid_price" in df.columns else pl.lit(False)).alias("_bid_updated"),
                (pl.col("ask_price").is_not_null().any() if "ask_price" in df.columns else pl.lit(False)).alias("_ask_updated"),
            ]
        )
        .rename({"_bucket_ts": tscol})
        .sort(["symbol", tscol])
    )


def _aggregate_resample_trade_chunk(df: pl.DataFrame, freq: str, tscol: str = "timestamp") -> pl.DataFrame:
    """Aggregate a raw trade DataFrame into per-(symbol, bucket) sums.

    Output columns per bucket:
      trade_amount        – total traded quantity
      trade_signed_amount – buy quantity minus sell quantity  (side == "buy" → +amount, else −amount)
      trade_price_amount  – sum(price × amount)               (divide by amount to recover VWAP)
    """
    if df.is_empty():
        return df

    polars_freq = _normalize_polars_freq(freq)

    # Build signed_amount if not already present.
    if "trade_signed_amount" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("side") == "buy")
            .then(pl.col("amount"))
            .otherwise(-pl.col("amount"))
            .alias("trade_signed_amount")
        )

    # price × amount for VWAP recovery.
    df = df.with_columns(
        (pl.col("price") * pl.col("amount")).alias("trade_price_amount")
    )

    return (
        df.sort([tscol])
        .with_columns(pl.col(tscol).dt.truncate(polars_freq).alias("_bucket_ts"))
        .group_by(["symbol", "_bucket_ts"], maintain_order=True)
        .agg([
            pl.col("amount").sum().alias("trade_amount"),
            pl.col("trade_signed_amount").sum().alias("trade_signed_amount"),
            pl.col("trade_price_amount").sum().alias("trade_price_amount"),
        ])
        .rename({"_bucket_ts": tscol})
        .sort(["symbol", tscol])
    )


def _expand_resampled_buckets(df: pl.DataFrame, freq: str, tscol: str = "timestamp") -> pl.DataFrame:
    if df.is_empty():
        return df

    value_cols, last_cols, summed_cols = _resample_column_sets(df.columns, tscol)
    symbols = df.select("symbol").unique().sort("symbol").get_column("symbol").to_list()
    frames = []

    for symbol in symbols:
        sdf = df.filter(pl.col("symbol") == symbol).sort(tscol)
        t_start = pd.Timestamp(sdf.select(pl.col(tscol).min()).item()).floor(freq)
        t_end = pd.Timestamp(sdf.select(pl.col(tscol).max()).item()).ceil(freq)
        grid = pl.Series(pd.date_range(t_start, t_end, freq=freq)).cast(pl.Datetime(TIMESTAMP_UNIT))
        filled = pl.DataFrame({tscol: grid}).join(sdf, on=tscol, how="left")

        # Only compute stale when quote update markers are available.
        has_bid_updated = "_bid_updated" in filled.columns
        has_ask_updated = "_ask_updated" in filled.columns
        if has_bid_updated and has_ask_updated:
            filled = filled.with_columns(
                (
                    (~pl.col("_bid_updated").fill_null(False))
                    & (~pl.col("_ask_updated").fill_null(False))
                ).alias("stale")
            )

        drop_update_cols = [c for c in ["_bid_updated", "_ask_updated"] if c in filled.columns]
        if drop_update_cols:
            filled = filled.drop(drop_update_cols)

        if last_cols:
            filled = filled.with_columns([pl.col(c).fill_null(strategy="forward").alias(c) for c in last_cols])

        filled = filled.with_columns(pl.lit(symbol).alias("symbol"))
        select_cols = [tscol, "symbol"] + value_cols
        if "stale" in filled.columns:
            select_cols.append("stale")
        frames.append(filled.select(select_cols))

    return pl.concat(frames, rechunk=False).sort(["symbol", tscol])


def _merge_resample_aggregates(df: pl.DataFrame, tscol: str = "timestamp") -> pl.DataFrame:
    if df.is_empty():
        return df

    _, last_cols, summed_cols = _resample_column_sets(df.columns, tscol)
    agg_exprs = [
        *[pl.col(c).last().alias(c) for c in last_cols],
        *[pl.col(c).sum().alias(c) for c in summed_cols],
    ]
    if "_bid_updated" in df.columns:
        agg_exprs.append(pl.col("_bid_updated").any().alias("_bid_updated"))
    if "_ask_updated" in df.columns:
        agg_exprs.append(pl.col("_ask_updated").any().alias("_ask_updated"))

    return (
        df.group_by(["symbol", tscol], maintain_order=True)
        .agg(agg_exprs)
        .sort(["symbol", tscol])
    )



def _convert_csv_to_parquet(
    data_type,
    exchange,
    mdy,
    csv_path: Path,
    parquet_path: Path,
    block_size: int = 32 * 1024 * 1024,
    force_reload: bool = False,
    resample_freq: str | None = None,
) -> Path:
    if parquet_path.exists() and not force_reload:
        logger.debug("Parquet already exists: %s", parquet_path)
        return parquet_path

    if parquet_path.exists() and force_reload:
        parquet_path.unlink(missing_ok=True)

    logger.debug("Converting %s -> %s", csv_path, parquet_path)
    mdy_date = pd.Timestamp(mdy).date()

    read_options = pv.ReadOptions(block_size=block_size)
    default_convert_options = pv.ConvertOptions(
        column_types=TARDIS_COLUMN_TYPES,
        null_values=["", "null", "None", "NULL", "NaN", "nan"],
    )

    def _open_gz(path: Path):
        return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")

    timestamps_read_as_int = False
    gz_f = _open_gz(csv_path)
    try:
        reader = pv.open_csv(
            gz_f,
            read_options=read_options,
            convert_options=default_convert_options,
        )
    except pa.ArrowInvalid:
        gz_f.close()
        fallback_types = _temporal_fallback_column_types(TARDIS_COLUMN_TYPES)
        fallback_convert_options = pv.ConvertOptions(
            column_types=fallback_types,
            null_values=["", "null", "None", "NULL", "NaN", "nan"],
        )
        gz_f = _open_gz(csv_path)
        reader = pv.open_csv(
            gz_f,
            read_options=read_options,
            convert_options=fallback_convert_options,
        )
        timestamps_read_as_int = True

    writer = None
    metadata_path = Path(str(parquet_path) + ".pkl")
    symbol_timestamp_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    global_first_timestamp: pd.Timestamp | None = None
    global_last_timestamp: pd.Timestamp | None = None
    resample_chunks: list[pl.DataFrame] = []

    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])

            if timestamps_read_as_int:
                table_df = pl.from_arrow(table)
                table_df = _cast_temporal_fallback_columns(table_df, TARDIS_COLUMN_TYPES)
                table = table_df.to_arrow()

            if False:
                table_df = pl.from_arrow(table)
                table_df = table_df.with_columns(
                    [
                        pl.lit(exchange).cast(pl.Utf8).alias("exchange"),
                        pl.lit(data_type).cast(pl.Utf8).alias("data_type"),
                        pl.lit(mdy_date).cast(pl.Date).alias("mdy"),
                    ]
                )
                table = table_df.to_arrow()

            if resample_freq:
                logger.debug("_convert_csv_to_parquet resample chunk rows=%d freq=%s", table_df.height, resample_freq)
                if data_type == 'trades':
                    resample_chunks.append(_aggregate_resample_trade_chunk(table_df, freq=resample_freq, tscol="timestamp"))
                else:
                    resample_chunks.append(_aggregate_resample_chunk(table_df, freq=resample_freq, tscol="timestamp"))
            else:
                if writer is None:
                    writer = pq.ParquetWriter(str(parquet_path), table.schema, compression="snappy")
                writer.write_table(table)
                logger.debug(f'wrote chunk')

            if "symbol" in table.schema.names and "timestamp" in table.schema.names:
                batch_ranges = (
                    table_df
                    .select(["symbol", "timestamp"])
                    .drop_nulls()
                    .group_by("symbol")
                    .agg(
                        [
                            pl.col("timestamp").min().alias("min_ts"),
                            pl.col("timestamp").max().alias("max_ts"),
                        ]
                    )
                )

                for row in batch_ranges.iter_rows(named=True):
                    symbol = row["symbol"]
                    min_ts = pd.Timestamp(row["min_ts"])
                    max_ts = pd.Timestamp(row["max_ts"])

                    prev = symbol_timestamp_bounds.get(symbol)
                    if prev is None:
                        symbol_timestamp_bounds[symbol] = (min_ts, max_ts)
                    else:
                        symbol_timestamp_bounds[symbol] = (min(prev[0], min_ts), max(prev[1], max_ts))

                    if global_first_timestamp is None:
                        global_first_timestamp = min_ts
                    else:
                        global_first_timestamp = min(global_first_timestamp, min_ts)

                    if global_last_timestamp is None:
                        global_last_timestamp = max_ts
                    else:
                        global_last_timestamp = max(global_last_timestamp, max_ts)
    finally:
        gz_f.close()
        if writer is not None:
            writer.close()

    if resample_freq:
        if resample_chunks:
            logger.debug("Resampling converted CSV %s at freq=%s before parquet write", csv_path, resample_freq)
            combined = pl.concat(resample_chunks, rechunk=False)
            combined = _merge_resample_aggregates(combined, tscol="timestamp")
            final_df = _expand_resampled_buckets(combined, freq=resample_freq, tscol="timestamp")
            final_df.write_parquet(parquet_path)
            logger.debug(f'wrote {parquet_path} (resample_chunks)')
        else:
            pl.DataFrame().write_parquet(parquet_path)
            logger.debug(f'wrote {parquet_path}')

    metadata = Metadata(
        symbols=sorted(symbol_timestamp_bounds.keys()),
        symbol_timestamp_bounds=symbol_timestamp_bounds,
        global_first_timestamp=global_first_timestamp,
        global_last_timestamp=global_last_timestamp,
        extras={"resample_freq": resample_freq} if resample_freq else {},
    )
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)
        logger.debug(f'wrote {metadata_path}')

    logger.info("Finished converting %s", parquet_path)
    logger.debug("Wrote parquet metadata %s", metadata_path)
    return parquet_path


def download_and_convert(
    exchange: str,
    data_type: str,
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: str = ".",
    force_reload: bool = False,
    cleanup_csv: bool = True,
    resample_freq: str | None = None,
) -> List[Path]:
    """
    Download and convert Tardis CSV files for an arbitrary (exchange, data_type, symbol) tuple.

    File naming convention:
    - CSV:     {exchange}_{datatype}_{date}_{symbol}.csv.gz
    - Parquet: {exchange}_{datatype}_{date}_{symbol}.parquet
    - Resampled parquet: {exchange}_{datatype}_{date}_{symbol}_{freq}.parquet

    Parameters
    ----------
    exchange : str
        Exchange name, e.g. "binance", "okex", "deribit".
    data_type : str
        Tardis data type, e.g. "trades", "quotes", "options_chain".
    symbol : str
        Instrument symbol in exchange notation.
    start_date : str
        Inclusive start date (YYYY-MM-DD).
    end_date : str
        Inclusive end date (YYYY-MM-DD).
    data_dir : str
        Directory where files are downloaded and stored.
    force_reload : bool
        If True, conversion is forced even when target parquet file already exists.
    cleanup_csv : bool
        If True, remove source CSV files after successful conversion. Defaults to True.
    resample_freq : str | None
        If provided, bucket and forward-fill each CSV to this frequency before writing parquet.

    Returns
    -------
    List[Path]
        Paths to created (or pre-existing) parquet files.
    """
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if end < start:
        raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")

    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    parquet_paths: List[Path] = []
    missing_or_forced_days = []
    for day in pd.date_range(start, end, freq="1D"):
        ds = str(day.date())
        stem = f"{exchange}_{data_type}_{ds}_{symbol}"
        parquet_name = f"{stem}_{resample_freq}.parquet" if resample_freq else f"{stem}.parquet"
        parquet_path = data_dir_path / parquet_name
        parquet_paths.append(parquet_path)

        if force_reload or (not parquet_path.exists()):
            missing_or_forced_days.append(day)

    if missing_or_forced_days:
        download_to = str(end + pd.Timedelta(days=1))
        datasets.download(
            exchange=exchange,
            data_types=[data_type],
            from_date=str(start),
            to_date=download_to,
            symbols=[symbol],
            download_dir=str(data_dir_path),
            api_key=os.environ.get('TARDIS_API_KEY', None),
        )
        logger.info(
            "Downloaded %s/%s/%s from %s to %s into %s",
            exchange,
            data_type,
            symbol,
            start_date,
            end_date,
            data_dir,
        )


    for day in pd.date_range(start, end, freq="1D"):
        ds = str(day.date())
        stem = f"{exchange}_{data_type}_{ds}_{symbol}"
        csv_path = data_dir_path / f"{stem}.csv.gz"
        parquet_name = f"{stem}_{resample_freq}.parquet" if resample_freq else f"{stem}.parquet"
        parquet_path = data_dir_path / parquet_name

        if parquet_path.exists() and not force_reload:
            logger.debug("Parquet already exists and force_reload=False, skipping convert: %s", parquet_path)
            continue

        if not csv_path.exists():
            logger.warning("Expected CSV not found (skip): %s", csv_path)
            continue

        _convert_csv_to_parquet(
            data_type,
            exchange,
            ds,
            csv_path,
            parquet_path,
            force_reload=force_reload,
            resample_freq=resample_freq,

        )

        if cleanup_csv:
            csv_path.unlink(missing_ok=True)

    return parquet_paths

def testme():
    paths = download_and_convert(
        exchange="deribit",
        data_type="options_chain",
        symbol="OPTIONS",
        start_date="2026-03-13",
        end_date="2026-03-13",
        data_dir="datasets/deribit_chain",
        force_reload=True,
        cleanup_csv=False,
        resample_freq='5min',
    )
    print(paths)


def resample(input_parquet_file, freq='1min', to=None, output_cols=[], summed_cols=[],
              output_tag='', force_reload=False, cleanup_intermediate_parquet=False):
    '''
    merge_data only deals with a single day (fr), the to argument only serves to
    cut the day sort for testing
    '''
    input_path = Path(input_parquet_file)
    polars_freq = _normalize_polars_freq(freq)
    invalid_summed_cols = [c for c in summed_cols if c not in output_cols]
    if invalid_summed_cols:
        raise ValueError(f"summed_cols must be a subset of output_cols, invalid: {invalid_summed_cols}")
    last_cols = [c for c in output_cols if c not in summed_cols]

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    output_path = input_path.with_name(f"{input_path.stem}_{output_tag}{freq}.parquet")
    if not force_reload and os.path.exists(output_path):
        return pl.read_parquet(output_path)

    tscol = 'timestamp'
    chunksize = 1_500_000
    chunks = []
    for chunk in _iter_parquet_chunks(input_path, chunk_size=chunksize):
        assert tscol in chunk.columns, f"Column '{tscol}' not found in chunk"
        assert 'symbol' in chunk.columns, f"Column 'symbol' not found in chunk"
        if chunk.schema[tscol] in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            chunk = chunk.with_columns(pl.from_epoch(pl.col(tscol), time_unit=TIMESTAMP_UNIT).alias(tscol))
        else:
            chunk = chunk.with_columns(pl.col(tscol).cast(pl.Datetime(TIMESTAMP_UNIT)))
        chunks.append(chunk)

    if not chunks:
        logger.debug("No data to write.")
        return

    df = pl.concat(chunks)
    if to is not None:
        df = df.filter(pl.col(tscol) <= pd.Timestamp(to))

    symbol_frames = []
    for symbol, sdf in df.partition_by('symbol', as_dict=True).items():
        sym = symbol[0] if isinstance(symbol, tuple) else symbol
        ts_min = pd.Timestamp(sdf.select(tscol).min().item())
        ts_max = pd.Timestamp(sdf.select(tscol).max().item())
        t_start = ts_min.floor(freq)
        t_end = ts_max.floor(freq)

        grid_df = pl.DataFrame(
            {tscol: pl.Series(pd.date_range(t_start, t_end, freq=freq)).cast(pl.Datetime(TIMESTAMP_UNIT))}
        )
        res = grid_df

        bucketed = sdf.with_columns(pl.col(tscol).dt.truncate(polars_freq).alias('_bucket_ts'))

        if last_cols:
            df_last = (
                bucketed.group_by('_bucket_ts')
                .agg([pl.col(c).last().alias(c) for c in last_cols])
                .rename({'_bucket_ts': tscol})
                .sort(tscol)
            )
            res = res.join(df_last, on=tscol, how='left')
            res = res.with_columns([pl.col(c).fill_null(strategy='forward').alias(c) for c in last_cols])

        if summed_cols:
            df_sum = (
                bucketed.group_by('_bucket_ts')
                .agg([pl.col(c).sum().alias(c) for c in summed_cols])
                .rename({'_bucket_ts': tscol})
                .sort(tscol)
            )
            res = res.join(df_sum, on=tscol, how='left')

        missing_cols = [c for c in output_cols if c not in res.columns]
        if missing_cols:
            res = res.with_columns([pl.lit(None, dtype=TARDIS_COLUMN_TYPES.get(c, pl.Float64)).alias(c) for c in missing_cols])

        res = res.with_columns(pl.lit(sym).alias('symbol'))
        res = res.select([tscol, 'symbol'] + output_cols)
        symbol_frames.append(res)

    if not symbol_frames:
        logger.debug("No data to write.")
        return

    final_df = pl.concat(symbol_frames)
    final_df = final_df.unique(subset=[tscol, 'symbol'], keep='last').sort(['symbol', tscol])

    if cleanup_intermediate_parquet:
        input_path.unlink(missing_ok=True)

    final_df.write_parquet(output_path)
    logger.debug(f"Wrote {len(final_df)} total rows to {output_path}")
    return final_df

