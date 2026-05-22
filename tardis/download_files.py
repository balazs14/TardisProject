import argparse
import logging
import os
import resource

from tardis import _CliHelpFormatter
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as pv
import requests
import sys
from tardis.utils import _configure_logging


logger = logging.getLogger(__name__)
DEFAULT_RESAMPLE_FREQ = "5min"

# Use microsecond resolution for all timestamps
TIMESTAMP_UNIT = 'us'


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
        "strike_price": pa.float64(),
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
    for level in range(101):
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

        types[f"bids[{level}].price"] = pa.float64()
        types[f"bids[{level}].amount"] = pa.float64()
        types[f"asks[{level}].price"] = pa.float64()
        types[f"asks[{level}].amount"] = pa.float64()

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


def _tardis_csv_url(exchange: str, data_type: str, day: pd.Timestamp, symbol: str) -> str:
    return (
        f"https://datasets.tardis.dev/v1/{exchange}/{data_type}/"
        f"{day.year:04d}/{day.month:02d}/{day.day:02d}/{symbol}.csv.gz"
    )


def _iter_tardis_csv_rows_streaming(
    url: str,
    api_key: str | None,
    chunk_size: int = 100_000,
    block_size: int = 8 * 1024 * 1024,
):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    block_size = max(block_size, chunk_size * 256)
    read_options = pv.ReadOptions(block_size=block_size)
    # The HTTP+gzip stream is not seekable, so we cannot safely "retry" parsing
    # from the start after a conversion failure. Parse temporal columns as int64
    # up front, then cast to timestamps/dates in Polars below.
    convert_options = pv.ConvertOptions(
        column_types=_temporal_fallback_column_types(TARDIS_COLUMN_TYPES),
        null_values=["", "null", "None", "NULL", "NaN", "nan"],
    )

    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with gzip.GzipFile(fileobj=response.raw) as gz:
            reader = pv.open_csv(gz, read_options=read_options, convert_options=convert_options)
            try:
                for batch in reader:
                    df = pl.from_arrow(pa.Table.from_batches([batch]))
                    df = _cast_temporal_fallback_columns(df, TARDIS_COLUMN_TYPES)
                    logger.debug(f"Processing batch {df}")
                    yield df
            finally:
                reader.close()


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


def _normalize_symbols_input(symbols) -> list[str]:
    """Normalize symbols input into a non-empty list of valid symbol strings.

    Invalid symbol entries do not raise; they are skipped with logger.warning.
    """
    if isinstance(symbols, str):
        candidates = [symbols]
    elif isinstance(symbols, (list, tuple, set)):
        candidates = list(symbols)
    else:
        logger.warning("Invalid symbols input type=%s; expected str or list-like", type(symbols).__name__)
        return []

    out: list[str] = []
    for s in candidates:
        if not isinstance(s, str):
            logger.warning("Skipping invalid symbol %r (type=%s)", s, type(s).__name__)
            continue
        s_clean = s.strip()
        if not s_clean:
            logger.warning("Skipping invalid empty symbol entry: %r", s)
            continue
        out.append(s_clean)
    return out


def _is_missing_dataset_http_error(exc: requests.HTTPError) -> bool:
    """Return True when an HTTP error indicates the dataset file is unavailable."""
    if exc.response is None:
        return False
    return exc.response.status_code in {400, 404}


def _download_url_to_file(url: str, destination: Path, api_key: str | None) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_raw_csv_gz(
    exchange: str,
    data_type: str,
    symbols,
    start_date: str,
    end_date: str | None = None,
    data_dir: str | None = None,
    force_reload: bool = False,
) -> List[Path]:
    """Download raw Tardis .csv.gz files without any conversion or resampling."""
    end_date = end_date or start_date
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if end < start:
        raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")

    data_dir_path = Path(data_dir) if data_dir is not None else Path(f"datasets/{exchange}_raw")
    data_dir_path.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("TARDIS_API_KEY", None)
    symbol_list = _normalize_symbols_input(symbols)
    raw_paths: list[Path] = []

    if not symbol_list:
        logger.warning("No valid symbols provided for raw download(exchange=%s, data_type=%s)", exchange, data_type)
        return raw_paths

    for symbol in symbol_list:
        for day in pd.date_range(start, end, freq="1D"):
            ds = str(day.date())
            stem = f"{exchange}_{data_type}_{ds}_{symbol}"
            csv_gz_path = data_dir_path / f"{stem}.csv.gz"
            raw_paths.append(csv_gz_path)

            if csv_gz_path.exists() and not force_reload:
                logger.debug("Raw CSV already exists and force_reload=False, skipping: %s", csv_gz_path)
                continue
            if csv_gz_path.exists() and force_reload:
                csv_gz_path.unlink(missing_ok=True)

            url = _tardis_csv_url(exchange=exchange, data_type=data_type, day=day, symbol=symbol)
            logger.info("Downloading raw csv.gz from %s", url)

            try:
                _download_url_to_file(url, csv_gz_path, api_key=api_key)
            except requests.HTTPError as exc:
                if _is_missing_dataset_http_error(exc):
                    status = exc.response.status_code if exc.response is not None else "unknown"
                    logger.warning("No dataset found for symbol=%s (%s) [HTTP %s]", symbol, url, status)
                    csv_gz_path.unlink(missing_ok=True)
                    continue
                raise

            logger.info("Finished raw download %s", csv_gz_path)

    return raw_paths


def download_resample(
    exchange: str,
    data_type: str,
    symbols,
    start_date: str,
    end_date: str | None = None,
    data_dir: str|None = None,
    force_reload: bool = False,
    resample_freq: str = DEFAULT_RESAMPLE_FREQ,
) -> List[Path]:
    """
    Stream Tardis CSV directly over HTTP and resample online per symbol bucket.

    This function avoids persisting raw CSV files and performs bucket aggregation
    as rows are being downloaded. It assumes rows are timestamp-ordered.
    """
    end_date = end_date or start_date
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if end < start:
        raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")

    data_dir_path = Path(data_dir) if data_dir is not None else Path(f"datasets/{exchange}_raw")  
    data_dir_path.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("TARDIS_API_KEY", None)
    symbol_list = _normalize_symbols_input(symbols)
    parquet_paths: list[Path] = []

    if not symbol_list:
        logger.warning("No valid symbols provided for download_resample(exchange=%s, data_type=%s)", exchange, data_type)
        return parquet_paths

    for symbol in symbol_list:
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

            try:
                resample_chunks: list[pl.DataFrame] = []
                chunk_idx = 0
                COMPACT_EVERY = 10  # merge accumulated chunks every N raw chunks to bound memory
                for chunk_df in _iter_tardis_csv_rows_streaming(url, api_key=api_key):
                    logger.debug(f"Processing chunk {chunk_df}")
                    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    last_ts = chunk_df["timestamp"][-1] if "timestamp" in chunk_df.columns else None
                    logger.debug(
                        "chunk %d: raw rows=%d cols=%d  last_ts=%s  rss=%.1f MB",
                        chunk_idx, chunk_df.height, chunk_df.width, last_ts, rss_mb,
                    )
                    pldf = _cast_temporal_fallback_columns(chunk_df, TARDIS_COLUMN_TYPES)
                    if pldf.schema.get("timestamp") != pl.Datetime(TIMESTAMP_UNIT):
                        pldf = pldf.with_columns(pl.col("timestamp").cast(pl.Datetime(TIMESTAMP_UNIT)))
                    logger.debug("Aggregating chunk for data_type=%s chunk=%d", data_type, chunk_idx)
                    agg = _aggregate_resample_chunk(
                        data_type=data_type,
                        df=pldf,
                        freq=resample_freq,
                        tscol="timestamp",
                    )
                    if not agg.is_empty():
                        resample_chunks.append(agg)
                    rss_mb_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    logger.debug(
                        "chunk %d: agg rows=%d  chunks_so_far=%d  rss=%.1f MB (+%.1f MB)",
                        chunk_idx, agg.height, len(resample_chunks), rss_mb_after, rss_mb_after - rss_mb,
                    )
                    # Periodically compact accumulated chunks to bound memory growth.
                    if len(resample_chunks) >= COMPACT_EVERY:
                        merged = _merge_resample_aggregates(
                            pl.concat(resample_chunks, rechunk=False), tscol="timestamp"
                        )
                        merged_last_ts = (
                            merged.select(pl.col("timestamp").max()).item()
                            if not merged.is_empty()
                            else None
                        )
                        logger.info("merged(last compact) timestamp=%s", merged_last_ts)
                        rss_mb_compact = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                        logger.debug(
                            "chunk %d: compacted %d chunks → %d rows  rss=%.1f MB",
                            chunk_idx, len(resample_chunks), merged.height, rss_mb_compact,
                        )
                        resample_chunks = [merged]
                    chunk_idx += 1
            except requests.HTTPError as exc:
                if _is_missing_dataset_http_error(exc):
                    status = exc.response.status_code if exc.response is not None else "unknown"
                    logger.warning("No dataset found for symbol=%s (%s) [HTTP %s]", symbol, url, status)
                    pl.DataFrame().write_parquet(parquet_path)
                    continue
                raise

            if not resample_chunks:
                pl.DataFrame().write_parquet(parquet_path)
                logger.info("Wrote empty parquet %s", parquet_path)
                continue

            combined = pl.concat(resample_chunks, rechunk=False)
            logger.debug(
                "post-concat: %d chunks → %d rows  rss=%.1f MB",
                len(resample_chunks), combined.height,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
            del resample_chunks
            combined = _merge_resample_aggregates(combined, tscol="timestamp")
            logger.debug(
                "post-merge: %d rows  rss=%.1f MB",
                combined.height, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
            final_df = _expand_resampled_buckets(combined, freq=resample_freq, tscol="timestamp")
            del combined
            logger.debug(
                "post-expand: %d rows  rss=%.1f MB",
                final_df.height, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
            # Sort columns alphabetically to match batch output
            final_df = final_df.select(sorted(final_df.columns))
            final_df.write_parquet(parquet_path)
            logger.debug(f'wrote {parquet_path}')

            logger.info("Finished streaming convert+resample %s", parquet_path)

    return parquet_paths


def peek_streaming_raw(
    exchange: str,
    data_type: str,
    symbol: str,
    start_date: str,
    end_date: str | None = None,
    *,
    head_rows: int = 10,
    transpose_preview: bool = True,
) -> None:
    end_date = end_date or start_date
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if end < start:
        raise ValueError(f"end_date ({end_date}) must be >= start_date ({start_date})")

    api_key = os.environ.get("TARDIS_API_KEY", None)
    for day in pd.date_range(start, end, freq="1D"):
        url = _tardis_csv_url(exchange=exchange, data_type=data_type, day=day, symbol=symbol)
        logger.info("Peek streaming from %s", url)
        try:
            iterator = _iter_tardis_csv_rows_streaming(url, api_key=api_key)
            first_chunk = next(iterator, None)
            if first_chunk is None or first_chunk.is_empty():
                logger.info("peek: no rows for %s", day.date())
                continue

            logger.info("peek: day=%s rows_in_first_chunk=%d cols=%d", day.date(), first_chunk.height, first_chunk.width)
            logger.info("peek columns: %s", first_chunk.columns)
            logger.info("peek inferred schema: %s", first_chunk.schema)
            logger.info(
                "peek preview (%s, first %d rows):\n%s",
                "transposed" if transpose_preview else "table",
                head_rows,
                _format_peek_preview(first_chunk, head_rows, transpose_preview=transpose_preview),
            )
            return
        except requests.HTTPError as exc:
            if _is_missing_dataset_http_error(exc):
                status = exc.response.status_code if exc.response is not None else "unknown"
                logger.warning("peek: no dataset found for %s (HTTP %s)", url, status)
                continue
            raise

    logger.warning("peek: no available data found in requested range")


def _format_peek_preview(df: pl.DataFrame, head_rows: int, *, transpose_preview: bool = True) -> str:
    preview = df.head(head_rows).to_pandas()
    if transpose_preview:
        preview = preview.transpose()
    return preview.to_string(max_rows=None, max_cols=None)


def _resample_column_sets(columns: list[str], tscol: str = "timestamp") -> tuple[list[str], list[str], list[str]]:
    value_cols = [c for c in columns if c not in {tscol, "symbol", "_bid_updated", "_ask_updated", "stale"}]
    summed_cols = [c for c in value_cols if _aggregation_for_column(c) == "sum"]
    last_cols = [c for c in value_cols if c not in summed_cols]
    return value_cols, last_cols, summed_cols


def _aggregate_resample_default_chunk(df: pl.DataFrame, freq: str, tscol: str = "timestamp") -> pl.DataFrame:
    if df.is_empty():
        return df
    polars_freq = _normalize_polars_freq(freq)
    _, last_cols, summed_cols = _resample_column_sets(df.columns, tscol)
    agg_exprs = [
        *[pl.col(c).sort_by(tscol).last().alias(c) for c in last_cols],
        *[pl.col(c).sum().alias(c) for c in summed_cols],
    ]
    if "bid_price" in df.columns:
        agg_exprs.append(pl.col("bid_price").is_not_null().any().alias("_bid_updated"))
    if "ask_price" in df.columns:
        agg_exprs.append(pl.col("ask_price").is_not_null().any().alias("_ask_updated"))

    # Single-pass bucket aggregation: preserves last/sum semantics and computes
    # quote-update flags without an extra groupby+join.
    # Use sort_by(tscol).last() per bucket to avoid a global pre-sort.
    return (
        df.with_columns(pl.col(tscol).dt.truncate(polars_freq).alias("_bucket_ts"))
        .group_by(["symbol", "_bucket_ts"], maintain_order=True)
        .agg(agg_exprs)
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


def _get_chunk_aggregator(data_type: str):
    """Return the chunk aggregation function for a Tardis data type.

    Keep the dispatch table explicit so future data types with custom
    accumulator logic remain easy to add and review.
    """
    aggregators = {
        "trades": _aggregate_resample_trade_chunk,
    }
    return aggregators.get(data_type, _aggregate_resample_default_chunk)


def _aggregate_resample_chunk(
    data_type: str,
    df: pl.DataFrame,
    freq: str,
    tscol: str = "timestamp",
) -> pl.DataFrame:
    """Aggregate one streamed chunk using the strategy for ``data_type``.

    This is the single dispatch entry point used by the streaming pipeline.
    Data types with custom stateful or accumulator-based logic should provide
    their own concrete aggregator and be registered in ``_get_chunk_aggregator``.
    """
    aggregator = _get_chunk_aggregator(data_type)
    logger.debug("Using chunk aggregator %s for data_type=%s", aggregator.__name__, data_type)
    return aggregator(df, freq=freq, tscol=tscol)


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

def testme():
    paths = download_resample(
        exchange="deribit",
        data_type="trades",
        symbols="OPTIONS",
        start_date="2026-03-26",
        end_date="2026-03-26",
        data_dir="datasets/deribit",
        force_reload=True,
        resample_freq='5min',
    )
    
    return paths


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Tardis CSV data and convert it to parquet.",
        formatter_class=_CliHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--exchange",
        required=True,
        help="Exchange name (e.g. deribit, okex, binance)",
    )
    parser.add_argument(
        "--data-type",
        required=True,
        dest="data_type",
        help="Tardis data type (e.g. trades, quotes, options_chain)",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Instrument symbol in exchange notation",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        dest="start_date",
        help="Inclusive start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        dest="end_date",
        help="Inclusive end date (YYYY-MM-DD). Defaults to --start-date when omitted.",  # keep explicit: None default would be suppressed
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory where files are downloaded and stored (default: current directory)",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force conversion even if target parquet already exists",
    )
    parser.add_argument(
        "--resample-freq",
        default=DEFAULT_RESAMPLE_FREQ,
        help="Resample frequency before parquet write (e.g. 1min, 5min)",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Stream one raw chunk and print head + inferred schema, then exit",
    )
    parser.add_argument(
        "--peek-rows",
        type=int,
        default=10,
        help="Number of rows to print in --peek mode",
    )
    parser.add_argument(
        "--peek-transposed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display --peek preview transposed so all columns are visible",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Download and save raw .csv.gz files only (no resample/parquet conversion)",
    )
    return parser


def _set_global_loglevel(loglevel: str) -> None:
    level = getattr(logging, loglevel.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {loglevel}")

    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=log_format, datefmt=date_format)
    else:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))


def run_cli_args(args: argparse.Namespace) -> None:
    args.end_date = args.end_date or args.start_date
    args.resample_freq = args.resample_freq or DEFAULT_RESAMPLE_FREQ
    _set_global_loglevel(args.loglevel)
    logger.info(
        "Parsed CLI args: exchange=%s data_type=%s symbol=%s"
        " start_date=%s end_date=%s data_dir=%s force_reload=%s"
        " resample_freq=%s loglevel=%s"
        " peek=%s peek_rows=%s peek_transposed=%s raw=%s",
        args.exchange,
        args.data_type,
        args.symbol,
        args.start_date,
        args.end_date,
        args.data_dir,
        args.force_reload,
        args.resample_freq,
        args.loglevel,
        args.peek,
        args.peek_rows,
        args.peek_transposed,
        args.raw,
    )

    if args.peek:
        peek_streaming_raw(
            exchange=args.exchange,
            data_type=args.data_type,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            head_rows=args.peek_rows,
            transpose_preview=args.peek_transposed,
        )
        return

    if args.raw:
        paths = download_raw_csv_gz(
            exchange=args.exchange,
            data_type=args.data_type,
            symbols=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            data_dir=args.data_dir,
            force_reload=args.force_reload,
        )
        for path in paths:
            logger.debug(path)
        return
    
    paths = download_resample(
        exchange=args.exchange,
        data_type=args.data_type,
        symbols=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        force_reload=args.force_reload,
        resample_freq=args.resample_freq,
    )
    for path in paths:
        logger.debug(path)


def run_cli(argv: list[str] | None = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    run_cli_args(args)



