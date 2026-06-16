import logging
import glob
from datetime import timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from tardis.download_files import TARDIS_COLUMN_TYPES
from tardis.implied_vol import compute_black_implied_vols
from tardis import test_utils as tu


logger = logging.getLogger(__name__)


def _subsample_on_time_grid(df, subsample_freq):
    """Keep rows only on timestamps aligned to the requested interval."""
    if "timestamp" not in df.columns:
        raise ValueError("Input frame must include 'timestamp' for subsampling")

    rows_in = df.height
    out = df.filter(pl.col("timestamp") == pl.col("timestamp").dt.truncate(subsample_freq))
    logger.debug(
        "subsample_on_time_grid every=%s rows_in=%d rows_out=%d",
        subsample_freq,
        rows_in,
        out.height,
    )
    return out

def _as_absolute_path(path_value, arg_name):
    """Validate user-provided path-like values are absolute and return Path."""
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{arg_name} must be an absolute path, got: {path_value}")
    return path

def compute_pcp_metrics(
    aligned_df,
    cost_per_notional=0.003,
    fut_mgn_rate=0.2,
    short_put_mgn_rate=0.5,
    short_call_mgn_rate=0.5,
    r=0.05,
    contract_size=0.01,
):
    """Compute put-call parity metrics from aligned call/put/future/spot legs."""
    pandas_interface = isinstance(aligned_df, pd.DataFrame)
    input_rows = len(aligned_df)
    
    # Inverse instruments quote option premium in underlying coin, while linear
    # instruments quote premium directly in quote currency. Apply spot conversion
    # only for inverse rows.

    logger.debug(
        "compute_pcp_metrics start rows=%d pandas_interface=%s contract_size=%s cost_per_notional=%s",
        input_rows,
        pandas_interface,
        contract_size,
        cost_per_notional,
    )
    df = pl.from_pandas(aligned_df) if pandas_interface else aligned_df

    expiry_col = "fut_exp" if "fut_exp" in df.columns else "exp"
    df = df.filter(pl.col(expiry_col).is_not_null())
    logger.debug(
        "compute_pcp_metrics filtered_exp rows_in=%d rows_kept=%d expiry_col=%s",
        input_rows,
        df.height,
        expiry_col,
    )

    if "timestamp" in df.columns and df.height > 0:
        day_stats = df.select(
            [
                pl.col("timestamp").dt.date().min().alias("min_day"),
                pl.col("timestamp").dt.date().max().alias("max_day"),
                pl.col("timestamp").dt.date().n_unique().alias("n_days"),
            ]
        ).row(0)
        logger.debug(
            "compute_pcp_metrics date_window min_day=%s max_day=%s n_days=%s",
            day_stats[0],
            day_stats[1],
            day_stats[2],
        )
    else:
        logger.debug("compute_pcp_metrics date_window unavailable (missing timestamp or empty frame)")

    inverse = pl.col("inverse") if "inverse" in df.columns else pl.lit(True)
    if "call_symbol" in df.columns:
        # Backward-compatible guard: older aligned Deribit files can carry a
        # stale inverse=True flag for linear USDC option symbols.
        is_linear_symbol = pl.col("call_symbol").cast(pl.String).str.contains(r"_[A-Z0-9]+-")
        inverse = pl.when(is_linear_symbol).then(pl.lit(False)).otherwise(inverse)
    index = (pl.col("spot_ask_price") + pl.col("spot_bid_price")) / 2
    call_ask_xs = pl.when(inverse).then(pl.col("call_ask_price") * pl.col("spot_ask_price")).otherwise(pl.col("call_ask_price"))
    call_bid_xs = pl.when(inverse).then(pl.col("call_bid_price") * pl.col("spot_bid_price")).otherwise(pl.col("call_bid_price"))
    put_ask_xs = pl.when(inverse).then(pl.col("put_ask_price") * pl.col("spot_ask_price")).otherwise(pl.col("put_ask_price"))
    put_bid_xs = pl.when(inverse).then(pl.col("put_bid_price") * pl.col("spot_bid_price")).otherwise(pl.col("put_bid_price"))

    tte = (pl.col("exp") - pl.col("timestamp")) / pl.duration(days=365)
    dscnt = (-(r) * tte).exp()

    pcpb_forward = (call_bid_xs - put_ask_xs - (pl.col("fut_ask_price") - pl.col("strike")) * dscnt) * contract_size
    pcpb_backward = (call_ask_xs - put_bid_xs - (pl.col("fut_bid_price") - pl.col("strike")) * dscnt) * contract_size * -1

    call_bid_amount = pl.col("call_bid_amount") if "call_bid_amount" in df.columns else pl.lit(None, dtype=pl.Float64)
    call_ask_amount = pl.col("call_ask_amount") if "call_ask_amount" in df.columns else pl.lit(None, dtype=pl.Float64)
    put_bid_amount = pl.col("put_bid_amount") if "put_bid_amount" in df.columns else pl.lit(None, dtype=pl.Float64)
    put_ask_amount = pl.col("put_ask_amount") if "put_ask_amount" in df.columns else pl.lit(None, dtype=pl.Float64)

    cost = index.abs() * cost_per_notional * contract_size
    capital_fwd = (
        (index * fut_mgn_rate) + (index * short_call_mgn_rate) + (call_bid_xs - put_ask_xs)
    ).clip(lower_bound=0) * contract_size
    capital_bck = (
        (index * fut_mgn_rate) + (pl.col("strike") * short_put_mgn_rate) + (put_bid_xs - call_ask_xs)
    ).clip(lower_bound=0) * contract_size

    out = df.with_columns(
        [
            index.alias("index"),
            call_ask_xs.alias("call_ask_price_xS"),
            call_bid_xs.alias("call_bid_price_xS"),
            put_ask_xs.alias("put_ask_price_xS"),
            put_bid_xs.alias("put_bid_price_xS"),
            tte.alias("tte"),
            pcpb_forward.alias("pcpb_forward"),
            pcpb_backward.alias("pcpb_backward"),
            cost.alias("cost"),
            capital_fwd.alias("capital_fwd"),
            capital_bck.alias("capital_bck"),
            (pcpb_forward - cost).alias("pcpb_fwd_real"),
            (pcpb_backward - cost).alias("pcpb_bck_real"),
            ((pcpb_forward / capital_fwd) * 10000).alias("pcpb_fwd_bp"),
            ((pcpb_backward / capital_bck) * 10000).alias("pcpb_bck_bp"),
            (((pcpb_forward - cost) / capital_fwd) * 10000).alias("pcpb_fwd_real_bp"),
            (((pcpb_backward - cost) / capital_bck) * 10000).alias("pcpb_bck_real_bp"),
            (((pcpb_forward / capital_fwd) * 10000) / tte).alias("pcpb_fwd_ann_bp"),
            (((pcpb_backward / capital_bck) * 10000) / tte).alias("pcpb_bck_ann_bp"),
            ((((pcpb_forward - cost) / capital_fwd) * 10000) / tte).alias("pcpb_fwd_real_ann_bp"),
            ((((pcpb_backward - cost) / capital_bck) * 10000) / tte).alias("pcpb_bck_real_ann_bp"),
            (((put_ask_xs - put_bid_xs) / index) * 10000).alias("put_opt_spread_bp"),
            (((call_ask_xs - call_bid_xs) / index) * 10000).alias("call_opt_spread_bp"),
            (pl.max_horizontal(((put_ask_xs - put_bid_xs) / index) * 10000, ((call_ask_xs - call_bid_xs) / index) * 10000)).alias("bigger_opt_spread_bp"),
            (pl.min_horizontal(((put_ask_xs - put_bid_xs) / index) * 10000, ((call_ask_xs - call_bid_xs) / index) * 10000)).alias("smaller_opt_spread_bp"),
            (((pcpb_forward - cost) / (contract_size * index)) * 10000).alias("amu_fwd_bp"),
            (((pcpb_backward - cost) / (contract_size * index)) * 10000).alias("amu_bck_bp"),
            (pl.min_horizontal(
                call_bid_amount,
                call_ask_amount,
                put_bid_amount,
                put_ask_amount,
            ) * index).alias("min_quote_size_dollar"),
            pl.lit(contract_size).alias("contract_size"),
            (0.5 * pl.col("fut_bid_price") + 0.5 * pl.col("fut_ask_price")).alias("fut_mid_price"),
            (pl.col("strike") / (0.5 * pl.col("fut_bid_price") + 0.5 * pl.col("fut_ask_price"))).alias("rel_strike"),
            pl.col("timestamp").dt.date().alias("mdy"),
        
        ]
    )

    logger.debug("compute_pcp_metrics done rows=%d columns=%d", out.height, len(out.columns))

    return out.to_pandas() if pandas_interface else out


def compact(from_date, to_date, exchanges, ref_syms, freq="5min", output_path=None,
            data_root=None,
            rel_strike_min=0.0, rel_strike_max=100,
            cols_needed=None,
            compute_implied_vols=False,
            subsample_freq=None,
        ):
    """Build a compact parquet by scanning aligned daily files under an absolute dataset root.

    Args:
        output_path: Absolute output parquet path. If omitted, an absolute path under
            ``data_root`` is used.
        data_root: Absolute path to datasets root that contains exchange folders.
            Defaults to ``<repo>/datasets`` resolved from this module location.
        subsample_freq: Optional interval (for example, ``"1h"`` or ``"1hr"``)
            used to keep only timestamps aligned to that grid.
    """

    if not isinstance(exchanges, (list, tuple)):
        exchanges = [exchanges]
    if not isinstance(ref_syms, (list, tuple)):
        ref_syms = [ref_syms]

    start = datetime.fromisoformat(str(from_date)).date()
    end = datetime.fromisoformat(str(to_date)).date()
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    day_strs = [d.strftime("%Y-%m-%d") for d in days]
    logger.info(f'potentially {len(day_strs)} days')

    if data_root is None:
        # Use repo-level datasets directory independent of current working directory.
        data_root = Path(__file__).resolve().parents[1] / "datasets"
    else:
        data_root = _as_absolute_path(data_root, "data_root")

    if output_path is None:
        output_path = data_root / f"compact_options_{from_date}_{to_date}_{freq}.parquet"
    else:
        output_path = _as_absolute_path(output_path, "output_path")

    data_root = str(data_root)
    output_path = str(output_path)
    
    writer = None
    total_rows = 0
    try:
        for ex in exchanges:
            files = []
            for day in day_strs:
                pattern = str(Path(data_root) / ex / f"{ex}_aligned_options_{day}_{freq}.parquet")
                files.extend(glob.glob(pattern))
            logger.info(f'found {len(files)} files for {ex}')
            for f in files:
                logger.debug("compact processing file=%s", f)
                if Path(f).stat().st_size == 0:
                    logger.warning("Skipping empty parquet file: %s", f)
                    continue
                raw = pl.read_parquet(f)
                logger.info("compact input file=%s rows=%d", f, len(raw))

                if subsample_freq:
                    raw = _subsample_on_time_grid(raw, subsample_freq)
                    if raw.height == 0:
                        logger.debug(
                            "compact skipping file=%s after subsample_freq=%s because no rows remain",
                            f,
                            subsample_freq,
                        )
                        continue

                # pcp
                df_with_pcp = compute_pcp_metrics(raw)

                if compute_implied_vols:
                    df = compute_black_implied_vols(
                            df_with_pcp,
                            r=0.05,
                            price_columns=["call_bid_price", "call_ask_price", "put_bid_price", "put_ask_price"],
                            output_columns=["call_bid_iv", "call_ask_iv", "put_bid_iv", "put_ask_iv"],
                        )
                else:
                    df = df_with_pcp

                # Keep rows near-the-money only.
                rel_strike_ok = (
                    (pl.col("rel_strike") > rel_strike_min)
                    & (pl.col("rel_strike") < rel_strike_max)
                )

                # Require a complete top-of-book across call/put/future legs.
                prices_present = (
                    pl.col("call_bid_price").is_not_null()
                    & pl.col("put_bid_price").is_not_null()
                    & pl.col("fut_bid_price").is_not_null()
                    & pl.col("call_ask_price").is_not_null()
                    & pl.col("put_ask_price").is_not_null()
                    & pl.col("fut_ask_price").is_not_null()
                )

                # Keep only requested reference underlyings.
                ref_sym_ok = pl.col("ref_sym").is_in(ref_syms)

                df = (
                    df.with_columns(pl.lit(ex).alias("exchange"))
                    .filter(rel_strike_ok & prices_present & ref_sym_ok)
                )
                if cols_needed:
                    df = df.select(cols_needed)

                table = df.to_arrow()
                if writer is None:
                    schema = pa.schema([
                        pa.field(
                            name,
                            TARDIS_COLUMN_TYPES.get(name, table.schema.field(name).type),
                        )
                        for name in table.column_names
                    ])
                    writer = pq.ParquetWriter(output_path, schema)
                table = table.cast(writer.schema)
                writer.write_table(table)
                total_rows += len(df)
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        logger.warning("No files found for compact in range %s to %s", from_date, to_date)
        return None
    logger.info("compact wrote %d rows to %s", total_rows, output_path)
    return output_path


def test_compute_pcp_metrics_inverse_linear_snapshot() -> None:
    ts = datetime(2025, 10, 10, 0, 0, 0)
    exp = datetime(2025, 10, 31, 0, 0, 0)
    df = pl.DataFrame(
        {
            "timestamp": [ts, ts],
            "exp": [exp, exp],
            "inverse": [True, True],
            "call_symbol": ["BTC-31OCT25-80000-C", "BTC_USDC-31OCT25-80000-C"],
            "put_symbol": ["BTC-31OCT25-80000-P", "BTC_USDC-31OCT25-80000-P"],
            "fut_sym": ["BTC-31OCT25", "BTC_USDC-31OCT25"],
            "spot_sym": ["BTC_USDC", "BTC_USDC"],
            "exchange": ["deribit", "deribit"],
            "ref_sym": ["BTCUSD", "BTCUSD"],
            "spot_bid_price": [100.0, 100.0],
            "spot_ask_price": [100.0, 100.0],
            "fut_bid_price": [105.0, 105.0],
            "fut_ask_price": [105.0, 105.0],
            "strike": [100.0, 100.0],
            "call_bid_price": [0.10, 10.0],
            "call_ask_price": [0.11, 11.0],
            "put_bid_price": [0.09, 9.0],
            "put_ask_price": [0.10, 10.0],
            "call_bid_amount": [1.0, 1.0],
            "call_ask_amount": [1.0, 1.0],
            "put_bid_amount": [1.0, 1.0],
            "put_ask_amount": [1.0, 1.0],
        }
    )

    out = compute_pcp_metrics(df)
    snap = out.select(
        [
            "call_symbol",
            "inverse",
            "call_bid_price",
            "call_bid_price_xS",
            "put_ask_price",
            "put_ask_price_xS",
            "pcpb_forward",
            "amu_fwd_bp",
        ]
    ).to_pandas()
    tu.assert_df_equal(
        snap,
        """
               call_symbol  inverse  call_bid_price  call_bid_price_xS  put_ask_price  put_ask_price_xS  pcpb_forward  amu_fwd_bp
0      BTC-31OCT25-80000-C     True             0.1               10.0            0.1              10.0      -0.049856  -528.563711
1  BTC_USDC-31OCT25-80000-C     True            10.0               10.0           10.0              10.0      -0.049856  -528.563711
""",
    )


def test_compute_pcp_metrics_linear_symbol_tail_sanity() -> None:
    ts = datetime(2025, 10, 10, 0, 0, 0)
    exp = datetime(2025, 10, 31, 0, 0, 0)
    df = pl.DataFrame(
        {
            "timestamp": [ts],
            "exp": [exp],
            "inverse": [True],
            "call_symbol": ["BTC_USDC-31OCT25-80000-C"],
            "put_symbol": ["BTC_USDC-31OCT25-80000-P"],
            "fut_sym": ["BTC_USDC-31OCT25"],
            "spot_sym": ["BTC_USDC"],
            "exchange": ["deribit"],
            "ref_sym": ["BTCUSD"],
            "spot_bid_price": [100.0],
            "spot_ask_price": [100.0],
            "fut_bid_price": [105.0],
            "fut_ask_price": [105.0],
            "strike": [100.0],
            "call_bid_price": [10.0],
            "call_ask_price": [11.0],
            "put_bid_price": [9.0],
            "put_ask_price": [10.0],
            "call_bid_amount": [1.0],
            "call_ask_amount": [1.0],
            "put_bid_amount": [1.0],
            "put_ask_amount": [1.0],
        }
    )

    out = compute_pcp_metrics(df)
    amu = float(out.select("amu_fwd_bp").item())
    assert abs(amu) < 1000, f"Unexpected AMU scale for linear symbol row: {amu}"
