import logging
import glob
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from tardis.download_files import TARDIS_COLUMN_TYPES
from tardis.implied_vol import compute_black_implied_vols


logger = logging.getLogger(__name__)
logger.setLevel('ERROR')

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
    logger.debug(
        "compute_pcp_metrics start rows=%d pandas_interface=%s contract_size=%s cost_per_notional=%s",
        input_rows,
        pandas_interface,
        contract_size,
        cost_per_notional,
    )
    df = pl.from_pandas(aligned_df) if pandas_interface else aligned_df

    df = df.filter(pl.col("fut_exp").is_not_null())
    logger.debug(
        "compute_pcp_metrics filtered_exp rows_in=%d rows_kept=%d",
        input_rows,
        df.height,
    )

    index = (pl.col("spot_ask_price") + pl.col("spot_bid_price")) / 2
    call_ask_xs = pl.col("call_ask_price") * pl.col("spot_ask_price")
    call_bid_xs = pl.col("call_bid_price") * pl.col("spot_bid_price")
    put_ask_xs = pl.col("put_ask_price") * pl.col("spot_ask_price")
    put_bid_xs = pl.col("put_bid_price") * pl.col("spot_bid_price")

    tte = (pl.col("exp") - pl.col("ts")) / pl.duration(days=365)
    dscnt = (-(r) * tte).exp()

    pcpb_forward = (call_bid_xs - put_ask_xs - (pl.col("fut_ask_price") - pl.col("strike")) * dscnt) * contract_size
    pcpb_backward = (call_ask_xs - put_bid_xs - (pl.col("fut_bid_price") - pl.col("strike")) * dscnt) * contract_size * -1

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
            pl.lit(contract_size).alias("contract_size"),
            (0.5 * pl.col("fut_bid_price") + 0.5 * pl.col("fut_ask_price")).alias("fut_mid_price"),
            (pl.col("strike") / (0.5 * pl.col("fut_bid_price") + 0.5 * pl.col("fut_ask_price"))).alias("rel_strike"),
            pl.col("ts").dt.date().alias("mdy"),
        ]
    )

    logger.debug("compute_pcp_metrics done rows=%d columns=%d", out.height, len(out.columns))

    return out.to_pandas() if pandas_interface else out


def compact(from_date, to_date, exchanges, ref_syms, freq="5min", output_path=None,
            rel_strike_min=0.0, rel_strike_max=100,
            cols_needed=None
        ):

    if not isinstance(exchanges, (list, tuple)):
        exchanges = [exchanges]
    if not isinstance(ref_syms, (list, tuple)):
        ref_syms = [ref_syms]

    start = datetime.fromisoformat(str(from_date)).date()
    end = datetime.fromisoformat(str(to_date)).date()
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    day_strs = [d.strftime("%Y-%m-%d") for d in days]
    logger.info(f'potentially {len(day_strs)} days')

    if output_path is None:
        output_path = f"datasets/compact_options_{from_date}_{to_date}_{freq}.parquet"

    writer = None
    total_rows = 0
    try:
        for ex in exchanges:
            files = []
            for day in day_strs:
                pattern = f"datasets/{ex}/{ex}_aligned_options_{day}_{freq}.parquet"
                files.extend(glob.glob(pattern))
            logger.info(f'found {len(files)} files for {ex}')
            for f in files:
                raw = pl.read_parquet(f)

                # pcp
                df_with_pcp = compute_pcp_metrics(raw)

                # implied vol
                df = compute_black_implied_vols(
                        df_with_pcp,
                        r=0.05,
                        price_columns=["call_bid_price", "call_ask_price", "put_bid_price", "put_ask_price"],
                        output_columns=["call_bid_iv", "call_ask_iv", "put_bid_iv", "put_ask_iv"],
                    )

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
