from tardis.pcp_metrics import compute_pcp_metrics
import logging
import polars as pl
import glob
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import timedelta, datetime
from tardis.download_and_convert import TARDIS_COLUMN_TYPES
from tardis.implied_vol import compute_black_implied_vols

logger = logging.getLogger(__name__)


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
