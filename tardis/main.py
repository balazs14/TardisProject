import sys
sys.path.append('/my/TardisProject')

from tardis.day_sample_options import sample_day_options, SampleConfig
from tardis.pcp_metrics import compute_pcp_metrics
from tardis.implied_vol import compute_black_implied_vols
import logging
import pandas as pd
import polars as pl
import itertools
import argparse
import glob
import pyarrow as pa
from datetime import datetime, timedelta
from tardis.process import compact  # noqa: F401


logger = logging.getLogger(__name__)


def _configure_logging(level: str = "ERROR"):
    level_name = str(level).strip().upper()
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

def one_day(day, exchange, config: SampleConfig, align_only=True):
    day_str = f'{pd.Timestamp(day).date()}'
    logger.info("one_day start exchange=%s day=%s freq=%s", exchange, day_str, config.freq)
    raw = sample_day_options(day_str, exchange, config)
    logger.info("one_day sampled exchange=%s day=%s rows=%d", exchange, day_str, len(raw))
    if align_only:
        out = raw
    else:
        out = compute_pcp_metrics(raw)

        # Downstream IV pass: compute inverse Black IVs from coin premium columns
        # and write them into existing *_iv columns.
        out = compute_black_implied_vols(
            out,
            r=0.05,
            price_columns=["call_bid_price", "call_ask_price", "put_bid_price", "put_ask_price"],
            output_columns=["call_bid_iv", "call_ask_iv", "put_bid_iv", "put_ask_iv"],
        )

    logger.info("one_day done exchange=%s day=%s rows=%d", exchange, day_str, len(out))
    return out.to_pandas()

def multi_day_exchange(days, exchanges, config: SampleConfig, align_only=True):
    if not isinstance(days, (list,tuple)):
        days = [days]
    if not isinstance(exchanges, (list,tuple)):
        exchanges = [exchanges]
    logger.info("multi_day_exchange start days=%d exchanges=%d freq=%s", len(days), len(exchanges), config.freq)
    dfs=[]
    for day, ex in itertools.product(days, exchanges):
        df = one_day(day, ex, config, align_only )
        df['mdy'] = pd.Timestamp(day)
        df['exchange'] = ex
        dfs.append(df)
    out = pd.concat(dfs)
    logger.info("multi_day_exchange done rows=%d", len(out))
    return out

def testme():
    _configure_logging('INFO')
    main('2026-03-14', 'deribit', cleanup_csv=False, align_only=True)
    #main('2026-02-13', 'okex')

def main(
    days,
    exchanges,
    freq='5min',
    post_resample_freq=None,
    force_reload=False,
    cleanup_csv=True,
    cleanup_intermediate_parquet=False,
    align_only=True,
):
    if not isinstance(days, (list,tuple)):
        days = [days]
    if not isinstance(exchanges, (list,tuple)):
        exchanges = [exchanges]
    logger.info(
        "main start days=%d exchanges=%d freq=%s post_resample_freq=%s force_reload=%s cleanup_csv=%s cleanup_intermediate_parquet=%s",
        len(days),
        len(exchanges),
        freq,
        post_resample_freq,
        force_reload,
        cleanup_csv,
        cleanup_intermediate_parquet,
    )
    dfs = []
    for ex in exchanges:
        cfg = SampleConfig(
            data_dir=f'datasets/{ex}_raw',
            output_dir=f'datasets/{ex}',
            freq=freq,
            post_resample_freq=post_resample_freq,
            force_reload=force_reload,
            cleanup_csv=cleanup_csv,
            cleanup_intermediate_parquet=cleanup_intermediate_parquet,
            )
        df = multi_day_exchange(days, ex, cfg, align_only=align_only)
        dfs.append(df)
    out = pd.concat(dfs)
    logger.info("main done rows=%d", len(out))
    return out

def _split_csv_or_scalar(value: str):
    items = [v.strip() for v in value.split(",") if v.strip()]
    if len(items) <= 1:
        return items[0] if items else value
    return items


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Tardis PCP pipeline for given days/exchanges")
    parser.add_argument("--days", required=True, help="Single day or comma-separated days (YYYY-MM-DD)")
    parser.add_argument("--exchanges", required=True, help="Single exchange or comma-separated exchanges")
    parser.add_argument("--loglevel", default="ERROR", help="Logging level (default: ERROR)")
    parser.add_argument("--freq", default="5min", help="Resampling frequency (default: 5min)")
    parser.add_argument("--post_resample_freq", default=None, help="Optional second-stage resample frequency before alignment (e.g. 15min)")
    parser.add_argument("--force_reload", type=_parse_bool, default=False, help="Force reload source data (default: False)")
    parser.add_argument("--cleanup_csv", type=_parse_bool, default=True, help="Remove source CSV files after conversion (default: True)")
    parser.add_argument("--cleanup_intermediate_parquet", type=_parse_bool, default=False, help="Remove intermediate parquet and pkl files after pipeline (default: False)")
    parser.add_argument("--align_only", type=_parse_bool, default=True, help="Stop after alignment, skip PCP metrics and IV (default: True)")

    args = parser.parse_args()
    days = _split_csv_or_scalar(args.days)
    exchanges = _split_csv_or_scalar(args.exchanges)
    _configure_logging(args.loglevel)

    logger.info(
        "cli args days=%s exchanges=%s freq=%s post_resample_freq=%s force_reload=%s cleanup_csv=%s cleanup_intermediate_parquet=%s",
        days,
        exchanges,
        args.freq,
        args.post_resample_freq,
        args.force_reload,
        args.cleanup_csv,
        args.cleanup_intermediate_parquet,
    )

    result = main(
        days=days,
        exchanges=exchanges,
        freq=args.freq,
        post_resample_freq=args.post_resample_freq,
        force_reload=args.force_reload,
        cleanup_csv=args.cleanup_csv,
        cleanup_intermediate_parquet=args.cleanup_intermediate_parquet,
        align_only=args.align_only,
    )
    print(result)

