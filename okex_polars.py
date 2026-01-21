import logging
import warnings
warnings.simplefilter('ignore')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import re
import os
from pprint import pprint
import requests
import itertools
import polars as pl
import pandas as pd
import numpy as np
import pickle
from datetime import date
import pyarrow as pa
import pyarrow.parquet as pq

from dataclasses import dataclass, field
from typing import List, Dict

from TardisProject.lib import pandas_utils as pu
from TardisProject.lib import utils

from tardis_client import TardisClient, Channel
from tardis_dev import datasets

datadir = 'okex'


def align_calls_puts(opt_df):
    # Separate calls and puts
    calls = opt_df.filter(pl.col('pc') == 'C').clone()
    puts = opt_df.filter(pl.col('pc') == 'P').clone()

    # Rename option specific columns
    rename_cols = ['ask_amount', 'ask_price', 'bid_amount', 'bid_price', 'symbol']
    calls = calls.rename({c: f'call_{c}' for c in rename_cols})
    puts = puts.rename({c: f'put_{c}' for c in rename_cols})

    # Drop shared columns from puts to avoid duplication
    shared_fut_cols = ['fut_ask_amount', 'fut_ask_price', 'fut_bid_amount', 'fut_bid_price', 'fut_exp']
    calls = calls.drop('pc')
    puts = puts.drop(['pc'] + shared_fut_cols, strict=False)

    # Merge on keys
    on_keys = ['ts', 'ref_sym', 'fut_sym', 'exp', 'strike']
    return calls.join(puts, on=on_keys, how='inner')


def pcp_breaking(opt_df):
    # c - p = f - k, pcpb = c-p-f+k
    df = opt_df.filter(pl.col('fut_exp').is_not_null())
    df = align_calls_puts(df)
    df = df.with_columns([
        (pl.col('call_ask_price') - pl.col('put_bid_price') - pl.col('fut_bid_price') + pl.col('strike')).alias('pcpb_forward'),
        (pl.col('call_bid_price') - pl.col('put_ask_price') - pl.col('fut_ask_price') + pl.col('strike')).alias('pcpb_backward')
    ])
    return df


def mark_up_with_futures(resampled_df):
    ref_sym = 'BTC-USD'

    fut_df = resampled_df.filter(
        (pl.col('ref_sym') == ref_sym) & (pl.col('pc') == 'F')
    ).rename({
        'ask_amount': 'fut_ask_amount',
        'ask_price': 'fut_ask_price',
        'bid_amount': 'fut_bid_amount',
        'bid_price': 'fut_bid_price',
        'exp': 'fut_exp'
    })

    opt_df = resampled_df.filter(
        (pl.col('ref_sym') == ref_sym) & (pl.col('pc').is_in(['C', 'P']))
    )

    opt_df = opt_df.join(
        fut_df.select(['ts', 'fut_sym', 'fut_ask_amount', 'fut_ask_price', 'fut_bid_amount', 'fut_bid_price', 'fut_exp']),
        on=['ts', 'fut_sym'],
        how='left'
    )

    return opt_df, fut_df


def merge_data(fr='2025-01-02', to='2025-01-02 00:10:00', freq='1min', output_tag='resampled'):
    chunksize = 500_000
    df_iters, dayinfo = iter_datasets(fr, chunksize=chunksize)

    iter_keys = list(df_iters.keys())

    dfs = {k: pl.DataFrame() for k in iter_keys}
    stop_conditions = {k: False for k in iter_keys}
    process_to = {k: pd.Timestamp.max.to_pydatetime() for k in iter_keys}

    cols_to_save = ['ref_sym', 'fut_sym', 'exp', 'pc', 'strike']
    cols_to_save += ['bid_price', 'ask_price', 'bid_amount', 'ask_amount']
    all_symbols = sorted(list(set(list(dayinfo.opt_symbols) + list(dayinfo.fut_symbols))))

    last_values = {sym: {col: None for col in cols_to_save} for sym in all_symbols}
    output_chunks = []
    logger.info(f'done prep')

    to_timestamp = pd.Timestamp.max.to_pydatetime() if to == 'max' else pd.Timestamp(to).to_pydatetime()

    while True:
        # 1. Refill buffers
        for sym in iter_keys:
            if not stop_conditions[sym] and len(dfs[sym]) < chunksize:
                try:
                    chunk = next(df_iters[sym])
                    chunk = augment_quotes(chunk)
                    if dfs[sym].is_empty():
                        dfs[sym] = chunk
                    else:
                        dfs[sym] = pl.concat([dfs[sym], chunk])
                except StopIteration:
                    stop_conditions[sym] = True
                if not dfs[sym].is_empty():
                    process_to[sym] = dfs[sym].select('ts').max().item()
                logger.debug(f'done refill buffer {sym}')

        if all([df.is_empty() for df in dfs.values()]):
            break

        # 2. Safe processing window
        cutoff = min([process_to[sym] for sym in iter_keys])
        if all([stop_conditions[sym] for sym in iter_keys]):
            cutoff = pd.Timestamp.max.to_pydatetime()

        logger.info(f'done read up to cutoff {cutoff}')

        # 3. Slice data up to cutoff
        if cutoff == pd.Timestamp.max.to_pydatetime():
            df_slices = dfs
            dfs = {sym: pl.DataFrame() for sym in iter_keys}
        else:
            df_slices = {sym: dfs[sym].filter(pl.col('ts') < cutoff) for sym in iter_keys}
            dfs = {sym: dfs[sym].filter(pl.col('ts') >= cutoff) for sym in iter_keys}

        if all([df_slices[sym].is_empty() for sym in iter_keys]):
            logger.info('refill buffers')
            continue

        # 4. Merge and Pivot
        min_ts = None
        max_ts = None
        current_data_map = {}

        for sym, df in df_slices.items():
            if df.is_empty():
                continue
            if sym == 'OPTIONS':
                for s, g in df.partition_by('symbol', as_dict=True).items():
                    current_data_map[s] = g
                    ts_min = g.select('ts').min().item()
                    ts_max = g.select('ts').max().item()
                    min_ts = ts_min if min_ts is None else min(min_ts, ts_min)
                    max_ts = ts_max if max_ts is None else max(max_ts, ts_max)
            else:
                current_data_map[sym] = df
                ts_min = df.select('ts').min().item()
                ts_max = df.select('ts').max().item()
                min_ts = ts_min if min_ts is None else min(min_ts, ts_min)
                max_ts = ts_max if max_ts is None else max(max_ts, ts_max)

        if min_ts is None:
            continue

        t_start = pd.Timestamp(min_ts).floor(freq).to_pydatetime()
        t_end = min(pd.Timestamp(max_ts).ceil(freq).to_pydatetime(), cutoff if cutoff != pd.Timestamp.max.to_pydatetime() else max_ts)

        # Create time grid
        grid = pl.from_pandas(pd.date_range(t_start, t_end, freq=freq))

        chunk_dfs = []
        for sym in all_symbols:
            df = current_data_map.get(sym)
            last_val = last_values[sym]

            if df is not None and not df.is_empty():
                df_dedup = df.unique(subset=['ts'], keep='last').select(cols_to_save+['ts'])
                # Forward fill
                df_filled = df_dedup.join(
                    grid.to_frame('ts'),
                    on='ts',
                    how='right'
                ).sort('ts').with_columns(
                    [pl.col(col).forward_fill() for col in cols_to_save]
                )
            else:
                df_filled = grid.to_frame('ts').with_columns([
                    pl.lit(None).alias(col) for col in cols_to_save
                ])

            # Update last values
            if not df_filled.is_empty():
                last_row = df_filled.tail(1).to_dicts()[0]
                for col in cols_to_save:
                    if last_row.get(col) is not None:
                        last_values[sym][col] = last_row[col]

            df_filled = df_filled.with_columns(pl.lit(sym).alias('symbol'))
            chunk_dfs.append(df_filled)

        resampled = pl.concat(chunk_dfs)
        resampled = resampled.sort('ts')

        if not resampled.is_empty():
            output_chunks.append(resampled)

        if cutoff > pl.Datetime(to):
            logger.info(f"Cutoff {cutoff} exceeds 'to' timestamp {to}. Finalizing output.")
            break

    logger.info("All chunks processed. Concatenating and writing final output.")
    if not output_chunks:
        logger.info("No data to write.")
        return

    final_df = pl.concat(output_chunks)
    final_df = final_df.unique(subset=['ts', 'symbol'], keep='last')

    fname = f'{datadir}/okex-options_{output_tag}_{fr}_{freq}.parquet'
    final_df.write_parquet(fname)
    logger.info(f"Wrote {len(final_df)} total rows to {fname}")


@dataclass(frozen=False, slots=False)
class DayInfo:
    date: str
    opt_symbols: List[str]
    fut_symbols: List[str]
    opt_quotes_fname: str
    fut_quotes_fnames: Dict[str, str]


def read_parquet_chunks(path, chunksize):
    """Helper to read a Parquet file in chunks and yield Polars DataFrames."""
    def iter_chunks():
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunksize):
            yield pl.from_arrow(batch)
    return iter_chunks()


def iter_datasets(fr='2025-01-02', chunksize=100000):
    dayinfo = download_multiple_csvs([fr])
    logger.info(f"Initializing iterators for {dayinfo.date}")
    opt_df_iter = read_parquet_chunks(dayinfo.opt_quotes_fname, chunksize=chunksize)

    fut_df_iters = {}
    for sym in dayinfo.fut_symbols:
        fut_df_iters[sym] = read_parquet_chunks(dayinfo.fut_quotes_fnames[sym], chunksize=chunksize)

    df_iters = fut_df_iters
    df_iters['OPTIONS'] = opt_df_iter
    logger.info(f'done creating {len(df_iters)} iterators')
    return df_iters, dayinfo


def augment_quotes(df: pl.DataFrame) -> pl.DataFrame:
    """Augment quotes dataframe with parsed symbol information.

    Expected symbol formats:
      - Futures:  BTC-USD-250331
      - Options:  BTC-USD-250331-30000-C
    local_timestamp assumed to be in microseconds since epoch.
    """
    if df.is_empty():
        return df

    parts = pl.col("symbol").str.split("-")

    df = df.with_columns(
        [
            parts.list.get(0).alias("_A"),
            parts.list.get(1).alias("_B"),
            parts.list.get(2).alias("exp_str"),
            parts.list.get(3, null_on_oob=True).alias("_D"),  # strike (options) or null (futures)
            parts.list.get(4, null_on_oob=True).alias("_E"),  # P/C (options) or null (futures)
        ]
    )

    df = df.with_columns(
        [
            # strike: only present for options; parse safely
            pl.when(pl.col("_D").is_not_null())
              .then(pl.col("_D").cast(pl.Float64, strict=False))
              .otherwise(pl.lit(None, dtype=pl.Float64))
              .alias("strike"),

            # pc: use 'F' for futures if missing
            pl.when(pl.col("_E").is_not_null())
              .then(pl.col("_E"))
              .otherwise(pl.lit("F"))
              .alias("pc"),

            # Normalize quote/settle ticker USD -> USDT if that's your convention
            pl.when(pl.col("_B") == "USD")
              .then(pl.lit("USDT"))
              .otherwise(pl.col("_B"))
              .alias("_BT"),
        ]
    )

    df = df.with_columns(
        [
            pl.concat_str([pl.col("_A"), pl.col("_BT"), pl.col("exp_str")], separator="-").alias("fut_sym"),
            pl.concat_str([pl.col("_A"), pl.col("_B")], separator="-").alias("ref_sym"),

            # exp_str like '250331' -> Date -> Datetime (midnight)
            pl.col("exp_str")
              .str.strptime(pl.Date, format="%y%m%d", strict=False)
              .cast(pl.Datetime)
              .alias("exp"),

            # local_timestamp microseconds since epoch -> Datetime(us)
            pl.from_epoch(pl.col("local_timestamp"), time_unit="us").alias("ts"),
        ]
    )

    df = df.drop(["_A", "_B", "_BT", "exp_str", "_D", "_E"], strict=False)
    return df



def download_multiple_csvs(frs=['2025-01-02']):
    todo = pl.DataFrame({
        'ex': ['okex-futures', 'okex-options'],
        'dt': ['derivative_ticker', 'quotes'],
        'sym': ['FUTURES', 'OPTIONS']
    })

    for fr in frs:
        to = str((pd.Timestamp(fr) + pd.Timedelta(days=1)).date())
        for row in todo.iter_rows(named=True):
            ex, dt, sym = row['ex'], row['dt'], row['sym']
            try:
                download_single_csv(ex, dt, fr, to, sym)
            except Exception as e:
                logger.warning(e)
                pass

        futfname = download_single_csv('okex-futures', 'derivative_ticker', fr, to, 'FUTURES')
        fut_symbols_fname = futfname + '_syms.pkl'
        if os.path.exists(fut_symbols_fname):
            with open(fut_symbols_fname, 'rb') as f:
                fut_symbols = pickle.load(f)
        else:
            futs_df = pl.read_parquet(futfname)
            fut_symbols = futs_df.select('symbol').unique().to_series().to_list()
            with open(fut_symbols_fname, 'wb') as f:
                pickle.dump(fut_symbols, f)

        fut_quotes_fnames = {}
        for futsym in fut_symbols:
            fut_quotes_fnames[futsym] = download_single_csv('okex-futures', 'quotes', fr, fr, futsym)

        opt_quotes_fname = download_single_csv('okex-options', 'quotes', fr, to, 'OPTIONS')
        opt_symbols_fname = opt_quotes_fname + '_syms.pkl'
        if os.path.exists(opt_symbols_fname):
            with open(opt_symbols_fname, 'rb') as f:
                opt_symbols = pickle.load(f)
        else:
            opts_df = pl.read_parquet(opt_quotes_fname)
            opt_symbols = opts_df.select('symbol').unique().to_series().to_list()
            with open(opt_symbols_fname, 'wb') as f:
                pickle.dump(opt_symbols, f)

        return DayInfo(fr, opt_symbols, fut_symbols, opt_quotes_fname, fut_quotes_fnames)


def download_single_csv(ex, dt, fr, to, sym):
    csv_fname = f'{datadir}/{ex}_{dt}_{fr}_{sym}.csv.gz'
    pq_fname = f'{datadir}/{ex}_{dt}_{fr}_{sym}.parquet'

    if os.path.exists(pq_fname):
        logger.debug(f'Parquet file already exists: {pq_fname}')
        return pq_fname

    if not os.path.exists(csv_fname):
        logger.info(f'Downloading {csv_fname}')
        datasets.download(
            exchange=ex,
            data_types=[dt],
            from_date=fr,
            to_date=to,
            symbols=[sym],
            api_key=os.environ.get('TARDIS_API_KEY', None)
        )
    else:
        logger.debug(f'CSV already here {csv_fname}')

    logger.info(f'Converting {csv_fname} to {pq_fname}...')

    df = pl.read_csv(csv_fname)
    df.write_parquet(pq_fname)
    logger.info(f'Finished converting to {pq_fname}')
    return pq_fname


if __name__ == '__main__':
    merge_data(fr='2025-01-02', to='2025-01-02 00:00:10', freq='1min', output_tag='1min')


