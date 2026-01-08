import logging
#import warnings
#warnings.simplefilter('error')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import re
import os
from pprint import pprint
import requests
#import asyncio
import itertools
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
    calls = opt_df[opt_df['pc'] == 'C'].copy()
    puts = opt_df[opt_df['pc'] == 'P'].copy()

    # Rename option specific columns
    rename_cols = ['ask_amount', 'ask_price', 'bid_amount', 'bid_price', 'symbol']
    calls = calls.rename(columns={c: f'call_{c}' for c in rename_cols})
    puts = puts.rename(columns={c: f'put_{c}' for c in rename_cols})

    # Drop shared columns from puts to avoid duplication
    shared_fut_cols = ['fut_ask_amount', 'fut_ask_price', 'fut_bid_amount', 'fut_bid_price', 'fut_exp']
    calls = calls.drop(columns=['pc'])
    puts = puts.drop(columns=['pc'] + shared_fut_cols, errors='ignore')

    # Merge on keys
    on_keys = ['ts', 'ref_sym', 'fut_sym', 'exp', 'strike']
    return pd.merge(calls, puts, on=on_keys, how='inner')

def pcp_breaking(opt_df):
    # c - p = f - k, pcpb = c-p-f+k
    df = align_calls_puts(opt_df.loc[~opt_df.fut_exp.isna()])
    df['pcpb_forward'] = df['call_ask_price'] - df['put_bid_price'] - df['fut_bid_price'] + df['strike']
    df['pcpb_backward'] = df['call_bid_price'] - df['put_ask_price'] - df['fut_ask_price'] + df['strike']
    return df


def mark_up_with_futures(resampled_df):
    ref_sym = 'BTC-USD'

    fut_df = resampled_df.stack('symbol').loc[lambda x: (x.ref_sym == ref_sym)&(x.pc.isin(['F']))].reset_index()
    fut_df = fut_df.rename(columns={'ask_amount':'fut_ask_amount', 'ask_price':'fut_ask_price', 'bid_amount':'fut_bid_amount', 'bid_price':'fut_bid_price', 'exp':'fut_exp'})
    opt_df = resampled_df.stack('symbol').loc[lambda x: (x.ref_sym == ref_sym)&(x.pc.isin(['C','P']))].reset_index()
    opt_df= opt_df.map_update(fut_df, on=['ts','fut_sym'], col=['fut_ask_amount', 'fut_ask_price', 'fut_bid_amount', 'fut_bid_price', 'fut_exp'])
    return opt_df, fut_df


def merge_data(fr='2025-01-02', to='2025-01-02 00:10:00', freq='1min', output_tag='resampled'):
    # OPTIMIZATION: Increased chunksize as Parquet reading is more efficient.
    chunksize = 500_000
    df_iters, dayinfo = iter_datasets(fr, chunksize=chunksize)

    iter_keys = list(df_iters.keys())

    dfs = {k: pd.DataFrame() for k in iter_keys}
    stop_conditions = {k: False for k in iter_keys}
    process_to = {k: pd.Timestamp.max for k in iter_keys}

    cols_to_save = ['ref_sym','fut_sym','exp','pc', 'strike']
    cols_to_save += ['bid_price','ask_price','bid_amount','ask_amount']
    all_symbols = sorted(list(set(list(dayinfo.opt_symbols) + list(dayinfo.fut_symbols))))
    full_columns = pd.MultiIndex.from_product([cols_to_save, all_symbols], names=['field', 'symbol'])

    last_values = pd.DataFrame(np.nan, index=all_symbols, columns=cols_to_save)
    output_chunks = []
    logger.info(f'done prep')
    while True:
        # 1. Refill buffers
        for sym in iter_keys:
            if not stop_conditions[sym] and len(dfs[sym]) < chunksize:
                try:
                    chunk = next(df_iters[sym])
                    chunk = augment_quotes(chunk)
                    if dfs[sym].empty:
                        dfs[sym] = chunk
                    else:
                        dfs[sym] = pd.concat([dfs[sym], chunk], ignore_index=True)
                except StopIteration:
                    stop_conditions[sym] = True
                if not dfs[sym].empty:
                    process_to[sym] = dfs[sym].ts.max()
                logger.debug(f'done refill buffer {sym}')

        if all([df.empty for df in dfs.values()]):
            break

        # 2. Safe processing window

        cutoff = min([process_to[sym] for sym in iter_keys])
        if all([stop_conditions[sym] for sym in iter_keys]):
            cutoff = pd.Timestamp.max

        logger.info(f'done read up to cutoff {cutoff}')
        # 3. Slice data up to cutoff
        if cutoff == pd.Timestamp.max:
            df_slices = dfs
            dfs = {sym: pd.DataFrame() for sym in iter_keys}
        else:
            df_masks = {sym: dfs[sym].ts < cutoff for sym in iter_keys}
            df_slices = {sym: dfs[sym].loc[df_masks[sym]] for sym in iter_keys}
            dfs = {sym: dfs[sym].loc[~df_masks[sym]] for sym in iter_keys}

        if all([df_slices[sym].empty for sym in iter_keys]):
            logger.info('refill buffers')
            continue

        # 4. Merge and Pivot
        # Determine grid bounds from data
        min_ts = pd.Timestamp.max
        max_ts = pd.Timestamp.min
        current_data_map = {}

        for sym, df in df_slices.items():
            if df.empty: continue
            if sym == 'OPTIONS':
                for s, g in df.groupby('symbol'):
                    current_data_map[s] = g
                    min_ts = min(min_ts, g['ts'].min())
                    max_ts = max(max_ts, g['ts'].max())
            else:
                current_data_map[sym] = df
                min_ts = min(min_ts, df['ts'].min())
                max_ts = max(max_ts, df['ts'].max())

        t_start = min_ts.floor(freq)
        t_end = min(max_ts.ceil(freq), cutoff)
        grid = pd.date_range(t_start, t_end, freq=freq)

        chunk_dfs = []
        for sym in all_symbols:
            df = current_data_map.get(sym)
            last_val = last_values.loc[sym]

            dummy_ts = t_start - pd.Timedelta(nanoseconds=1)
            dummy_df = pd.DataFrame([last_val.values], index=[dummy_ts], columns=cols_to_save)

            if df is not None:
                df_dedup = df.drop_duplicates(subset=['ts'], keep='last').set_index('ts')[cols_to_save]
                combined_sym = pd.concat([dummy_df, df_dedup]).sort_index()
            else:
                combined_sym = dummy_df

            res = combined_sym.reindex(grid, method='ffill')
            last_values.loc[sym] = res.iloc[-1]

            # Restore column levels (field, symbol)
            res.columns = pd.MultiIndex.from_product([res.columns, [sym]])
            chunk_dfs.append(res)

        resampled = pd.concat(chunk_dfs, axis=1)
        resampled.index.name = 'ts'

        # Sort columns to ensure consistent CSV output
        resampled.sort_index(axis=1, inplace=True)

        # OPTIMIZATION: Batch results in memory and write once at the end.
        if not resampled.empty:
            output_chunks.append(resampled)

        if cutoff > pd.Timestamp(to):
            logger.info(f"Cutoff {cutoff} exceeds 'to' timestamp {to}. Finalizing output.")
            break

    # OPTIMIZATION: Perform a single write operation at the end.
    logger.info("All chunks processed. Concatenating and writing final output.")
    if not output_chunks:
        logger.info("No data to write.")
        return

    final_df = pd.concat(output_chunks)
    final_df.columns.names = ['field','symbol']

    # Because chunks can have overlapping grid boundaries, remove duplicate timestamps.
    final_df = final_df[~final_df.index.duplicated(keep='last')]

    fname = f'{datadir}/okex-options_{output_tag}_{fr}_{freq}.parquet'
    #    final_df.columns = [
    #        '__'.join(str(x) for x in col).strip('_')
    #       for col in final_df.columns.to_flat_index()
    #    ]
    #    final_df.to_csv(fname, mode='w', header=True, index=True)
    final_df.to_parquet(fname)
    logger.info(f"Wrote {len(final_df)} total rows to {fname}")


@dataclass(frozen=False, slots=False)
class DayInfo:
    date: str
    opt_symbols: List[str]
    fut_symbols: List[str]
    opt_quotes_fname: str
    fut_quotes_fnames: Dict[str, str]

def read_parquet_chunks(path, chunksize):
    """Helper to read a Parquet file in chunks and yield pandas DataFrames."""
    pf = pq.ParquetFile(path)
    def iter_chunks():
        for batch in pf.iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
    return iter_chunks()

def iter_datasets(fr='2025-01-02', chunksize=100000):
    dayinfo = download_multiple_csvs([fr])
    logger.info(f"Initializing iterators for {dayinfo.date}")
    opt_df_iter = read_parquet_chunks(dayinfo.opt_quotes_fname, chunksize=chunksize)

    fut_df_iters = {}
    for sym in dayinfo.fut_symbols:
        fut_df_iters[sym] = read_parquet_chunks(dayinfo.fut_quotes_fnames[sym], chunksize=chunksize)

    # just for cleaner coded
    df_iters = fut_df_iters
    df_iters['OPTIONS'] = opt_df_iter
    logger.info(f'done creating {len(df_iters)} iterators')
    return df_iters, dayinfo

def augment_quotes(df):
    # OPTIMIZATION: More robust and explicit parsing of symbol strings.
    if df.empty:
        return df

    splits = df['symbol'].str.split('-', expand=True)

    # Futures have 3 parts (e.g., BTC-USD-250331), Options have 5 (e.g., BTC-USD-250331-30000-C)
    if splits.shape[1] == 5:
        df[['_A', '_B', 'exp_str', 'strike_str', 'pc']] = splits
        df['strike'] = df.strike_str.astype(float)
    elif splits.shape[1] == 3:
        df[['_A', '_B', 'exp_str']] = splits
        df['strike'] = np.nan
        df['pc'] = 'F'
    else:
        logger.warning(f"Unexpected symbol format in chunk. Shape: {splits.shape}. Skipping augmentation.")
        # Add necessary columns if they don't exist to prevent downstream errors
        for col in ['fut_sym', 'ref_sym', 'exp', 'pc', 'strike', 'ts']:
            if col not in df: df[col] = np.nan
        return df

    df['fut_sym'] = df[['_A', '_B', 'exp_str']].agg('-'.join, axis=1)
    df['ref_sym'] = df[['_A', '_B']].agg('-'.join, axis=1)
    df['exp'] = pd.to_datetime(df['exp_str'], format='%y%m%d')
    df['ts'] = pd.to_datetime(df['local_timestamp'], unit='us')

    # Drop intermediate columns to save memory
    df.drop(columns=['_A', '_B', 'exp_str','strike_str'], inplace=True, errors='ignore')

    return df

def download_multiple_csvs(frs=['2025-01-02']):

    todo = pd.DataFrame(columns=['ex','dt','sym'], index=pd.RangeIndex(0))
    todo.loc[0] = ('okex-futures','derivative_ticker','FUTURES')
    todo.loc[1] = ('okex-options','quotes','OPTIONS')

    for fr in frs:
        to = str((pd.Timestamp(fr)+pd.Timedelta('1D')).date())
        for ex,dt,sym in todo.itertuples(index=False):
            try:
                download_single_csv(ex,dt,fr,to,sym)
            except Exception as e:
                logger.warning(e)
                pass

        futfname = download_single_csv('okex-futures','derivative_ticker',fr, to, 'FUTURES')
        fut_symbols_fname = futfname+'_syms.pkl'
        if os.path.exists(fut_symbols_fname):
            with open(fut_symbols_fname, 'rb') as f:
                fut_symbols = pickle.load(f)
        else:
            futs_df = pd.read_parquet(futfname)
            fut_symbols = futs_df.symbol.unique()
            with open(fut_symbols_fname, 'wb') as f:
                pickle.dump(fut_symbols, f)
        fut_quotes_fnames = {}
        for futsym in fut_symbols:
            fut_quotes_fnames[futsym] = download_single_csv('okex-futures','quotes',fr,fr,futsym)

        opt_quotes_fname = download_single_csv('okex-options','quotes',fr,to,'OPTIONS')
        opt_symbols_fname = opt_quotes_fname+'_syms.pkl'
        if os.path.exists(opt_symbols_fname):
            with open(opt_symbols_fname, 'rb') as f:
                opt_symbols = pickle.load(f)
        else:
            opts_df = pd.read_parquet(opt_quotes_fname)
            opt_symbols = opts_df.symbol.unique()
            with open(opt_symbols_fname, 'wb') as f:
                pickle.dump(opt_symbols, f)

        return DayInfo(fr, opt_symbols, fut_symbols, opt_quotes_fname, fut_quotes_fnames)


def download_single_csv(ex, dt, fr, to, sym):
    # OPTIMIZATION: Convert to Parquet on first download for much faster subsequent reads.
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
            #api_key=os.environ.get('TARDIS_API_KEY', None)
            api_key=os.environ.get('TD.qL-Zzmk3z6IQfxW4.OQKJd-YucJo2nAu.l4lrXPsUzs45JP6.wv7ngt7Ou-gBXJu.TWRWGtiZGHYBtff.EIbB', None)
        )
    else:
        logger.debug(f'CSV already here {csv_fname}')

    logger.info(f'Converting {csv_fname} to {pq_fname}...')
    # Read CSV in chunks and write to Parquet to handle very large files without high memory usage.
    csv_iter = pd.read_csv(csv_fname, chunksize=500_000, iterator=True,
                           dtype={
                               'exchange':'string',
                               'symbol': 'string',
                               'timestamp': 'int64',
                               'local_timestamp': 'int64',
                               'ask_amount':'float64',
                               'bid_amount':'float64',
                               'ask_price': 'float64',
                               'bid_price': 'float64',
                           })

    first_chunk = next(csv_iter)
    table = pa.Table.from_pandas(first_chunk, preserve_index=False)
    with pq.ParquetWriter(pq_fname, table.schema) as writer:
        writer.write_table(table)
        for chunk in csv_iter:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            writer.write_table(table)
    logger.info(f'Finished converting to {pq_fname}')
    return pq_fname

if __name__ == '__main__':
    merge_data(fr='2025-01-02', to='2025-01-03', freq='1min', output_tag='1min')
