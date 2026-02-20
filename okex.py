import logging
import warnings
warnings.simplefilter('ignore')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import re
import os
import sys
from pathlib import Path

from pprint import pprint
import requests
import itertools
import polars as pl
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pv


from dataclasses import dataclass, field
from typing import List, Dict

from TardisProject.lib import pandas_utils as pu
from TardisProject.lib import utils

from tardis_client import TardisClient, Channel
from tardis_dev import datasets

datadir = 'okex'

# Use microsecond resolution for all timestamps
TIMESTAMP_UNIT = 'us'

def align_calls_puts(opt_df):
    pandas_interface = False
    if isinstance(opt_df, pd.DataFrame):
        pandas_interface = True
        opt_df = pl.from_pandas(opt_df)
    # Separate calls and puts
    calls = opt_df.filter(pl.col('pc') == 'C').clone()
    puts = opt_df.filter(pl.col('pc') == 'P').clone()

    # Rename option specific columns
    rename_cols = ['ask_amount', 'ask_price', 'bid_amount', 'bid_price', 'symbol', 'stale']
    calls = calls.rename({c: f'call_{c}' for c in rename_cols})
    puts = puts.rename({c: f'put_{c}' for c in rename_cols})

    # Drop shared columns from puts to avoid duplication
    shared_fut_cols = ['fut_ask_amount', 'fut_ask_price', 'fut_bid_amount', 'fut_bid_price', 'fut_exp', 'fut_stale',
                       'spot_ask_amount', 'spot_ask_price', 'spot_bid_amount', 'spot_bid_price', 'spot_stale']
    calls = calls.drop('pc')
    puts = puts.drop(['pc'] + shared_fut_cols, strict=False)

    # Merge on keys
    on_keys = ['ts', 'ref_sym', 'fut_sym', 'exp', 'strike']
    res =  calls.join(puts, on=on_keys, how='inner')
    if pandas_interface:
        return res.to_pandas()
    return res

def pcp_breaking_pandas(
    opt_df,
    cost_per_notional=0.003,
    abs_cost=0,
    fut_mgn_rate=0.2,
    short_put_mgn_rate=0.5,
    short_call_mgn_rate=0.5,
):
    assert isinstance(opt_df, pd.DataFrame)
    df = opt_df.loc[~opt_df.fut_exp.isna()]

    df['index'] = (df['spot_ask_price'] + df['spot_bid_price']) / 2
    df['call_ask_price_xS'] = df['call_ask_price'] * df['spot_ask_price']
    df['call_bid_price_xS'] = df['call_bid_price'] * df['spot_bid_price']
    df['put_ask_price_xS'] = df['put_ask_price'] * df['spot_ask_price']
    df['put_bid_price_xS'] = df['put_bid_price'] * df['spot_bid_price']

    r = 0.05
    contract_size = 0.01
    df['tte'] = (df['exp'] - df['ts']) / pd.Timedelta('365 days')
    dscnt = np.exp(-r * df['tte'])

    df['pcpb_forward'] = ((
        df['call_bid_price_xS']
        - df['put_ask_price_xS']
        - (df['fut_ask_price'] - df['strike']) * dscnt
    ) * contract_size)

    df['pcpb_backward'] = (-1 * (
        df['call_ask_price_xS']
        - df['put_bid_price_xS']
        - (df['fut_bid_price'] - df['strike']) * dscnt
    ) * contract_size)

    cost = df['index'].abs() * cost_per_notional * contract_size

    df['cost'] = cost

    df['capital_fwd'] = (
        df['index'] * fut_mgn_rate
        + df['index'] * short_call_mgn_rate
        + df['call_bid_price_xS']
        - df['put_ask_price_xS']
    ).clip(lower=0) * contract_size
    df['capital_bck'] = (
        df['index'] * fut_mgn_rate
        + df['strike'] * short_put_mgn_rate
        + df['put_bid_price_xS']
        - df['call_ask_price_xS']
    ).clip(lower=0) * contract_size

    df['pcpb_fwd_real'] = (df['pcpb_forward'] - cost).clip(lower=0)
    df['pcpb_bck_real'] = (df['pcpb_backward'] - cost).clip(lower=0)

    df['pcpb_fwd_real_rel'] = df['pcpb_fwd_real'] / df['capital_fwd']
    df['pcpb_bck_real_rel'] = df['pcpb_bck_real'] / df['capital_bck']

    df['pcpb_fwd_rel'] = (df['pcpb_forward'] / df['capital_fwd']).clip(lower=0)
    df['pcpb_bck_rel'] = (df['pcpb_backward'] / df['capital_bck']).clip(lower=0)

    df['pcpb_fwd_ann'] = df['pcpb_fwd_rel'] / df['tte']
    df['pcpb_bck_ann'] = df['pcpb_bck_rel'] / df['tte']

    df['pcpb_fwd_real_ann'] = df['pcpb_fwd_real_rel'] / df['tte']
    df['pcpb_bck_real_ann'] = df['pcpb_bck_real_rel'] / df['tte']

    df['put_opt_spread_bp'] = (df['put_ask_price_xS'] - df['put_bid_price_xS']) / (contract_size * df['index']) * 10000
    df['call_opt_spread_bp'] = (df['call_ask_price_xS'] - df['call_bid_price_xS']) / (contract_size * df['index']) * 1000
    df['bigger_opt_spread_bp'] = np.maximum(df.put_opt_spread_bp, df.call_opt_spread_bp)

    df['amu_bp'] = ((np.maximum(df['pcpb_forward'], df['pcpb_backward'])- cost)
                    / (contract_size * df['index']) * 10000 + df['bigger_opt_spread_bp'])

    return df

def pcp_breaking_polars(
    opt_df,
    cost_per_notional=0.003,
    abs_cost=0,
    fut_mgn_rate=0.2,
    short_put_mgn_rate=0.5,
    short_call_mgn_rate=0.5,
):
    pandas_interface = False
    if isinstance(opt_df, pd.DataFrame):
        pandas_interface = True
        opt_df = pl.from_pandas(opt_df)

    df = opt_df.filter(pl.col('fut_exp').is_not_null())

    spot_ask_price = pl.col('spot_ask_price')
    spot_bid_price = pl.col('spot_bid_price')
    call_ask_price = pl.col('call_ask_price')
    call_bid_price = pl.col('call_bid_price')
    put_ask_price = pl.col('put_ask_price')
    put_bid_price = pl.col('put_bid_price')
    fut_ask_price = pl.col('fut_ask_price')
    fut_bid_price = pl.col('fut_bid_price')
    strike = pl.col('strike')

    index = (spot_ask_price + spot_bid_price) / 2
    call_ask_price_xS = (call_ask_price * spot_ask_price).alias('call_ask_price_xS')
    call_bid_price_xS = (call_bid_price * spot_bid_price).alias('call_bid_price_xS')
    put_ask_price_xS = (put_ask_price * spot_ask_price).alias('put_ask_price_xS')
    put_bid_price_xS = (put_bid_price * spot_bid_price).alias('put_bid_price_xS')

    r = 0.05
    contract_size = 0.01
    tte = ((pl.col('exp') - pl.col('ts')) / pl.duration(days=365)).alias('tte')
    dscnt = (-(r) * tte).exp()

    pcpb_forward = (
        (call_bid_price_xS - put_ask_price_xS - (fut_ask_price - strike) * dscnt)
        * contract_size
    ).clip(lower_bound=0).alias('pcpb_forward')

    pcpb_backward = (
        (call_ask_price_xS - put_bid_price_xS - (fut_bid_price - strike) * dscnt)
        * contract_size
        * -1
    ).clip(lower_bound=0).alias('pcpb_backward')

    cost = (index.abs() * cost_per_notional * contract_size).alias('cost')

    capital_fwd = (
        (index * fut_mgn_rate)
        + (index * short_call_mgn_rate)
        + (call_bid_price_xS - put_ask_price_xS)
    ).clip(lower_bound=0) * contract_size

    capital_bck = (
        (index * fut_mgn_rate)
        + (strike * short_put_mgn_rate)
        + (put_bid_price_xS - call_ask_price_xS)
    ).clip(lower_bound=0) * contract_size

    def _min0(pl_col):
        return pl.when(pl_col > 0).then(pl_col).otherwise(0)

    pcpb_fwd_real = _min0(pcpb_forward - cost).alias('pcpb_fwd_real')
    pcpb_bck_real = _min0(pcpb_backward - cost).alias('pcpb_bck_real')

    pcpb_fwd_real_rel = (pcpb_fwd_real / capital_fwd).alias('pcpb_fwd_real_rel')
    pcpb_bck_real_rel = (pcpb_bck_real / capital_bck).alias('pcpb_bck_real_rel')

    pcpb_fwd_rel = (pcpb_forward / capital_fwd).alias('pcpb_fwd_rel')
    pcpb_bck_rel = (pcpb_backward / capital_bck).alias('pcpb_bck_rel')

    pcpb_fwd_ann = (pcpb_fwd_rel / tte).alias('pcpb_fwd_ann')
    pcpb_bck_ann = (pcpb_bck_rel / tte).alias('pcpb_bck_ann')

    pcpb_fwd_real_ann = (pcpb_fwd_real_rel / tte).alias('pcpb_fwd_real_ann')
    pcpb_bck_real_ann = (pcpb_bck_real_rel / tte).alias('pcpb_bck_real_ann')

    bigger_opt_spread = (
        pl.max_horizontal(
            (put_ask_price_xS - put_bid_price_xS),
            (call_ask_price_xS - call_bid_price_xS),
        )
        * contract_size
    ).alias('bigger_opt_spread')

    amu = (pl.max_horizontal(pcpb_forward, pcpb_backward) + bigger_opt_spread).alias('amu')

    df = df.with_columns(
        [
            index.alias('index'),
            call_ask_price_xS,
            call_bid_price_xS,
            put_ask_price_xS,
            put_bid_price_xS,
            tte,
            pcpb_forward,
            pcpb_backward,
            cost,
            capital_fwd.alias('capital_fwd'),
            capital_bck.alias('capital_bck'),
            pcpb_fwd_real,
            pcpb_bck_real,
            pcpb_fwd_real_rel,
            pcpb_bck_real_rel,
            pcpb_fwd_rel,
            pcpb_bck_rel,
            pcpb_fwd_ann,
            pcpb_bck_ann,
            pcpb_fwd_real_ann,
            pcpb_bck_real_ann,
            bigger_opt_spread,
            amu,
        ]
    )

    if pandas_interface:
        return df.to_pandas()
    return df





def mark_up_with_futures(resampled_df, ref_syms=['BTC-USD','ETH-USD']):
    fut_dfs = []
    opt_dfs = []
    for ref_sym  in ref_syms:
        fut_df = resampled_df.filter(
            (pl.col('ref_sym') == ref_sym) & (pl.col('pc') == 'F')
        ).rename({
            'ask_amount': 'fut_ask_amount',
            'ask_price': 'fut_ask_price',
            'bid_amount': 'fut_bid_amount',
            'bid_price': 'fut_bid_price',
            'stale': 'fut_stale',
            'exp': 'fut_exp'
        })

        spot_df = resampled_df.filter(
            (pl.col('ref_sym') == ref_sym) & (pl.col('pc') == 'S')
        ).rename({
            'ask_amount': 'spot_ask_amount',
            'ask_price': 'spot_ask_price',
            'bid_amount': 'spot_bid_amount',
            'bid_price': 'spot_bid_price',
            'stale': 'spot_stale',
        })

        fut_df = fut_df.join(
            spot_df.select(['ts', 'spot_sym', 'spot_ask_amount', 'spot_ask_price', 'spot_bid_amount', 'spot_bid_price', 'spot_stale']),
            on=['ts', 'spot_sym'],
            how='left'
        )

        opt_df = resampled_df.filter(
            (pl.col('ref_sym') == ref_sym) & (pl.col('pc').is_in(['C', 'P']))
        )

        opt_df = opt_df.join(
            fut_df.select(['ts', 'fut_sym', 'fut_ask_amount', 'fut_ask_price', 'fut_bid_amount', 'fut_bid_price', 'fut_exp', 'fut_stale']),
            on=['ts', 'fut_sym'],
            how='left'
        )

        opt_df = opt_df.join(
            spot_df.select(['ts', 'spot_sym', 'spot_ask_amount', 'spot_ask_price', 'spot_bid_amount', 'spot_bid_price', 'spot_stale']),
            on=['ts', 'spot_sym'],
            how='left'
        )
        fut_dfs.append(fut_df)
        opt_dfs.append(opt_df)

    opt_all = pl.concat(opt_dfs) if opt_dfs else pl.DataFrame()
    fut_all = pl.concat(fut_dfs) if fut_dfs else pl.DataFrame()

    return opt_all, fut_all


def merge_data(fr,  to=None, freq='1min', output_tag='resampled', force_reload=False, cleanup=False):
    '''
    merge_data only deals with a single day (fr), the to argument only serves to
    cut the day sort for testing
    '''
    fr = pd.Timestamp(fr)
    if to is None:
        to = fr + pd.Timedelta('1D')
    to = pd.Timestamp(to)
    if pd.Timestamp(fr.date()) == fr:
        fr = f'{fr.date()}'
    else:
        fr = f'{fr}'
    if pd.Timestamp(to.date()) == to:
        to = f'{to.date()}'
    else:
        to = f'{to}'

    fname = f'{datadir}/okex-options_{output_tag}_{fr}_{freq}.parquet'
    if not force_reload and os.path.exists(fname):
        return pl.read_parquet(fname)
    chunksize = 1_500_000
    df_iters, dayinfo = iter_datasets(fr, chunksize=chunksize, cleanup=cleanup)

    iter_keys = list(df_iters.keys())

    dfs = {k: pl.DataFrame() for k in iter_keys}
    stop_conditions = {k: False for k in iter_keys}
    process_to = {k: datetime.max for k in iter_keys}

    cols_to_save = ['ref_sym', 'fut_sym', 'spot_sym', 'exp', 'pc', 'strike']
    cols_to_save += ['bid_price', 'ask_price', 'bid_amount', 'ask_amount']
    all_symbols = sorted(list(set(list(dayinfo.opt_symbols) + list(dayinfo.fut_symbols) + ['BTC-USDT','ETH-USDT'])))

    last_values = {sym: {col: None for col in cols_to_save} for sym in all_symbols}
    output_chunks = []
    logger.info(f'done prep')

    to_timestamp = datetime.max if to == 'max' else datetime.fromisoformat(to)

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
            cutoff = datetime.max

        logger.info(f'done read up to cutoff {cutoff}')

        # 3. Slice data up to cutoff
        if cutoff == datetime.max:
            df_slices = dfs
            dfs = {sym: pl.DataFrame() for sym in iter_keys}
        else:
            df_slices = {sym: dfs[sym].filter(pl.col('ts') < cutoff) for sym in iter_keys}
            dfs = {sym: dfs[sym].filter(pl.col('ts') >= cutoff) for sym in iter_keys}

        if all([df_slices[sym].is_empty() for sym in iter_keys]):
            logger.info('ran out of data for the day before the end of the day')
            break

        # 4. Merge and Pivot
        min_ts = None
        max_ts = None
        current_data_map = {}

        for sym, df in df_slices.items():
            if df.is_empty():
                continue
            if sym == 'OPTIONS':
                for ss, g in df.partition_by('symbol', as_dict=True).items():
                    s = ss[0] # apparently the key here is a tuple
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

        # Floor/ceil to frequency using pandas for precision
        t_start = pd.Timestamp(min_ts).floor(freq)
        t_end = min(pd.Timestamp(max_ts).ceil(freq), cutoff if cutoff != datetime.max else max_ts)

        # Create time grid with microsecond precision
        grid = (
            pl.Series(pd.date_range(t_start, t_end, freq=freq))
            .cast(pl.Datetime(TIMESTAMP_UNIT))
        )
        chunk_dfs = []
        schema = {
            "ts": pl.Datetime("us"),
            "symbol": pl.String,
            "ref_sym": pl.String,
            "fut_sym": pl.String,
            "spot_sym": pl.String,
            "exp": pl.Datetime("us"),
            "pc": pl.String,
            "strike": pl.Float64,
            "bid_price": pl.Float64,
            "ask_price": pl.Float64,
            "bid_amount": pl.Float64,
            "ask_amount": pl.Float64,
            'stale': pl.Boolean,
        }

        for sym in all_symbols:
            df = current_data_map.get(sym)
            last_val = last_values[sym]

            if df is not None and not df.is_empty():
                df_dedup = df.unique(subset=['ts'], keep='last').select(['ts']+cols_to_save).sort('ts')
                # Forward fill
                df_filled = pl.DataFrame({'ts':grid}).join_asof(
                    df_dedup,
                    on='ts',
                    strategy='backward'
                )
            else:
                df_filled = pl.DataFrame({'ts':grid}).with_columns([
                    pl.lit(None, dtype=schema[col]).alias(col) for col in cols_to_save
                ])

            # Update last values
            if not df_filled.is_empty():
                last_row = df_filled.tail(1).to_dicts()[0]
                for col in cols_to_save:
                    if last_row.get(col) is not None:
                        last_values[sym][col] = last_row[col]

            df_filled = df_filled.with_columns(pl.lit(sym).alias('symbol'))
            df_filled = df_filled.select(['ts','symbol'] + cols_to_save) #reorder columns
            chunk_dfs.append(df_filled)

        for i, d in enumerate(chunk_dfs):
            if i == 0:
                base = d.schema
            else:
                if d.schema != base:
                    print("Schema mismatch at", i)
                    print("base:", base)
                    print("this:", d.schema)
                    break


        resampled = pl.concat(chunk_dfs)
        resampled = resampled.sort('ts')

        if not resampled.is_empty():
            output_chunks.append(resampled)

        if cutoff > to_timestamp:
            logger.info(f"Cutoff {cutoff} exceeds 'to' timestamp {to}. Finalizing output.")
            break

    logger.info("All chunks processed. Concatenating and writing final output.")
    if not output_chunks:
        logger.info("No data to write.")
        return

    final_df = pl.concat(output_chunks)
    final_df = final_df.unique(subset=['ts', 'symbol'], keep='last')

    # Ensure all datetime columns use microsecond resolution
    final_df = final_df.with_columns([
        pl.col('ts').cast(pl.Datetime(TIMESTAMP_UNIT)),
        pl.col('exp').cast(pl.Datetime(TIMESTAMP_UNIT))
    ])

    # extend static metadata over all rows for each symbol, even
    # when there was no quote then.

    static_cols = ["ref_sym", "fut_sym", "spot_sym", "exp", "pc", "strike"]

    final_df = final_df.with_columns(
        [
            pl.col(c)
            .fill_null(pl.col(c).drop_nulls().first().over("symbol"))
            .alias(c)
            for c in static_cols
        ]
    )

    check_consistent  = final_df.group_by("symbol").agg(
        [pl.col(c).n_unique().alias(f"{c}_n_unique") for c in static_cols]
    ).filter(
        pl.any_horizontal([pl.col(f"{c}_n_unique") > 1 for c in static_cols])
    )
    if len(check_consistent):
        raise RuntimeError('inconsistent metadata')


    # forward fill price data
    price_cols = ["bid_price", "ask_price", "bid_amount", "ask_amount"]
    final_df = final_df.sort(["symbol", "ts"])

    final_df = final_df.with_columns(
        (
            pl.col("bid_price").is_null() | pl.col("ask_price").is_null()
        ).alias("stale")
    )

    final_df = final_df.with_columns(
        [
            pl.col(c)
            .fill_null(strategy="forward")
            .over("symbol")
            .alias(c)
            for c in price_cols
        ]
    )

    if cleanup:
        cleanup_daily_files(dayinfo)

    final_df.write_parquet(fname)
    logger.info(f"Wrote {len(final_df)} total rows to {fname}")
    return final_df

def cleanup_daily_files(dayinfo):
    def _corresponding_csv(fn):
        res = fn.replace('.parquet', '.csv.gz')
        return res
    fnames = [dayinfo.opt_quotes_fname, dayinfo.btc_quotes_fname, dayinfo.eth_quotes_fname,
              dayinfo.opt_symbols_fname, dayinfo.fut_symbols_fname, dayinfo.fut_derivative_ticker_fname]
    fnames += dayinfo.fut_quotes_fnames.values()
    fnames += [_corresponding_csv(fn) for fn in fnames]
    for fn in fnames:
        Path(fn).unlink(missing_ok=True)
    return

@dataclass(frozen=False, slots=False)
class DayInfo:
    date: str
    opt_symbols: List[str]
    fut_symbols: List[str]
    opt_quotes_fname: str
    fut_quotes_fnames: Dict[str, str]
    btc_quotes_fname: str
    eth_quotes_fname: str
    opt_symbols_fname: str
    fut_symbols_fname: str
    fut_derivative_ticker_fname: str


def read_parquet_chunks(path, chunksize):
    """Helper to read a Parquet file in chunks and yield Polars DataFrames."""
    def iter_chunks():
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunksize):
            yield pl.from_arrow(batch)
    return iter_chunks()


def iter_datasets(fr='2025-01-02', chunksize=100000, cleanup=False):
    dayinfo = download_multiple_csvs(fr, cleanup=cleanup)
    logger.info(f"Initializing iterators for {dayinfo.date}")
    opt_df_iter = read_parquet_chunks(dayinfo.opt_quotes_fname, chunksize=chunksize)

    fut_df_iters = {}
    for sym in dayinfo.fut_symbols:
        fut_df_iters[sym] = read_parquet_chunks(dayinfo.fut_quotes_fnames[sym], chunksize=chunksize)

    df_iters = fut_df_iters
    df_iters['OPTIONS'] = opt_df_iter

    df_iters['BTC-USDT'] = read_parquet_chunks(dayinfo.btc_quotes_fname, chunksize=chunksize)
    df_iters['ETH-USDT'] = read_parquet_chunks(dayinfo.eth_quotes_fname, chunksize=chunksize)

    logger.info(f'done creating {len(df_iters)} iterators')
    return df_iters, dayinfo


def augment_quotes(df: pl.DataFrame) -> pl.DataFrame:
    """Augment quotes dataframe with parsed symbol information.

    Expected symbol formats:
      - Futures:  BTC-USD-250331
      - Options:  BTC-USD-250331-30000-C
      - Spot: BTC-USD
    local_timestamp assumed to be in microseconds since epoch.
    """
    if df.is_empty():
        return df

    parts = pl.col("symbol").str.split("-")

    df = df.with_columns(
        [
            parts.list.get(0).alias("_A"),
            parts.list.get(1).alias("_B"),
            parts.list.get(2, null_on_oob=True).alias("exp_str"),
            parts.list.get(3, null_on_oob=True).alias("_D"),  # strike (options) or null (futures)
            parts.list.get(4, null_on_oob=True).alias("_E"),  # P/C (options) or null (futures)
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("_D").is_not_null())
            .then(pl.col("_D").cast(pl.Float64, strict=False))
            # strike: only present for options; parse safely
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("strike"),

            # pc: use 'F' for futures, S for spot
            pl.when(pl.col('exp_str').is_null())
            .then(pl.lit('S'))
            .when(pl.col('_E').is_null())
            .then(pl.lit('F'))
            .otherwise(pl.col('_E'))
            .alias("pc"),

            # for options on XXXUSD the right hedge instrument is the XXXUSDT future
            pl.when((pl.col("_B").is_in(["USD","USDT"])&(pl.col("_E").is_in(['C','P']) )))
            .then(pl.lit("USDT"))
            .otherwise(pl.col("_B"))
            .alias("_BT"),

            pl.when((pl.col("_B").is_in(["USD","USDT"]))&(pl.col("_E").is_in(['F','S']) ))
            .then(pl.lit("USD"))
            .otherwise(pl.lit("USD"))
            .alias("_BB"),
        ]
    )

    df = df.with_columns(
        [
            pl.concat_str([pl.col("_A"), pl.col("_BT"), pl.col("exp_str")], separator="-").alias("fut_sym"),

            pl.concat_str([pl.col("_A"), pl.col("_BT")], separator="-").alias("spot_sym"),

            pl.concat_str([pl.col("_A"), pl.col("_BB")], separator="-").alias("ref_sym"),

            # exp_str like '250331' -> Date -> Datetime (midnight, microsecond precision)
            pl.col("exp_str")
              .str.strptime(pl.Date, format="%y%m%d", strict=False)
              .cast(pl.Datetime(TIMESTAMP_UNIT))
              .alias("exp"),

            # local_timestamp microseconds since epoch -> Datetime(us)
            pl.from_epoch(pl.col("local_timestamp"), time_unit=TIMESTAMP_UNIT).alias("ts"),
        ]
    )

    df = df.drop(["_A", "_B", "_BT", "_BB", "exp_str", "_D", "_E"], strict=False)

    #print('augment quotes\n',df.to_pandas())
    return df



def download_multiple_csvs(fr='2025-01-02', cleanup=False):
    todo = pl.DataFrame({
        'ex': ['okex-futures', 'okex-options','okex','okex'],
        'dt': ['derivative_ticker', 'quotes', 'quotes','quotes'],
        'sym': ['FUTURES', 'OPTIONS','BTC-USDT','ETH-USDT']
    })

    to = str((pd.Timestamp(fr) + pd.Timedelta(days=1)).date())
    for row in todo.iter_rows(named=True):
        ex, dt, sym = row['ex'], row['dt'], row['sym']
        try:
            fname = download_single_csv(ex, dt, fr, to, sym, cleanup=cleanup)
            if sym=='BTC-USDT':
                btc_quotes_fname = fname
            if sym=='ETH-USDT':
                eth_quotes_fname = fname
        except Exception as e:
            logger.warning(e)
            pass

    fut_derivative_ticker_fname = download_single_csv('okex-futures', 'derivative_ticker', fr, to, 'FUTURES', cleanup=cleanup)
    fut_symbols_fname = fut_derivative_ticker_fname + '_syms.pkl'
    if os.path.exists(fut_symbols_fname):
        with open(fut_symbols_fname, 'rb') as f:
            fut_symbols = pickle.load(f)
    else:
        fut_symbols = (
            pl.scan_parquet(fut_derivative_ticker_fname)
              .select(pl.col("symbol"))
              .unique()
              .collect(streaming=True)     # keep memory down
              .get_column("symbol")
              .to_list()
        )

        with open(fut_symbols_fname, 'wb') as f:
            pickle.dump(fut_symbols, f)

    fut_quotes_fnames = {}
    for futsym in fut_symbols:
        fut_quotes_fnames[futsym] = download_single_csv('okex-futures', 'quotes', fr, fr, futsym, cleanup=cleanup)

    opt_quotes_fname = download_single_csv('okex-options', 'quotes', fr, to, 'OPTIONS', cleanup=cleanup)
    opt_symbols_fname = opt_quotes_fname + '_syms.pkl'
    if os.path.exists(opt_symbols_fname):
        with open(opt_symbols_fname, 'rb') as f:
            opt_symbols = pickle.load(f)
    else:
        opt_symbols = (
            pl.scan_parquet(opt_quotes_fname)
              .select(pl.col("symbol"))

              .unique()
              .collect(streaming=True)     # keep memory down
              .get_column("symbol")
              .to_list()
        )
        with open(opt_symbols_fname, 'wb') as f:
            pickle.dump(opt_symbols, f)

    return DayInfo(fr, opt_symbols, fut_symbols, opt_quotes_fname,
                   fut_quotes_fnames, btc_quotes_fname, eth_quotes_fname,
                   opt_symbols_fname, fut_symbols_fname, fut_derivative_ticker_fname)


def download_single_csv(ex, dt, fr, to, sym, cleanup=False):
    csv_fname = f'{datadir}/{ex}_{dt}_{fr}_{sym}.csv.gz'
    pq_fname  = f'{datadir}/{ex}_{dt}_{fr}_{sym}.parquet'

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
            api_key=os.environ.get('TARDIS_API_KEY', None),
            download_dir=datadir
        )

    logger.info(f"Streaming convert {csv_fname} to {pq_fname}")

    convert_opts = pv.ConvertOptions(
        column_types={
            "timestamp": pa.int64(),
            "local_timestamp": pa.int64(),
            "bid_price": pa.float64(),
            "ask_price": pa.float64(),
            "bid_amount": pa.float64(),
            "ask_amount": pa.float64(),
            "symbol": pa.string(),
            "exchange": pa.string(),
        },
        null_values=["", "null", "None"],
    )

    # Smaller block_size reduces peak memory; increase if you want bigger row groups.
    read_opts = pv.ReadOptions(block_size=512 * 1024 * 1024)  # 0.5 G

    reader = pv.open_csv(
        csv_fname,
        read_options=read_opts,
        convert_options=convert_opts,
    )

    writer = None
    batches = []
    target_batches = 8  # combine batches into larger row groups

    try:
        for batch in reader:  # batch: pyarrow.RecordBatch
            batches.append(batch)

            if len(batches) >= target_batches:
                table = pa.Table.from_batches(batches)
                if writer is None:
                    writer = pq.ParquetWriter(pq_fname, table.schema, compression="snappy")
                writer.write_table(table)
                batches.clear()

        # flush remainder
        if batches:
            table = pa.Table.from_batches(batches)
            if writer is None:
                writer = pq.ParquetWriter(pq_fname, table.schema, compression="snappy")
            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()

    logger.info(f"Finished streaming convert to {pq_fname}")
    if cleanup:
        Path(csv_fname).unlink(missing_ok=True)
        logger.info(f"Cleaned up {csv_fname}")
    return pq_fname

# def download_single_csv(ex, dt, fr, to, sym):
#     csv_fname = f'{datadir}/{ex}_{dt}_{fr}_{sym}.csv.gz'
#     pq_fname = f'{datadir}/{ex}_{dt}_{fr}_{sym}.parquet'

#     if os.path.exists(pq_fname):
#         logger.debug(f'Parquet file already exists: {pq_fname}')
#         return pq_fname

#     if not os.path.exists(pq_fname):

#         logger.info(f'Downloading {csv_fname}')
#         datasets.download(
#             exchange=ex,
#             data_types=[dt],
#             from_date=fr,
#             to_date=to,
#             symbols=[sym],
#             api_key=os.environ.get('TARDIS_API_KEY', None),
#             download_dir=datadir
#         )
#         logger.info(f'Converting {csv_fname} to {pq_fname}...')

#         df = pl.read_csv(csv_fname, dtypes={
#             "timestamp": pl.Int64,
#             "local_timestamp": pl.Int64,
#             "bid_price": pl.Float64,
#             "ask_price": pl.Float64,
#             "bid_amount": pl.Float64,
#             "ask_amount": pl.Float64,
#             "symbol": pl.String,
#             "exchange": pl.String,
#         })
#         df.write_parquet(pq_fname)
#         logger.info(f'Finished converting to {pq_fname}')
#     else:
#         logger.debug(f'parquet file  already here {pq_fname}')

#     return pq_fname

def explore(day='2026-01-01'):
    # read the prepared data
    resampled_df = merge_data(day, freq='1min', output_tag='test',
                              force_reload=False, cleanup=False)
    opt_df, fut_df = mark_up_with_futures(resampled_df)
    opt_df_aligned = align_calls_puts(opt_df)
    return opt_df_aligned.to_pandas(), fut_df.to_pandas()


def main(fr='2026-01-01', to=None, cleanup=False, output_tag='test', force_reload=False):
    if to is None:
        to = fr
    pcpbs = []
    for day in pd.date_range(fr, to, freq='1D'):

        resampled_df = merge_data(day, freq='1min', output_tag='prod',
                        force_reload=force_reload, cleanup=cleanup)
        opt_df, fut_df = mark_up_with_futures(resampled_df)
        opt_df_aligned = align_calls_puts(opt_df)
        pcpb = pcp_breaking(opt_df_aligned)
        pcpb.write_parquet(f'{datadir}/pcpb_{output_tag}_{day.date()}.parquet')
        #print(pcpb.to_pandas())
        pcpbs.append(pcpb)
    if len(pcpbs) == 1:
        return pcpbs[0]
    return pcpbs

if __name__ == '__main__':
    if len(sys.argv)>1:
        day = sys.argv[1]
    else:
        day = '2025-01-01'
    if len(sys.argv)>2:
        to = sys.argv[2]
    else:
        to = None

    print(day)
    main(day, to, cleanup=False, output_tag='test')
