import logging
import os
import json
import requests
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tardis_dev import datasets
import glob

# 1. Initialize Environment
load_dotenv()
API_KEY = os.environ.get("TARDIS_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATADIR = 'deribit_data'
os.makedirs(DATADIR, exist_ok=True)

def get_deribit_availability():
    """Fetches and normalizes metadata for timezone-naive comparison."""
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    resp = requests.get("https://api.tardis.dev/v1/exchanges/deribit", headers=headers)
    if resp.status_code != 200: return {}
    
    availability = {}
    for s in resp.json().get('availableSymbols', []):
        symbol_id = s.get('id')
        if not symbol_id: continue
        # Normalize to naive for logic consistency
        since = pd.to_datetime(s.get('availableSince')).tz_localize(None)
        to_raw = s.get('availableTo')
        to = pd.to_datetime(to_raw).tz_localize(None) if to_raw else pd.Timestamp.max
        availability[symbol_id] = {'since': since, 'to': to}
    return availability

def convert_to_parquet(csv_files, pq_fname):
    """
    Unified conversion that maps 'best_ask_price' to 'ask_price'.
    This ensures Futures and Options data align on the same schema.
    """
    writer = None
    final_cols = ['ts', 'symbol', 'ask_price', 'bid_price', 'local_timestamp']
    # Tardis derivative_ticker uses 'best_' prefix for top-of-book
    potential_prices = ['ask_price', 'bid_price', 'best_ask_price', 'best_bid_price']

    for csv_file in csv_files:
        if not os.path.exists(csv_file): continue
        logger.info(f"Processing {csv_file}...")
        
        csv_iter = pd.read_csv(csv_file, chunksize=1_000_000, iterator=True,
                               usecols=lambda c: c in ['symbol', 'local_timestamp', 'timestamp'] or c in potential_prices)
        
        for chunk in csv_iter:
            # Normalize column names across different Tardis datasets
            chunk = chunk.rename(columns={'best_ask_price': 'ask_price', 'best_bid_price': 'bid_price'})
            if 'local_timestamp' not in chunk.columns and 'timestamp' in chunk.columns:
                chunk['local_timestamp'] = chunk['timestamp']
            
            chunk['ts'] = pd.to_datetime(chunk['local_timestamp'], unit='us')
            
            # Ensure Parquet schema is identical for every row/chunk
            for col in final_cols:
                if col not in chunk.columns: chunk[col] = np.nan
            
            table = pa.Table.from_pandas(chunk[final_cols], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(pq_fname, table.schema)
            writer.write_table(table)
            
    if writer: writer.close()
    logger.info(f"Finalized unified Parquet: {pq_fname}")

def run_deribit_attack(date_str='2025-12-05', freq='1min'):
    target_dt = pd.to_datetime(date_str).tz_localize(None)
    # 10-minute warm-up buffer prevents nulls at the 00:00 start
    start_time_str = (target_dt - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
    end_date_str = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    with open("deribit_pcp_targets.json", "r") as f:
        targets = json.load(f)

    availability = get_deribit_availability()
    active_targets = []
    needed_symbols = set()
    for t in targets:
        legs = [t['call_symbol'], t['put_symbol'], t['future_symbol']]
        # Use availability check to avoid requesting non-existent data
        if all(availability.get(s) and availability[s]['since'] <= target_dt <= availability[s]['to'] for s in legs):
            active_targets.append(t)
            needed_symbols.update(legs)

    if not active_targets:
        logger.warning(f"No active triplets for {date_str}.")
        return

    # C. Acquisition - Use Bulk Keywords for efficiency
    pq_path = f"{DATADIR}/deribit_{date_str}_combined.parquet"    
    if not os.path.exists(pq_path):
        logger.info(f"Downloading market data with warm-up buffer...")
        datasets.download(exchange="deribit", data_types=["quotes"], 
                          from_date=start_time_str, to_date=end_date_str, 
                          symbols=["OPTIONS", "FUTURES"], download_dir=DATADIR, api_key=API_KEY)        
        all_csvs = glob.glob(f"{DATADIR}/deribit_quotes_*.csv.gz")
        convert_to_parquet(all_csvs, pq_path)

    # D. Resampling logic
    raw_df = pd.read_parquet(pq_path)
    grid = pd.date_range(target_dt, target_dt + pd.Timedelta(days=1), freq=freq, inclusive='left')
    symbol_data = {}
    grouped = raw_df.groupby('symbol')
    
    for sym in needed_symbols:
        # Standardize matching: some exchanges use different casing/suffixes
        if sym in grouped.groups:
            df_s = grouped.get_group(sym).drop_duplicates('ts', keep='last').set_index('ts')
            # Forward-fill ensures we use the latest quote on the synchronized grid
            symbol_data[sym] = df_s[['ask_price', 'bid_price']].reindex(grid, method='ffill')

    # E. Calculate Violations
    final_chunks = []
    for t in active_targets:
        c, p, f = t['call_symbol'], t['put_symbol'], t['future_symbol']
        if all(s in symbol_data for s in [c, p, f]):
            merged = pd.concat([
                symbol_data[c].rename(columns={'ask_price': 'c_ask', 'bid_price': 'c_bid'}),
                symbol_data[p].rename(columns={'ask_price': 'p_ask', 'bid_price': 'p_bid'}),
                symbol_data[f].rename(columns={'ask_price': 'f_ask', 'bid_price': 'f_bid'})
            ], axis=1).reset_index().rename(columns={'index': 'ts'})
            
            merged['strike'] = float(t['strike'])
            # PCP Breaking Formula standardized to USD arbitrage value
            merged['pcpb_forward'] = (merged['c_ask'] - merged['p_bid']) * merged['f_bid'] - (merged['f_bid'] - merged['strike'])
            merged['pcpb_backward'] = (merged['c_bid'] - merged['p_ask']) * merged['f_ask'] - (merged['f_ask'] - merged['strike'])
            final_chunks.append(merged)

    if final_chunks:
        out_file = f"{DATADIR}/deribit_pcp_results_{date_str}_{freq}.parquet"
        pd.concat(final_chunks).to_parquet(out_file)
        logger.info(f"SUCCESS: Analysis results saved to {out_file}")

if __name__ == "__main__":
    run_deribit_attack()