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
    writer = None
    final_cols = ['ts', 'symbol', 'ask_price', 'bid_price', 'local_timestamp']
    potential_prices = ['ask_price', 'bid_price', 'best_ask_price', 'best_bid_price', 'mark_price', 'last_price']

    for csv_file in csv_files:
        if not os.path.exists(csv_file): continue
        logger.info(f"Processing {csv_file}...")
        
        csv_iter = pd.read_csv(csv_file, chunksize=1_000_000, iterator=True,
                               usecols=lambda c: c in ['symbol', 'local_timestamp', 'timestamp'] or c in potential_prices)
        
        for chunk in csv_iter:
            # Alapértelmezett átnevezések (ha vannak)
            chunk = chunk.rename(columns={'best_ask_price': 'ask_price', 'best_bid_price': 'bid_price'})
            
            # 2. LOGIKA A HATÁRIDŐS ÁRAKHOZ: 
            # Ha nincs ask_price (mert Futures), használjuk a mark_price-t vagy last_price-t
            if 'ask_price' not in chunk.columns or chunk['ask_price'].isnull().all():
                if 'mark_price' in chunk.columns:
                    chunk['ask_price'] = chunk['mark_price']
                    chunk['bid_price'] = chunk['mark_price']
                elif 'last_price' in chunk.columns:
                    chunk['ask_price'] = chunk['last_price']
                    chunk['bid_price'] = chunk['last_price']

            if 'local_timestamp' not in chunk.columns and 'timestamp' in chunk.columns:
                chunk['local_timestamp'] = chunk['timestamp']
            
            chunk['ts'] = pd.to_datetime(chunk['local_timestamp'], unit='us')
            
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
        logger.info(f"Downloading market data...")
        
        # Opciók: marad a 'quotes' (itt a Deribit ask_price/bid_price formátumot ad)
        datasets.download(exchange="deribit", data_types=["quotes"], 
                          from_date=start_time_str, to_date=end_date_str, 
                          symbols=["OPTIONS"], download_dir=DATADIR, api_key=API_KEY)
        
        # Határidős ügyletek: 'derivative_ticker' (itt best_ask_price/best_bid_price van)
        datasets.download(exchange="deribit", data_types=["derivative_ticker"], 
                          from_date=start_time_str, to_date=end_date_str, 
                          symbols=["FUTURES"], download_dir=DATADIR, api_key=API_KEY)        
        
        # FONTOS: A glob-nak mindkét fájltípust meg kell találnia!
        all_csvs = glob.glob(f"{DATADIR}/deribit_*.csv.gz")
        convert_to_parquet(all_csvs, pq_path)

    # D. Resampling logic
    raw_df = pd.read_parquet(pq_path)
    grid = pd.date_range(target_dt, target_dt + pd.Timedelta(days=1), freq=freq, inclusive='left')
    symbol_data = {}
    grouped = raw_df.groupby('symbol')
    
    raw_df = pd.read_parquet(pq_path)
    logger.info(f"--- DIAGNOSTICS ---")
    logger.info(f"Total rows in Parquet: {len(raw_df)}")
    logger.info(f"Unique symbols found in data: {raw_df['symbol'].unique()}")

    grid = pd.date_range(target_dt, target_dt + pd.Timedelta(days=1), freq=freq, inclusive='left')
    symbol_data = {}
    grouped = raw_df.groupby('symbol')
    
    for sym in needed_symbols:
        if sym in grouped.groups:
            df_s = grouped.get_group(sym).sort_values('ts').drop_duplicates('ts', keep='last').set_index('ts')
            
            # CHECK: Does the future actually have price data?
            non_null_asks = df_s['ask_price'].count()
            logger.info(f"Symbol {sym}: {len(df_s)} raw rows, {non_null_asks} non-null ask prices.")
            
            if non_null_asks == 0:
                logger.warning(f"!!! Symbol {sym} has ZERO price data. Check CSV column names.")

            symbol_data[sym] = df_s[['ask_price', 'bid_price']].reindex(grid, method='ffill')
        else:
            logger.error(f"!!! Symbol {sym} NOT FOUND in grouped data. Mismatch in target JSON?")


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