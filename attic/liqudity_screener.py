import os
import requests
import pandas as pd
import gzip
import json
import time
import urllib.parse
from datetime import datetime, timedelta
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
from tardis.utils import _configure_logging

# 1. Setup Environment
load_dotenv()
API_KEY = os.environ.get("TARDIS_API_KEY")

# 2. Setup Robust Session (Handles Retries and Network Glitches)
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

def parse_expiry(expiry_str):
    if not expiry_str: return None
    try:
        if any(m in expiry_str.upper() for m in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]):
            return datetime.strptime(expiry_str.upper(), "%d%b%y")
        if len(expiry_str) == 6 and expiry_str.isdigit():
            return datetime.strptime(expiry_str, "%y%m%d")
    except: return None
    return None

def liquidity_screener(df, symbol):
    if df.empty:
        return None

    # Oszlopnevek normalizálása (Tardis néha 'amount'-ot használ 'size' helyett)
    qty_col = 'amount' if 'amount' in df.columns else 'size'
    ts_col = 'timestamp' if 'timestamp' in df.columns else 'local_timestamp'

    # Alapvető számítások
    total_trades = len(df)
    total_volume = df[qty_col].sum()
    avg_trade_size = df[qty_col].mean()
    median_trade_size = df[qty_col].median()
    
    # Időbeli adatok (Tardis timestamp nanoszekundumban vagy mikroszekundumban van)
    df['dt'] = pd.to_datetime(df[ts_col], unit='us') # vagy 'ns' az exchangetől függően
    time_span_minutes = (df['dt'].max() - df['dt'].min()).total_seconds() / 60
    
    # Hány különböző percben volt kereskedés? (0-60-ig egy órás mintánál)
    active_minutes = df['dt'].dt.minute.nunique()
    
    # Árfolyam tartomány (Proxy a slippage-re)
    price_range = (df['price'].max() - df['price'].min()) / df['price'].mean() if total_trades > 1 else 0

    # Eredmények összegyűjtése egy szótárba
    stats = {
        "symbol": symbol,
        "total_trades": total_trades,
        "total_volume": round(total_volume, 4),
        "avg_trade_size": round(avg_trade_size, 4),
        "median_trade_size": round(median_trade_size, 4),
        "trades_per_minute": round(total_trades / time_span_minutes, 2) if time_span_minutes > 0 else 0,
        "active_minutes_per_hour": active_minutes,
        "relative_price_range": round(price_range, 6)
    }

    return stats

def get_working_days_list(opt, benchmark_str="2025-12-01", today=datetime(2026, 1, 3)):
    expiry_str = opt['metadata'].get('expiry')
    expiry_dt = parse_expiry(expiry_str)
    target_dates = []

    # 1. ESET: AKTÍV opció vagy hiányzó lejárat (Benchmark hét munkanapjai)
    if not expiry_dt or expiry_dt >= today:
        # Dec 1, 2025 hétfőre esik. Vegyük azt a hetet (H-P).
        base_date = datetime.strptime(benchmark_str, "%Y-%m-%d")
        # Megkeressük a hét hétfőjét, ha nem pont hétfőt adtál meg:
        start_of_week = base_date - timedelta(days=base_date.weekday())
        
        for i in range(5):
            day = start_of_week + timedelta(days=i)
            target_dates.append(day.strftime("%Y-%m-%d"))

    # 2. ESET: LEJÁRT opció (Lejárat előtti utolsó 5 munkanap)
    else:
        # Elindulunk a lejárat előtti naptól visszafelé
        current_day = expiry_dt - timedelta(days=1)
        while len(target_dates) < 5:
            # 0=Hétfő, 4=Péntek (tehát < 5 jelent munkanapot)
            if current_day.weekday() < 5:
                target_dates.append(current_day.strftime("%Y-%m-%d"))
            current_day -= timedelta(days=1)
        
        # Visszaállítjuk időrendi sorrendbe (opcionális)
        target_dates.reverse()

    return target_dates

def run_robust_sample(limit=3):
    if not API_KEY:
        print("❌ Error: TARDIS_API_KEY not found in .env file.")
        return

    # Load symbols
    with open("available_options.json", "r") as f:
        options_data = json.load(f)
    
    # Filter for BTC and ETH (taking a slice for the limit)
    target_options = [opt for opt in options_data if opt['id'].upper().startswith(('BTC', 'ETH'))]
    
    print(f"🚀 Starting Sampler. Target: First {limit} instruments.")
    print("-" * 50)

    headers = {"Authorization": f"Bearer {API_KEY}"}
    processed_count = 0
    today = datetime(2025, 12, 30)
    benchmark_date = "2025-12-01"

    liquidity_stats = []

    for opt in target_options:
        if processed_count >= limit:
            break

        exchange = opt['exchange']
        symbol = opt['id']
        api_exchange = "okex-options" if exchange == "okx" else exchange
        
        # Determine target date
        target_dates = get_working_days_list(opt, benchmark_str=benchmark_date, today=today)
        
        for target_date in target_dates:
            year, month, day = target_date.split("-")
            
            # URL Encode the symbol (Essential for symbols like BTC_USDC or symbols with slashes)
            encoded_symbol = urllib.parse.quote(symbol, safe='')
            url = f"https://datasets.tardis.dev/v1/{api_exchange}/trades/{year}/{month}/{day}/{encoded_symbol}.csv.gz"
        
            print(f"[{processed_count + 1}] Requesting: {exchange} | {symbol} | Date: {target_date}")
            try:
                # Small sleep to prevent DNS flooding
                time.sleep(0.5)
            
                response = session.get(url, headers=headers, stream=True, timeout=10)
            
                if response.status_code == 200:
                    with gzip.GzipFile(fileobj=response.raw) as gz:
                        df = pd.read_csv(gz)
                        print(f"✅ SUCCESS: {symbol} has {len(df)} trades.")
                        # RUN LIQUIDITY SCREENER
                        stats = liquidity_screener(df, symbol)
                        if stats:
                            print("---- LIQUIDITY STATS ----")
                            for k, v in stats.items():
                                print(f"{k}: {v}")

                        liquidity_stats.append(stats)

                        #add target date range to stats
                        if stats:
                            stats['target_date'] = target_date

                        #add exchange to stats
                        if stats:
                            stats['exchange'] = exchange
                    
                        # SHOW HEAD OF DATASET
                        if not df.empty:
                            print("\n--- DATA PREVIEW ---")
                            print(df.head(3).to_string())
                            print("-" * 30 + "\n")
                        else:
                            print("ℹ️ File was empty (Header only).")
                        
                elif response.status_code == 404:
                    print(f"ℹ️ 404: No trades occurred for {symbol} on this date.\n")
                elif response.status_code == 401:
                    print(f"❌ 401: Unauthorized. Check API Key.\n")
                else:
                    print(f"⚠️ Unexpected Status {response.status_code} for {symbol}\n")
                processed_count += 1

            except Exception as e:
                print(f"🔥 Network/Parsing Error for {symbol}: {e}\n")
                # If we hit a network error, wait longer before next try
                time.sleep(2)

    print("-" * 50)
    print(f"Sampler finished. Processed {processed_count} attempts.")

    #output liquidity stats to csv
    if liquidity_stats:
        stats_df = pd.DataFrame(liquidity_stats)
        stats_df.to_csv("liquidity_stats_output.csv", index=False)
        print("Liquidity stats saved to liquidity_stats_output.csv")

if __name__ == "__main__":
    _loglevel = "INFO"
    if "--loglevel" in sys.argv:
        _i = sys.argv.index("--loglevel")
        if _i + 1 >= len(sys.argv):
            raise SystemExit("Expected value after --loglevel")
        _loglevel = sys.argv[_i + 1]
        del sys.argv[_i:_i + 2]
    _configure_logging(_loglevel)

    run_robust_sample(limit=10000000)  # Set a high limit to process all available options