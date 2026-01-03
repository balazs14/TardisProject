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

def run_robust_sample(limit=10):
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

    for opt in target_options:
        if processed_count >= limit:
            break

        exchange = opt['exchange']
        symbol = opt['id']
        api_exchange = "okex-options" if exchange == "okx" else exchange
        
        # Determine target date
        expiry_dt = parse_expiry(opt['metadata'].get('expiry'))
        target_date = benchmark_date if not expiry_dt or expiry_dt >= today else (expiry_dt - timedelta(days=7)).strftime("%Y-%m-%d")
        year, month, day = target_date.split('-')
        
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

if __name__ == "__main__":
    run_robust_sample(limit=10)