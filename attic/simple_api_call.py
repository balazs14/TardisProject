import requests
import pandas as pd
import io
import gzip
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("TARDIS_API_KEY") or os.getenv("API_KEY")

if not api_key:
    raise ValueError("API key not found in .env file. Please add TARDIS_API_KEY or API_KEY to your .env file.")

# Headers with API authentication
headers = {"Authorization": f"Bearer {api_key}"}

# The folder where we will save files
userpc=os.getenv("USERPROFILE")
output_folder = os.path.join(userpc, "Downloads", "tardis_sample_data")

# The specific Option Contract and Date
symbol = "btc-260327-40000-p"
year, month, day = "2025", "12", "05"
exchange = "binance-european-options"

# The list of all possible data types (from your error message)
data_types = [
    'quotes',                
    'derivative_ticker',     
    'trades',
    'book_ticker',
    'liquidations',
    'options_chain',
    'book_snapshot_5',
    'book_snapshot_25',
    'incremental_book_L2'
]

# Create the directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

print(f"Processing data for {symbol} on {year}-{month}-{day}...\n")

for dtype in data_types:
    # Construct the URL
    url = f"https://datasets.tardis.dev/v1/{exchange}/{dtype}/{year}/{month}/{day}/{symbol}.csv.gz"
    
    print(f"Checking {dtype}...", end=" ")
    
    try:
        # Stream the request so we don't download huge files into memory immediately
        response = requests.get(url, stream=True, headers=headers)
        
        if response.status_code == 200:
            print("Found! Processing...", end=" ")
            
            # Decompress the stream on the fly
            with gzip.open(io.BytesIO(response.content), 'rt') as f:
                content = f.read()
                
                # Check if file is empty or has no columns
                if not content.strip():
                    print("File is empty (no data).")
                    continue
                
                # Try to parse the CSV
                try:
                    df = pd.read_csv(io.StringIO(content), nrows=100 if dtype != 'quotes' else None)
                    
                    if df.empty:
                        print("File parsed but contains no rows.")
                        continue
                    
                    # LOGIC: Full download for 'quotes', Sample for others
                    if dtype == 'quotes':
                        filename = f"FULL_{dtype}.csv"
                        print(f"Saving {len(df)} rows...", end=" ")
                    else:
                        filename = f"SAMPLE_{dtype}.csv"
                        print(f"Sampled {len(df)} rows...", end=" ")

                    # Save to the target folder
                    save_path = os.path.join(output_folder, filename)
                    df.to_csv(save_path, index=False)
                    print(f"Saved to {filename}")
                    
                except pd.errors.EmptyDataError:
                    print("No columns to parse (empty file).")
                    
        elif response.status_code == 404:
            print("Not found (No data for this specific option).")
        else:
            print(f"Error: Status {response.status_code}")

    except Exception as e:
        print(f"Network/Parsing Error: {e}")

print(f"\nDone! Check your folder: {output_folder}")