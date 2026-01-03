import os
import requests
import re
import json

# Load API key from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.environ.get("TARDIS_API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}

def parse_detailed_metadata(symbol):
    """
    Identifies Vanilla Options vs Strategies and extracts strike/expiry.
    Returns None if the instrument is not an option.
    """
    s = symbol.upper()
    
    # 1. EXCLUSIONS: Skip Perpetuals and non-option pairs
    if any(x in s for x in ["-PERP", "PERPETUAL", "SPOT", "INDEX", "SWAP"]):
        return None

    metadata = {
        "instrument_type": "Vanilla Option",
        "expiry": None,
        "strike": None,
        "option_type": None,
        "is_strategy": False
    }

    # 2. IDENTIFY COMPLEX STRATEGIES (Common on Deribit)
    # Exclude combos like Condors, Diagonals, and Spreads
    strategies = ["-CCOND-", "-CDIAG-", "-CCAL-", "-CSR-", "-BCOND-", "-BDIAG-", "-BCAL-", "-BSR-"]
    if any(strat in s for strat in strategies):
        metadata["instrument_type"] = "Complex Strategy"
        metadata["is_strategy"] = True
        return metadata

    # 3. PARSING LOGIC
    # Format A: Standard (e.g., BTC-27DEC24-60000-C)
    match_a = re.search(r'-([0-9A-Z]{6,9})-([0-9.]+)-(C|P|CALL|PUT)$', s)
    if match_a:
        metadata["expiry"] = match_a.group(1)
        metadata["strike"] = match_a.group(2)
        metadata["option_type"] = "Call" if match_a.group(3).startswith('C') else "Put"
        return metadata

    # Format B: Binance/Huobi variant (e.g., BTC-USDT-210625-C-57000)
    match_b = re.search(r'-([0-9]{6})-(C|P)-([0-9.]+)$', s)
    if match_b:
        metadata["expiry"] = match_b.group(1)
        metadata["option_type"] = "Call" if match_b.group(2) == 'C' else "Put"
        metadata["strike"] = match_b.group(3)
        return metadata

    # Format C: Crypto.com (e.g., BTCUSD-230630-CW21000)
    match_c = re.search(r'-([0-9]{6})-(C|P)W([0-9.]+)$', s)
    if match_c:
        metadata["expiry"] = match_c.group(1)
        metadata["option_type"] = "Call" if match_c.group(2) == 'C' else "Put"
        metadata["strike"] = match_c.group(3)
        return metadata

    # 4. FINAL FILTER: If no Call/Put indicator is found, it's not a vanilla option
    if not any(x in s for x in ["-C-", "-P-", "-CALL", "-PUT", "-C", "-P"]):
        return None

    return metadata

def discover_all_options():
    print("Fetching supported exchanges from Tardis...")
    exchanges_resp = requests.get("https://api.tardis.dev/v1/exchanges")
    if exchanges_resp.status_code != 200:
        print("Could not connect to Tardis API.")
        return
    
    exchanges = exchanges_resp.json()
    inventory = []

    for exchange in exchanges:
        ex_id = exchange['id']
        print(f"Scanning {ex_id}...")
        
        # Get the list of all symbols for this exchange
        url = f"https://api.tardis.dev/v1/exchanges/{ex_id}"
        resp = requests.get(url, headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            symbols = data.get('availableSymbols', [])
            
            count_found = 0
            for s in symbols:
                # Handle cases where symbol is a dict (like BitMEX)
                sym_id = s['id'] if isinstance(s, dict) else s
                
                meta = parse_detailed_metadata(sym_id)
                
                if meta:
                    inventory.append({
                        "exchange": ex_id,
                        "id": sym_id,
                        "metadata": meta
                    })
                    count_found += 1
            
            if count_found > 0:
                print(f"  ✅ Found {count_found} options.")
        else:
            print(f"  ❌ Error {resp.status_code} (Exchange might be delisted).")

    return inventory

if __name__ == "__main__":
    all_options = discover_all_options()
    
    if all_options:
        with open("available_options.json", "w") as f:
            json.dump(all_options, f, indent=2)
        
        print(f"\nDiscovery Finished!")
        print(f"Total Vanilla Options Found: {len(all_options)}")
        print("Results saved to 'available_options.json'")