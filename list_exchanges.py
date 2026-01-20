import os
import requests
import re
import json
from collections import defaultdict

# Load API key from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.environ.get("TARDIS_API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}

def parse_robust_metadata(symbol):
    """
    Robustly identifies Options and Dated Futures by parsing from the end of the string.
    """
    s = symbol.upper()
    
    # Exclude non-expiring instruments
    if any(x in s for x in ["-PERP", "PERPETUAL", "SPOT", "INDEX", "SWAP"]):
        return None

    # Flexible Expiry patterns (e.g., 27DEC24, 8SEP23, 250331)
    # Matches 1-2 digits, 3 letters, 2 digits OR exactly 6 digits
    expiry_pattern = r'([0-9]{1,2}[A-Z]{3}[0-9]{2}|[0-9]{6})'
    
    parts = s.split('-')
    if len(parts) < 2:
        return None

    metadata = {
        "instrument_type": None,
        "underlying": None,
        "expiry": None,
        "strike": None,
        "option_type": None
    }

    # 1. IDENTIFY EXPIRY FIRST
    # We look for parts matching the expiry pattern
    exp_matches = [i for i, p in enumerate(parts) if re.fullmatch(expiry_pattern, p)]
    if not exp_matches:
        return None
    
    # Use the last match as the expiry (handles complex underlyings with dates)
    exp_idx = exp_matches[-1]
    metadata["expiry"] = parts[exp_idx]
    metadata["underlying"] = "-".join(parts[:exp_idx])

    # 2. DIFFERENTIATE OPTION VS FUTURE
    # Options usually have C/P/CALL/PUT and a Strike
    pc_indicators = ['C', 'P', 'CALL', 'PUT']
    pc_val = next((p for p in parts[exp_idx:] if p in pc_indicators), None)

    if pc_val:
        metadata["instrument_type"] = "Option"
        metadata["option_type"] = "Call" if pc_val.startswith('C') else "Put"
        
        # Strike is the numeric part after expiry that isn't the PC indicator
        remaining = [p for i, p in enumerate(parts) if i > exp_idx and p != pc_val]
        for r in remaining:
            if re.match(r'^[0-9.]+$', r):
                metadata["strike"] = r
                break
        
        # Validation: An option needs an expiry and a strike
        if metadata["strike"]:
            return metadata
    else:
        # If no PC indicator, it's a Dated Future
        metadata["instrument_type"] = "Future"
        return metadata

    return None

def discover_pcp_triplets():
    print("Fetching supported exchanges from Tardis...")
    exchanges_resp = requests.get("https://api.tardis.dev/v1/exchanges")
    if exchanges_resp.status_code != 200:
        return []
    
    exchanges = exchanges_resp.json()
    inventory = []

    for exchange in exchanges:
        ex_id = exchange['id']
        print(f"Scanning {ex_id}...")
        
        url = f"https://api.tardis.dev/v1/exchanges/{ex_id}"
        resp = requests.get(url, headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            symbols = data.get('availableSymbols', [])
            
            for s in symbols:
                sym_id = s['id'] if isinstance(s, dict) else s
                meta = parse_robust_metadata(sym_id)
                if meta:
                    inventory.append({"exchange": ex_id, "id": sym_id, "metadata": meta})

    # Corrected Grouping logic: avoid 'NoneType' errors
    # tree[ex][und][exp] -> {"future": ID, "strikes": {strike: {type: ID}}}
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"future": None, "strikes": defaultdict(dict)})))

    for item in inventory:
        m = item['metadata']
        ex = item['exchange']
        und = m['underlying']
        exp = m['expiry']
        
        node = tree[ex][und][exp]
        
        if m['instrument_type'] == "Future":
            node["future"] = item['id']
        else:
            strike = m['strike']
            node["strikes"][strike][m['option_type']] = item['id']

    # Flatten tree into explicit PCP triplets
    triplets = []
    for ex, unds in tree.items():
        for und, exps in unds.items():
            for exp, data in exps.items():
                fut_id = data["future"]
                if not fut_id: 
                    continue # PCP requires a future
                
                for strike, opts in data["strikes"].items():
                    if 'Call' in opts and 'Put' in opts:
                        triplets.append({
                            "exchange": ex, 
                            "underlying": und, 
                            "expiry": exp, 
                            "strike": strike,
                            "call_symbol": opts['Call'], 
                            "put_symbol": opts['Put'], 
                            "future_symbol": fut_id
                        })
    return triplets

if __name__ == "__main__":
    pcp_triplets = discover_pcp_triplets()
    with open("pcp_triplets.json", "w") as f:
        json.dump(pcp_triplets, f, indent=2)
    
    print(f"\nDone! Found {len(pcp_triplets)} total triplets.")
    deribit_triplets = [t for t in pcp_triplets if t['exchange'] == 'deribit']
    print(f"Deribit triplets: {len(deribit_triplets)}")
    
    # Save a Deribit-only list for the next phase
    with open("deribit_pcp_targets.json", "w") as f:
        json.dump(deribit_triplets, f, indent=2)