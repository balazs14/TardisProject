import os
import requests
import re

# Load API key using your provided logic
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.environ.get("TARDIS_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing TARDIS_API_KEY in .env file.")

headers = {"Authorization": f"Bearer {API_KEY}"}

def parse_metadata(symbol):
    """
    Parses Strike, Expiry, and Option Type from standard exchange symbols.
    """
    # Try to find common patterns: e.g., 27DEC24 or 241227, followed by strike, followed by C/P
    # Standard format: BTC-27DEC24-60000-C or BTC-241227-60000-P
    match = re.search(r'[-_]([0-9A-Z]{6,7})[-_]([0-9\.]+)[-_](C|P|CALL|PUT)$', symbol, re.IGNORECASE)
    
    if match:
        return {
            "expiry": match.group(1),
            "strike": match.group(2),
            "type": "Call" if match.group(3).upper().startswith('C') else "Put"
        }
    
    # Fallback for BitMEX style or others: .BXXX or similar
    if ".B" in symbol or "-C" in symbol or "-P" in symbol:
         return {"expiry": "See Symbol", "strike": "See Symbol", "type": "Option"}
         
    return {"expiry": None, "strike": None, "type": "Unknown"}

def get_options_data():
    # 1. Get all exchanges
    print("Fetching list of exchanges...")
    exchanges_url = "https://api.tardis.dev/v1/exchanges"
    exchanges = requests.get(exchanges_url).json()
    
    all_options = []

    for exchange in exchanges:
        ex_id = exchange['id']
        print(f"Checking {ex_id} for options...")
        
        url = f"https://api.tardis.dev/v1/exchanges/{ex_id}"
        resp = requests.get(url, headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            symbols_list = data.get('availableSymbols', [])
            
            for s in symbols_list:
                # FIX: Handle if symbol is a dictionary (BitMEX) or a string (Deribit)
                symbol_id = s['id'] if isinstance(s, dict) else s
                
                # Check if it's an option by naming convention
                upper_sym = symbol_id.upper()
                if any(suffix in upper_sym for suffix in ["-C", "-P", "CALL", "PUT"]) or ".B" in upper_sym:
                    all_options.append({
                        "exchange": ex_id,
                        "id": symbol_id,
                        "metadata": parse_metadata(symbol_id)
                    })
        else:
            print(f"  - Error {resp.status_code} on {ex_id}")

    return all_options

if __name__ == "__main__":
    options_list = get_options_data()
    
    print(f"\n--- SUCCESS: Found {len(options_list)} Options ---")
    
    # Print header
    print(f"{'EXCHANGE':<18} | {'OPTION ID':<35} | {'TYPE'}")
    print("-" * 70)
    
    # Print first 20 results
    for item in options_list[:20]:
        print(f"{item['exchange']:<18} | {item['id']:<35} | {item['metadata']['type']}")

    # Save results to JSON
    import json
    with open("available_options.json", "w") as f:
        json.dump(options_list, f, indent=2)
    print(f"\nFull list of {len(options_list)} options saved to 'available_options.json'")

    for item in options_list:
        if item['metadata']['expiry'] == 'See Symbol' or item['metadata']['strike'] == 'See Symbol':
            with open("options_needing_manual_check.txt", "a") as f:
                f.write(f"{item['exchange']} | {item['id']}\n")
    