import subprocess
import sys
from datetime import date, timedelta

# Beállítások: Teszt időszak (2 nap)
start_date = date(2024, 5, 1)
end_date = date(2024, 5, 3) # Ez a nap már nem fut le

current_date = start_date

print("=== STARTING 2-DAY TEST RUN (PYTHON VERSION) ===")

while current_date < end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    print(f"\n------------------------------------------------")
    print(f"Processing Day: {date_str}")
    print(f"------------------------------------------------")
    
    # A fő script meghívása
    # Windowson a 'python' parancsot hívjuk
    try:
        subprocess.run([sys.executable, "deribit_polars.py", date_str], check=True)
        print(f"SUCCESS: {date_str} finished.")
    except subprocess.CalledProcessError:
        print(f"FAILURE: Error processing {date_str}")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

    # Nap léptetése
    current_date += timedelta(days=1)

print("\n=== TEST RUN COMPLETED ===")