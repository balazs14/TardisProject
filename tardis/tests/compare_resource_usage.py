import os
import shutil
import tempfile
import time
import tracemalloc
import psutil
from pathlib import Path
import pandas as pd

from tardis import download_and_convert as dac
import sys
from tardis.lib.utils import _configure_logging

def measure_disk_usage(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

def run_and_measure(method, *args, **kwargs):
    process = psutil.Process(os.getpid())
    tracemalloc.start()
    mem_before = process.memory_info().rss
    disk_before = measure_disk_usage(kwargs["data_dir"])
    t0 = time.time()
    result = method(*args, **kwargs)
    t1 = time.time()
    mem_after = process.memory_info().rss
    peak_mem, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    disk_after = measure_disk_usage(kwargs["data_dir"])
    return {
        "result": result,
        "time": t1 - t0,
        "mem_before": mem_before,
        "mem_after": mem_after,
        "peak_mem": peak_mem,
        "disk_before": disk_before,
        "disk_after": disk_after,
        "disk_used": disk_after - disk_before,
    }

def main():
    # Use a deterministic fake download as in the tests
    exchange = "binance"
    data_type = "quotes"
    symbol = "OPTIONS"
    day = "2023-02-01"
    resample_freq = "5min"

    # Create a temp dir for each method
    for method_name, method in [
        ("batch", dac.download_and_convert),
        ("streaming", dac.download_and_convert_streaming_resample),
    ]:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a fake CSV for batch, or patch streaming to read it
            def fake_download(*, exchange, data_types, from_date, to_date, symbols, download_dir, **kwargs):
                csv_path = Path(download_dir) / f"{exchange}_{data_types[0]}_{day}_{symbols[0]}.csv.gz"
                df = pd.DataFrame(
                    {
                        "exchange": [exchange] * 6,
                        "symbol": ["A", "A", "A", "B", "B", "B"],
                        "timestamp": [
                            1675209660000000,
                            1675209720000000,
                            1675210320000000,
                            1675209780000000,
                            1675209840000000,
                            1675210080000000,
                        ],
                        "local_timestamp": [
                            1675209660000000,
                            1675209720000000,
                            1675210320000000,
                            1675209780000000,
                            1675209840000000,
                            1675210080000000,
                        ],
                        "bid_price": [100.0, 101.0, 103.0, 200.0, 201.0, 202.0],
                        "ask_price": [101.0, 102.0, 104.0, 201.0, 202.0, 203.0],
                        "volume": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
                    }
                )
                df.to_csv(csv_path, index=False, compression="gzip")
            dac.datasets.download = fake_download

            if method_name == "streaming":
                # Write the CSV file directly for streaming
                fake_download(
                    exchange=exchange,
                    data_types=[data_type],
                    from_date=day,
                    to_date=day,
                    symbols=[symbol],
                    download_dir=tmpdir,
                )
                # Patch streaming to read the same CSV
                def fake_iter_tardis_csv_rows_streaming(url, api_key=None):
                    import csv, gzip
                    csv_path = Path(tmpdir) / f"{exchange}_{data_type}_{day}_{symbol}.csv.gz"
                    with gzip.open(csv_path, "rt") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            yield row
                dac._iter_tardis_csv_rows_streaming = fake_iter_tardis_csv_rows_streaming

            print(f"\nRunning {method_name}...")
            kwargs = dict(
                exchange=exchange,
                data_type=data_type,
                symbol=symbol,
                start_date=day,
                end_date=day,
                data_dir=tmpdir,
                resample_freq=resample_freq,
            )
            if method_name == "batch":
                kwargs["cleanup_csv"] = False
            stats = run_and_measure(method, **kwargs)
            print(f"Time: {stats['time']:.3f}s")
            print(f"Peak RAM: {stats['peak_mem']/1024/1024:.2f} MB")
            print(f"Disk used: {stats['disk_used']/1024:.2f} KB")

if __name__ == "__main__":
    _loglevel = "INFO"
    if "--loglevel" in sys.argv:
        _i = sys.argv.index("--loglevel")
        if _i + 1 >= len(sys.argv):
            raise SystemExit("Expected value after --loglevel")
        _loglevel = sys.argv[_i + 1]
        del sys.argv[_i:_i + 2]
    _configure_logging(_loglevel)

    main()
