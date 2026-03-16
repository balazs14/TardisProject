import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor


def process_single_csv(csv_file, output_dir, freq='5min'):
    """
    Process a single CSV file, aggregate it on a 5-minute grid, and save daily Parquet files.

    :param csv_file: Path to the CSV file.
    :param output_dir: Directory to save the daily Parquet files.
    :param freq: Aggregation frequency (default is '5min').
    """

    chunk = pd.read_csv(csv_file, dtype={
        'timestamp': 'int64',
        'local_timestamp': 'int64',
        'price': 'float64',
        'amount': 'float64',
        'side': 'string'
    })

    # Ensure the timestamp column is in datetime format
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='us', errors='coerce')

    # Set the timestamp as the index
    chunk.set_index('timestamp', inplace=True)
    chunk['side'] = np.where(chunk.side == 'buy', 1, -1)
    chunk['signed_amount'] = chunk.side * chunk.amount

    # Resample and aggregate
    resampled = chunk.resample(freq, offset='0min').apply({
        'price': 'mean',  # Average price
        'amount': 'sum',  # Sum of sizes
        'signed_amount': 'sum'
    })

    # Reset the index and adjust the timestamp to the end of the period
    resampled.reset_index(inplace=True)
    resampled['timestamp'] = resampled['timestamp'] + pd.Timedelta(freq)

    date = resampled.timestamp.dt.date.max()

    resampled.set_index('timestamp', inplace=True)
    daily_parquet_path = os.path.join(output_dir, f"{date}.parquet")
    table = pa.Table.from_pandas(resampled)
    pq.write_table(table, daily_parquet_path)
    print(f"Saved daily Parquet file: {daily_parquet_path}")


def process_csv_files(input_dir, output_dir, freq='5min'):
    """
    Process multiple CSV files concurrently, aggregate them, and save daily Parquet files.

    :param input_dir: Directory containing the CSV files.
    :param output_dir: Directory to save the daily Parquet files.
    :param freq: Aggregation frequency (default is '5min').
    """
    # Find all CSV files in the input directory
    csv_files = glob(os.path.join(input_dir, 'binance_trades_*BTCUSDT.csv.gz'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    if False:
        for csv in csv_files:
            process_single_csv(csv, output_dir, freq)
    else:
        # Process files concurrently
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_csv, csv_file, output_dir, freq) for csv_file in csv_files]
            for future in futures:
                future.result()  # Wait for all processes to complete


def concatenate_parquet_files(output_dir, final_output_file):
    """
    Concatenate all daily Parquet files into a single Parquet file.

    :param output_dir: Directory containing the daily Parquet files.
    :param final_output_file: Path to the final concatenated Parquet file.
    """
    daily_parquet_files = glob(os.path.join(output_dir, '*.parquet'))
    if not daily_parquet_files:
        raise FileNotFoundError(f"No Parquet files found in directory: {output_dir}")

    print(f"Concatenating {len(daily_parquet_files)} daily Parquet files...")
    dfs = [pd.read_parquet(parquet_file) for parquet_file in daily_parquet_files]
    final_df = pd.concat(dfs).sort_index()

    table = pa.Table.from_pandas(final_df)
    pq.write_table(table, final_output_file)
    print(f"Final aggregated data saved to {final_output_file}")


if __name__ == "__main__":
    input_directory = "/my/TardisProject/bihary/datasets/"  # Replace with the directory containing your CSV files
    daily_output_directory = "/my/TardisProject/bihary/datasets/daily_parquets/"  # Directory for daily Parquet files
    final_output_parquet = "/my/TardisProject/bihary/datasets/BTCUSDT_final.parquet"  # Final concatenated Parquet file

    # Step 1: Process CSV files and save daily Parquet files
    process_csv_files(input_directory, daily_output_directory)

    # Step 2: Concatenate daily Parquet files into a single Parquet file
    concatenate_parquet_files(daily_output_directory, final_output_parquet)
