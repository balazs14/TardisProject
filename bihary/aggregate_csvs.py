import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from glob import glob

def process_csv_files(input_dir, output_file, freq='5min'):
    """
    Process multiple CSV files, aggregate them on a 5-minute grid, and save as a single Parquet file.

    :param input_dir: Directory containing the CSV files.
    :param output_file: Path to the output Parquet file.
    :param freq: Aggregation frequency (default is '5min' for 5 minutes).
    """
    # Find all CSV files in the input directory
    csv_files = glob(os.path.join(input_dir, 'binance_trades_*BTCUSDT.csv.gz'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {input_dir}")

    # Initialize an empty list to store aggregated chunks
    aggregated_chunks = []

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        # Read the CSV file in chunks
        csv_iter = pd.read_csv(csv_file, chunksize=5_000_000, dtype={'timestamp':'int64',
                                                                   'local_timestamp':'int64',
                                                                   'price':'float64',
                                                                   'amount':'float64'})

        for chunk in csv_iter:
            # Ensure the timestamp column is in datetime format
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='us', errors='coerce')

            # Set the timestamp as the index
            chunk.set_index('timestamp', inplace=True)
            chunk['side'] = np.where(chunk.side == 'buy', 1, -1)
            chunk['signed_amount'] = chunk.side * chunk.amount
            # Resample and aggregate
            resampled = chunk.resample(freq).apply({
                'price': 'mean',  # Average price
                'amount': 'sum',   # Sum of sizes
                'signed_amount': 'sum'
            })

            # Reset the index and adjust the timestamp to the end of the period
            resampled.reset_index(inplace=True)
            resampled['timestamp'] = resampled['timestamp'] + pd.Timedelta(freq)

            # Append the aggregated chunk to the list
            aggregated_chunks.append(resampled)

    # Concatenate all aggregated chunks
    final_df = pd.concat(aggregated_chunks).set_index('timestamp').sort_index()

    # Save the final DataFrame to a Parquet file
    table = pa.Table.from_pandas(final_df)
    pq.write_table(table, output_file)
    print(f"Aggregated data saved to {output_file}")

if __name__ == "__main__":
    input_directory = "/my/TardisProject/datasets/"  # Replace with the directory containing your CSV files
    output_parquet = "/my/TardisProject/datasets/BTCUSDT.parquet"  # Replace with the desired output file path

    process_csv_files(input_directory, output_parquet)
