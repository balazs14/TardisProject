import os
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tardis_dev import datasets

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

datadir = 'binance'

def download_binance_btcusdt(fr='2019-01-01', to=None, symbol='BTCUSDT'):
    """
    Downloads Binance BTC/USDT data using the tardis_dev.datasets module and saves it as Parquet.
    :param fr: Start date in 'YYYY-MM-DD' format
    :param to: End date in 'YYYY-MM-DD' format (optional, defaults to now)
    :param symbol: Trading pair symbol (e.g., 'BTC-USDT')
    """
    csv_fname = f'{datadir}/binance_trades_{fr}_{symbol}.csv.gz'
    pq_fname = f'{datadir}/binance_trades_{fr}_{symbol}.parquet'

    if os.path.exists(pq_fname):
        logger.debug(f'Parquet file already exists: {pq_fname}')
        return pq_fname

    if not os.path.exists(csv_fname):
        logger.info(f'Downloading {csv_fname}')
        datasets.download(
            exchange='binance',
            data_types=['trades'],
            from_date=fr,
            to_date=to,
            symbols=[symbol],
            api_key=os.environ.get('TARDIS_API_KEY', None)
        )
    else:
        logger.debug(f'CSV already here {csv_fname}')

    logger.info(f'Converting {csv_fname} to {pq_fname}...')
    # Read CSV in chunks and write to Parquet to handle large files efficiently
    csv_iter = pd.read_csv(csv_fname, chunksize=500_000, iterator=True,
                           dtype={
                               'exchange': 'string',
                               'symbol': 'string',
                               'timestamp': 'int64',
                               'local_timestamp': 'int64',
                               'price': 'float64',
                               'quantity': 'float64',
                               'side': 'string',
                           })

    first_chunk = next(csv_iter)
    table = pa.Table.from_pandas(first_chunk, preserve_index=False)
    with pq.ParquetWriter(pq_fname, table.schema) as writer:
        writer.write_table(table)
        for chunk in csv_iter:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            writer.write_table(table)
    logger.info(f'Finished converting to {pq_fname}')
    return pq_fname


if __name__ == "__main__":
    # Parameters
    symbol = 'BTCUSDT'  # Binance uses USDT instead of USD
    start_date = '2023-02-01'
    end_date = '2026-01-11'  # Fetch up to the present

    logger.info(f"Downloading {symbol} data from {start_date} to {end_date or 'now'}...")
    download_binance_btcusdt(fr=start_date, to=end_date, symbol=symbol)
