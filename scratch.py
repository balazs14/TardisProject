import re
from pprint import pprint
import requests
import asyncio
from datetime import date
import pandas as pd
import lib.pandas_utils as pu
import lib.utils as utils

from bs4 import BeautifulSoup

from tardis_client import TardisClient, Channel
from tardis_dev import datasets

# 
# The goal of these is to explore programmatically the available data on
# https://docs.tardis.dev/downloadable-csv-files
# 

# this here is not really the beginnings of a library, these are scripts to be modified at will
# for exploration

def pick_exchanges():
    # the task here is to explore all the exchanges and see where
    # we can find at least bid/ask data for options contracts, as well as for
    # the underlying. This here I just looked at where the available channels contain
    # some obvious strings like "quote" or "Book"
    
    resp = requests.get("https://api.tardis.dev/v1/exchanges")
    resp.raise_for_status()
    exchanges = resp.json()
    for ex in exchanges:
        print(f"{ex['id']:<20} {ex['name']:<25} since {ex['availableSince'][:10]}")
    exdf = pd.DataFrame(exchanges)
    #
    patterns = [r'[qQ]uote', r'[dD]epth', r'[Bb]ook']
    compiled = [re.compile(p) for p in patterns]
    def has_option_quotes(row):
        channels = list(row.availableChannels)
        has_quotes = any( reg.search(ch) for reg in compiled for ch in channels)
        is_option = 'option' in row.id 
        not_expired = pd.isna(row.availableTo)
        #not_expired = True
        return has_quotes and is_option and not_expired
    df = exdf.loc[exdf.apply(has_option_quotes, axis=1)]
    #
    for key, row in df.iterrows():
        pprint ((row.id, row.availableChannels))


def get_info_on_exchange():
    #
    # get the symbol list for an exchage
    #

    exchange = 'binance-european-options'
    #exchange = 'bitmex'
    resp = requests.get(f"https://api.tardis.dev/v1/exchanges/{exchange}")
    resp.raise_for_status()
    info = resp.json()
    availableChannels = info['availableChannels']
    availableSymbols = pd.DataFrame(info['availableSymbols'])
    print(info['id'])
    print(availableChannels)
    print(availableSymbols)
    #
    fr = pu.next_bday(pd.Timestamp('2025-08-01',tz='UTC'))
    print(f'fr = {fr}')
    to = fr
    afr = availableSymbols.availableSince.timestampize()
    ato = availableSymbols.availableTo.timestampize()
    print(availableSymbols.loc[(fr>afr) & (ato>to)])
    currdf = availableSymbols.loc[(fr>afr) & (ato>to)].copy()
    currdf['maturity'] = currdf.id.str.split('-').apply(lambda x: pd.Timestamp('20'+x[1][:2]+'-'+x[1][2:4]+'-'+x[1][4:]))
    currdf['undr'] = currdf.id.str.split('-').apply(lambda x: x[0])
    currdf['strike'] = currdf.id.str.split('-').apply(lambda x: float(x[2]))
    currdf['pc'] = currdf.id.str.split('-').apply(lambda x: x[3])
    print(currdf[['undr','maturity']].drop_duplicates().set_index(['maturity','undr'],drop=False).sort_index().unstack('undr').undr.fillna(0))
    btc_symbols = currdf.loc[currdf.undr=='BTC', 'id']
    
    
def get_stream_of_data():
    #
    # get stream of lines from an exchange
    # once we know the exchange, the channel, the symbol, we can start downloading
    # some data. This is in general only practical as a stream, the full csv files are too big
    #

    #
    async def replay():
        tardis_client = TardisClient()

        # tardis_client.replay method returns Async Generator
        # https://rickyhan.com/jekyll/update/2018/01/27/python36.html
        messages = tardis_client.replay(
            exchange="binance-european-options",
            from_date="2025-08-01",
            to_date="2025-08-02",
            #filters=[Channel(name="trade", symbols=["XBTUSD","ETHUSD"]), Channel("orderBookL2", ["XBTUSD"])],
            #filters=[Channel(name="depth100", symbols=btc_symbols.to_list())]
            # ['trade', 'depth100', 'index', 'markPrice', 'ticker', 'openInterest']
            filters=[Channel(name="openInterest", symbols=[])]
        )
        #

        count = 0
        async for local_timestamp, message in messages:
            # local timestamp is a Python datetime that marks timestamp when given message has been received
            # message is a message object as provided by exchange real-time stream
            print(message)
            #print(pd.Timestamp(local_timestamp))
            count = count+1
            if count > 10 : break
    #
    asyncio.run(replay())
    #await replay()
    

def example_download_of_a_dataset():

    datasets.download(
        exchange="binance-european-options",
        data_types=[
            'book_snapshot_25'
        ],
        from_date="2025-05-01",
        to_date="2025-05-01",
        symbols=['OPTIONS','FUTURES'],
        api_key="YOUR API KEY (optionally)",
    )


def list_tardis_csv(exchange, channel, date):
    url = f"https://datasets.tardis.dev/v1/{exchange}/{channel}/{pd.Timestamp(date):%Y/%m/%d}/"
    print(url)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    #return [a['href'] for a in soup.find_all('a') if a['href'].endswith('.csv.gz')]
    return resp.text
files = list_tardis_csv("binance-european-options", "trade", date(2025, 5, 1))
print(files)
            

def convert_csv_to_parquet():

    for L1 in pd.read_csv('datasets/deribit_options_chain_2023-11-01_OPTIONS.csv.gz', chunksize=100):
        break
    print(L1)

    trades = pd.read_csv('datasets/deribit_trades_2023-11-01_OPTIONS.csv.gz')

    trades.to_parquet('datasets/deribit_trades_2023-11-01_OPTIONS.parquet')

    
