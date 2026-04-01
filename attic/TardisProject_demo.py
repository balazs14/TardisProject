# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
df = pd.read_csv('okex/okex-futures_quotes_2026-01-01_BTC-USD-260227.csv.gz')
df['ts'] = pd.to_datetime(df.local_timestamp, unit='1us')
df.set_index('ts').ask_price.plot()
df.ask_price.describe()

