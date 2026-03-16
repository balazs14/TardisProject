from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from tardis.lib import test_utils as tu
#.lib.test_utils import assert_df_equal, df_from_string

from okex import pcp_breaking_pandas
from okex import pcp_breaking_polars


@pytest.fixture
def sample_opt_df_pandas():
    df = tu.df_from_string('''
name  value
ts              2026-01-01
exp             2027-01-01
fut_exp         2027-01-01T00:00:00
spot_ask_price                101.0
spot_bid_price                 99.0
call_ask_price                0.112
call_bid_price                 0.11
put_ask_price                 0.092
put_bid_price                 0.091
fut_ask_price                102.01
fut_bid_price                 102.0
strike                        100.0 ''').set_index('name')

    df = df.T
    df.timestampize(['ts','exp','fut_exp'])
    float_cols = 'spot_ask_price,spot_bid_price,call_ask_price,call_bid_price,put_ask_price,put_bid_price,fut_ask_price,fut_bid_price,strike'.split(',')
    df.floatize(float_cols)


    return df

@pytest.fixture
def sample_opt_df_polars(sample_opt_df_pandas):
    return pl.from_pandas(sample_opt_df_pandas)


@pytest.fixture
def expected_snapshot_string():
    return '''                                    value
name
ts                    2026-01-01 00:00:00
exp                   2027-01-01 00:00:00
fut_exp               2027-01-01 00:00:00
spot_ask_price                      101.0
spot_bid_price                       99.0
call_ask_price                      0.112
call_bid_price                       0.11
put_ask_price                       0.092
put_bid_price                       0.091
fut_ask_price                      102.01
fut_bid_price                       102.0
strike                              100.0
index                               100.0
call_ask_price_xS                  11.312
call_bid_price_xS                   10.89
put_ask_price_xS                    9.292
put_bid_price_xS                    9.009
tte                                   1.0
pcpb_forward                     -0.00314
pcpb_backward                   -0.004005
cost                                0.003
capital_fwd                       0.71598
capital_bck                       0.67697
pcpb_fwd_real                    -0.00614
pcpb_bck_real                   -0.007005
pcpb_fwd_real_bp               -85.752555
pcpb_bck_real_bp               -103.48186
pcpb_fwd_bp                    -43.851943
pcpb_bck_bp                    -59.166751
pcpb_fwd_ann_bp                -43.851943
pcpb_bck_ann_bp                -59.166751
pcpb_fwd_real_ann_bp           -85.752555
pcpb_bck_real_ann_bp           -103.48186
put_opt_spread_bp                    28.3
call_opt_spread_bp                   42.2
bigger_opt_spread_bp                 42.2
amu_bp                         -19.197114

'''


def test_pcp_breaking_polars_snapshot(sample_opt_df_polars, expected_snapshot_string):
    out = pcp_breaking_polars(sample_opt_df_polars).to_pandas().T
    out.index.name = 'name'
    out.columns = ['value']
    tu.assert_df_equal(out, expected_snapshot_string)


def test_pcp_breaking_pandas_snapshot(sample_opt_df_pandas, expected_snapshot_string):
    out = pcp_breaking_pandas(sample_opt_df_pandas).T
    tu.assert_df_equal(out, expected_snapshot_string)
