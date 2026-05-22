import argparse
from datetime import datetime

import polars as pl
import pytest

from tardis import align_option_chain_trades_quotes as aoctq
from tardis import entry_point


def test_align_option_chain_trades_quotes_left_joins_trades_and_quotes():
    df_option_chain = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 3, 12, 16, 50, 0),
                datetime(2026, 3, 12, 16, 55, 0),
            ],
            "symbol": ["AVAX_USDC-13MAR26-7-P", "AVAX_USDC-13MAR26-7-P"],
            "strike_price": [7.0, 7.0],
            "expiration": [
                datetime(2026, 3, 13, 8, 0, 0),
                datetime(2026, 3, 13, 8, 0, 0),
            ],
            "type": ["put", "put"],
            "bid_price": [0.025, 0.026],
            "ask_price": [0.026, 0.027],
        }
    )

    df_trades = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 12, 16, 50, 0)],
            "symbol": ["AVAX_USDC-13MAR26-7-P"],
            "trade_amount": [100.0],
            "trade_signed_amount": [-100.0],
            "trade_price_amount": [2.55],
        }
    )

    df_quotes = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 12, 16, 50, 0)],
            "symbol": ["AVAX_USDC-13MAR26-7-P"],
            "bid_price": [0.024],
            "ask_price": [0.028],
            "bid_amount": [10.0],
            "ask_amount": [12.0],
            "stale": [False],
            "local_timestamp": [datetime(2026, 3, 12, 16, 50, 1)],
        }
    )

    out = aoctq.align_option_chain_trades_quotes(df_option_chain, df_trades, df_quotes).sort("timestamp")

    assert out.height == 2
    assert out.get_column("trade_amount").to_list() == [100.0, None]
    assert out.get_column("trade_signed_amount").to_list() == [-100.0, None]
    assert out.get_column("quote_bid_price").to_list() == [0.024, None]
    assert out.get_column("quote_ask_price").to_list() == [0.028, None]
    assert out.get_column("quote_stale").to_list() == [False, None]
    assert out.get_column("bid_price").to_list() == [0.025, 0.026]
    assert "quote_local_timestamp" not in out.columns


def test_create_aligned_option_chain_trades_quotes_dispatches_okex(monkeypatch):
    expected = pl.DataFrame({"timestamp": [], "symbol": [], "strike_price": [], "expiration": [], "type": []}, schema={
        "timestamp": pl.Datetime("us"),
        "symbol": pl.String,
        "strike_price": pl.Float64,
        "expiration": pl.Datetime("us"),
        "type": pl.String,
    })

    def fake_create(date, symbol, sample_freq, raw_data_dir):
        assert date == "2026-01-20"
        assert symbol == "OPTIONS"
        assert sample_freq == "5min"
        assert raw_data_dir == "datasets/okex_raw/"
        return expected

    monkeypatch.setattr(aoctq, "create_okex_aligned_option_chain_trades_quotes", fake_create)

    out = aoctq.create_aligned_option_chain_trades_quotes("okex", "2026-01-20")

    assert out.equals(expected)


def test_create_aligned_option_chain_trades_quotes_dispatches_deribit(monkeypatch):
    expected = pl.DataFrame({"timestamp": [], "symbol": [], "strike_price": [], "expiration": [], "type": []}, schema={
        "timestamp": pl.Datetime("us"),
        "symbol": pl.String,
        "strike_price": pl.Float64,
        "expiration": pl.Datetime("us"),
        "type": pl.String,
    })

    def fake_create(date, symbol, sample_freq, raw_data_dir):
        assert date == "2026-03-12"
        assert symbol == "BTC_OPTIONS"
        assert sample_freq == "15min"
        assert raw_data_dir == "custom/deribit_raw/"
        return expected

    monkeypatch.setattr(aoctq, "create_deribit_aligned_option_chain_trades_quotes", fake_create)

    out = aoctq.create_aligned_option_chain_trades_quotes(
        "deribit",
        "2026-03-12",
        symbol="BTC_OPTIONS",
        sample_freq="15min",
        raw_data_dir="custom/deribit_raw/",
    )

    assert out.equals(expected)


def test_create_aligned_option_chain_trades_quotes_prints_error_for_unsupported_exchange(capsys):
    with pytest.raises(AssertionError):
        aoctq.create_aligned_option_chain_trades_quotes("binance", "2026-03-12")

    captured = capsys.readouterr()
    assert "Unsupported exchange: binance. Expected okex or deribit." in captured.out


def test_create_deribit_aligned_option_chain_trades_quotes_errors_when_any_input_is_missing(monkeypatch):
    non_empty = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 12, 16, 50, 0)],
            "symbol": ["AVAX_USDC-13MAR26-7-P"],
            "strike_price": [7.0],
            "expiration": [datetime(2026, 3, 13, 8, 0, 0)],
            "type": ["put"],
            "trade_amount": [100.0],
            "trade_signed_amount": [-100.0],
            "trade_price_amount": [2.55],
            "bid_price": [0.024],
            "ask_price": [0.028],
            "bid_amount": [10.0],
            "ask_amount": [12.0],
            "stale": [False],
        }
    )
    empty = pl.DataFrame()

    def fake_access_files(exchange, datatype, date, symbol_list, sample_freq, raw_data_dir):
        if datatype == "options_chain":
            return empty, []
        return non_empty, ["OPTIONS"]

    monkeypatch.setattr(aoctq, "access_files", fake_access_files)

    with pytest.raises(AssertionError, match="Missing options_chain data"):
        aoctq.create_deribit_aligned_option_chain_trades_quotes(
            date="2026-03-12",
            symbol="OPTIONS",
            sample_freq="5min",
            raw_data_dir="datasets/deribit_raw/",
        )


def test_entry_point_dispatches_align_option_chain_trades_quotes(monkeypatch):
    called = {}

    def fake_autocomplete(_parser):
        return None

    def fake_run_cli_args(args: argparse.Namespace):
        called["exchange"] = args.exchange
        called["date"] = args.date
        called["symbol"] = args.symbol

    monkeypatch.setattr(entry_point.argcomplete, "autocomplete", fake_autocomplete)
    monkeypatch.setattr(entry_point.align_option_chain_trades_quotes, "run_cli_args", fake_run_cli_args)

    entry_point.run_cli([
        "align-option-chain-trades-quotes",
        "--exchange", "okex",
        "--date", "2026-01-20",
    ])

    assert called == {"exchange": "okex", "date": "2026-01-20", "symbol": "OPTIONS"}