import argparse
from datetime import datetime

import polars as pl
import pytest

from tardis import align_derivative_ticker_trades_quotes as adttq
from tardis import entry_point


def test_align_derivative_ticker_trades_quotes_left_joins_trades_and_quotes():
    df_derivative_ticker = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 26, 0, 0, 0), datetime(2026, 3, 26, 0, 5, 0)],
            "symbol": ["BTC-24APR26", "BTC-24APR26"],
            "mark_price": [71250.0, 71275.0],
            "index_price": [71222.68, 71194.89],
            "open_interest": [1250.0, 1251.0],
            "local_timestamp": [datetime(2026, 3, 26, 0, 0, 1), datetime(2026, 3, 26, 0, 5, 1)],
            "funding_timestamp": [datetime(2026, 3, 26, 8, 0, 0), datetime(2026, 3, 26, 8, 0, 0)],
            "bid_price": [71250.0, 71275.0],
            "ask_price": [71250.0, 71275.0],
            "bid_amount": [None, None],
            "ask_amount": [None, None],
        }
    )

    df_trades = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 26, 0, 0, 0)],
            "symbol": ["BTC-24APR26"],
            "trade_amount": [10.0],
            "trade_signed_amount": [-10.0],
            "trade_price_amount": [712500.0],
        }
    )

    df_quotes = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 26, 0, 0, 0)],
            "symbol": ["BTC-24APR26"],
            "bid_price": [71240.0],
            "ask_price": [71260.0],
            "bid_amount": [5.0],
            "ask_amount": [7.0],
            "stale": [False],
            "local_timestamp": [datetime(2026, 3, 26, 0, 0, 1)],
        }
    )

    out = adttq.align_derivative_ticker_trades_quotes(df_derivative_ticker, df_trades, df_quotes).sort("timestamp")

    assert out.height == 2
    assert out.get_column("trade_amount").to_list() == [10.0, None]
    assert out.get_column("quote_bid_price").to_list() == [71240.0, None]
    assert out.get_column("quote_ask_price").to_list() == [71260.0, None]
    assert out.get_column("mark_price").to_list() == [71250.0, 71275.0]
    assert "local_timestamp" not in out.columns
    assert "funding_timestamp" not in out.columns
    assert "quote_local_timestamp" not in out.columns


def test_create_deribit_aligned_derivative_ticker_trades_quotes_errors_when_any_input_is_missing(monkeypatch):
    non_empty = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 26, 0, 0, 0)],
            "symbol": ["BTC-24APR26"],
            "mark_price": [71250.0],
            "trade_amount": [10.0],
            "trade_signed_amount": [-10.0],
            "trade_price_amount": [712500.0],
            "bid_price": [71240.0],
            "ask_price": [71260.0],
            "bid_amount": [5.0],
            "ask_amount": [7.0],
            "stale": [False],
        }
    )
    empty = pl.DataFrame()

    def fake_access_files(exchange, datatype, date, symbol_list, sample_freq, raw_data_dir):
        if datatype == "quotes":
            return empty, []
        return non_empty, ["BTC-24APR26"]

    monkeypatch.setattr(adttq, "access_files", fake_access_files)

    with pytest.raises(AssertionError, match="Missing quotes data"):
        adttq.create_deribit_aligned_derivative_ticker_trades_quotes(
            date="2026-03-26",
            symbol="BTC-24APR26",
            sample_freq="5min",
            raw_data_dir="datasets/deribit_raw/",
        )


def test_create_aligned_derivative_ticker_trades_quotes_dispatches_deribit(monkeypatch):
    expected = pl.DataFrame(
        {"timestamp": [], "symbol": [], "mark_price": [], "exp": [], "ref_sym": [], "spot_sym": [], "inverse": []},
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "mark_price": pl.Float64,
            "exp": pl.Datetime("us"),
            "ref_sym": pl.String,
            "spot_sym": pl.String,
            "inverse": pl.Boolean,
        },
    )

    def fake_create(date, symbol, sample_freq, raw_data_dir):
        assert date == "2026-03-26"
        assert symbol == "BTC-24APR26"
        assert sample_freq == "15min"
        assert raw_data_dir == "custom/deribit_raw/"
        return expected

    monkeypatch.setattr(adttq, "create_deribit_aligned_derivative_ticker_trades_quotes", fake_create)

    out = adttq.create_aligned_derivative_ticker_trades_quotes(
        "deribit",
        "2026-03-26",
        "BTC-24APR26",
        sample_freq="15min",
        raw_data_dir="custom/deribit_raw/",
    )

    assert out.equals(expected)


def test_create_aligned_derivative_ticker_trades_quotes_dispatches_okex(monkeypatch):
    expected = pl.DataFrame(
        {"timestamp": [], "symbol": [], "mark_price": []},
        schema={
            "timestamp": pl.Datetime("us"),
            "symbol": pl.String,
            "mark_price": pl.Float64,
        },
    )

    def fake_create(date, symbol, sample_freq, raw_data_dir):
        assert date == "2026-03-26"
        assert symbol == "BTC-USD-260327"
        assert sample_freq == "15min"
        assert raw_data_dir == "custom/okex_raw/"
        return expected

    monkeypatch.setattr(adttq, "create_okex_aligned_derivative_ticker_trades_quotes", fake_create)

    out = adttq.create_aligned_derivative_ticker_trades_quotes(
        "okex",
        "2026-03-26",
        "BTC-USD-260327",
        sample_freq="15min",
        raw_data_dir="custom/okex_raw/",
    )

    assert out.equals(expected)


def test_create_aligned_derivative_ticker_trades_quotes_prints_error_for_unknown_exchange(capsys):
    with pytest.raises(AssertionError):
        adttq.create_aligned_derivative_ticker_trades_quotes("binance", "2026-03-26", "BTC-24APR26")

    captured = capsys.readouterr()
    assert "Unsupported exchange: binance. Expected okex or deribit." in captured.out


def test_entry_point_dispatches_align_derivative_ticker_trades_quotes(monkeypatch):
    called = {}

    def fake_autocomplete(_parser):
        return None

    def fake_run_cli_args(args: argparse.Namespace):
        called["exchange"] = args.exchange
        called["date"] = args.date
        called["symbol"] = args.symbol

    monkeypatch.setattr(entry_point.argcomplete, "autocomplete", fake_autocomplete)
    monkeypatch.setattr(entry_point.align_derivative_ticker_trades_quotes, "run_cli_args", fake_run_cli_args)

    entry_point.run_cli([
        "align-derivative-ticker-trades-quotes",
        "--exchange", "deribit",
        "--date", "2026-03-26",
        "--symbol", "BTC-24APR26",
    ])

    assert called == {"exchange": "deribit", "date": "2026-03-26", "symbol": "BTC-24APR26"}