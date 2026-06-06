import argparse
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from tardis import align_put_call_quotes_trades_chain as apcqtc
from tardis import entry_point


def _base_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    t1 = datetime(2026, 4, 10, 12, 0, 0)
    t2 = datetime(2026, 4, 10, 12, 5, 0)
    exp = datetime(2026, 4, 10, 8, 0, 0)

    call_symbol = "BTC_USDC-10APR26-100000-C"
    put_symbol = "BTC_USDC-10APR26-100000-P"
    fut_symbol = "BTC_USDC-10APR26"
    spot_symbol = "BTC_USDC"

    df_opt_quotes = pl.DataFrame(
        {
            "timestamp": [t1, t1, t2, t2],
            "symbol": [call_symbol, put_symbol, call_symbol, put_symbol],
            "exp": [exp, exp, exp, exp],
            "strike": [100000.0, 100000.0, 100000.0, 100000.0],
            "exchange": ["deribit", "deribit", "deribit", "deribit"],
            "inverse": [True, True, True, True],
            "fut_sym": [fut_symbol, fut_symbol, fut_symbol, fut_symbol],
            "ref_sym": ["BTCUSD", "BTCUSD", "BTCUSD", "BTCUSD"],
            "spot_sym": [spot_symbol, spot_symbol, spot_symbol, spot_symbol],
            "pc": ["C", "P", "C", "P"],
            "bid_price": [0.10, 0.20, 0.11, 0.21],
            "ask_price": [0.12, 0.22, 0.13, 0.23],
            "bid_amount": [5.0, 6.0, 7.0, 8.0],
            "ask_amount": [9.0, 10.0, 11.0, 12.0],
            "stale": [False, False, False, False],
        }
    )

    df_opt_trades = pl.DataFrame(
        {
            "timestamp": [t1, t1],
            "symbol": [call_symbol, put_symbol],
            "trade_amount": [10.0, 20.0],
            "trade_signed_amount": [10.0, -20.0],
            "trade_price_amount": [1.15, 4.20],
        }
    )

    df_option_chain = pl.DataFrame(
        {
            "timestamp": [t1, t1, t2, t2],
            "symbol": [call_symbol, put_symbol, call_symbol, put_symbol],
            "open_interest": [111.0, 211.0, 112.0, 212.0],
        }
    )

    df_fut = pl.DataFrame(
        {
            "timestamp": [t1, t2],
            "symbol": [fut_symbol, fut_symbol],
            "bid_price": [100000.0, 100100.0],
            "ask_price": [100010.0, 100110.0],
            "bid_amount": [1.0, 1.5],
            "ask_amount": [2.0, 2.5],
            "stale": [False, False],
        }
    )

    df_spot = pl.DataFrame(
        {
            "timestamp": [t1, t2],
            "symbol": [spot_symbol, spot_symbol],
            "bid_price": [99900.0, 99950.0],
            "ask_price": [99910.0, 99960.0],
            "bid_amount": [3.0, 3.5],
            "ask_amount": [4.0, 4.5],
            "stale": [False, False],
        }
    )

    return df_opt_quotes, df_opt_trades, df_option_chain, df_fut, df_spot


def test_align_put_call_quotes_trades_chain_adds_leg_trade_and_open_interest_columns():
    df_opt_quotes, df_opt_trades, df_option_chain, df_fut, df_spot = _base_inputs()

    out = apcqtc.align_put_call_quotes_trades_chain(
        df_opt_quotes,
        df_opt_trades,
        df_option_chain,
        df_fut,
        df_spot,
    ).sort("timestamp")

    assert out.height == 2

    assert out.get_column("call_trade_amount").to_list() == [10.0, None]
    assert out.get_column("call_trade_signed_amount").to_list() == [10.0, None]
    assert out.get_column("call_trade_price_amount").to_list() == [1.15, None]

    assert out.get_column("put_trade_amount").to_list() == [20.0, None]
    assert out.get_column("put_trade_signed_amount").to_list() == [-20.0, None]
    assert out.get_column("put_trade_price_amount").to_list() == [4.20, None]

    assert out.get_column("call_open_interest").to_list() == [111.0, 112.0]
    assert out.get_column("put_open_interest").to_list() == [211.0, 212.0]

    assert out.get_column("call_bid_price").to_list() == [0.10, 0.11]
    assert out.get_column("put_bid_price").to_list() == [0.20, 0.21]


def test_create_aligned_put_call_quotes_trades_chain_dispatches_okex(monkeypatch):
    expected = pl.DataFrame({"timestamp": [], "call_symbol": []}, schema={"timestamp": pl.Datetime("us"), "call_symbol": pl.String})

    def fake_create(date, sample_freq, raw_data_dir):
        assert date == "2026-04-10"
        assert sample_freq == "5min"
        assert raw_data_dir == "datasets/okex_raw/"
        return expected

    monkeypatch.setattr(apcqtc, "create_okex_aligned_put_call_quotes_trades_chain", fake_create)

    out = apcqtc.create_aligned_put_call_quotes_trades_chain("okex", "2026-04-10")

    assert out.equals(expected)


def test_create_aligned_put_call_quotes_trades_chain_dispatches_deribit(monkeypatch):
    expected = pl.DataFrame({"timestamp": [], "call_symbol": []}, schema={"timestamp": pl.Datetime("us"), "call_symbol": pl.String})

    def fake_create(date, sample_freq, raw_data_dir):
        assert date == "2026-04-10"
        assert sample_freq == "15min"
        assert raw_data_dir == "custom/deribit_raw/"
        return expected

    monkeypatch.setattr(apcqtc, "create_deribit_aligned_put_call_quotes_trades_chain", fake_create)

    out = apcqtc.create_aligned_put_call_quotes_trades_chain(
        "deribit",
        "2026-04-10",
        sample_freq="15min",
        raw_data_dir="custom/deribit_raw/",
    )

    assert out.equals(expected)


def test_create_aligned_put_call_quotes_trades_chain_prints_error_for_unsupported_exchange(capsys):
    with pytest.raises(AssertionError):
        apcqtc.create_aligned_put_call_quotes_trades_chain("binance", "2026-04-10")

    captured = capsys.readouterr()
    assert "Unsupported exchange: binance. Expected okex or deribit." in captured.out


def test_create_deribit_aligned_put_call_quotes_trades_chain_errors_when_options_chain_is_missing(monkeypatch):
    non_empty = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 4, 10, 12, 0, 0)],
            "symbol": ["BTC_USDC-10APR26-100000-C"],
            "open_interest": [111.0],
            "trade_amount": [10.0],
            "trade_signed_amount": [10.0],
            "trade_price_amount": [1.15],
            "bid_price": [0.10],
            "ask_price": [0.12],
            "bid_amount": [5.0],
            "ask_amount": [9.0],
            "stale": [False],
            "exp": [datetime(2026, 4, 10, 8, 0, 0)],
            "strike": [100000.0],
            "exchange": ["deribit"],
            "inverse": [True],
            "fut_sym": ["BTC_USDC-10APR26"],
            "ref_sym": ["BTCUSD"],
            "spot_sym": ["BTC_USDC"],
            "pc": ["C"],
        }
    )
    empty = pl.DataFrame()

    def fake_access_files(exchange, datatype, date, symbol_list, sample_freq, raw_data_dir):
        if datatype == "options_chain":
            return empty, []
        return non_empty, ["OPTIONS"]

    monkeypatch.setattr(apcqtc, "access_files", fake_access_files)
    monkeypatch.setattr(apcqtc, "mark_up", lambda df, *_args, **_kwargs: df)

    with pytest.raises(AssertionError, match="Missing options_chain data"):
        apcqtc.create_deribit_aligned_put_call_quotes_trades_chain(
            date="2026-04-10",
            sample_freq="5min",
            raw_data_dir="datasets/deribit_raw/",
        )


def test_run_cli_args_skips_when_output_exists_and_no_recreate(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "aligned.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("already here", encoding="utf-8")

    args = apcqtc.build_cli_parser().parse_args(
        [
            "--exchange", "okex",
            "--date", "2026-04-10",
            "--output", str(output_path),
        ]
    )

    monkeypatch.setattr(apcqtc, "create_aligned_put_call_quotes_trades_chain", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not run")))

    apcqtc.run_cli_args(args)

    captured = capsys.readouterr()
    assert f"skipped_existing={output_path}" in captured.out


def test_entry_point_dispatches_align_put_call_quotes_trades_chain(monkeypatch):
    called = {}

    def fake_autocomplete(_parser):
        return None

    def fake_run_cli_args(args: argparse.Namespace):
        called["exchange"] = args.exchange
        called["date"] = args.date

    monkeypatch.setattr(entry_point.argcomplete, "autocomplete", fake_autocomplete)
    monkeypatch.setattr(entry_point.align_put_call_quotes_trades_chain, "run_cli_args", fake_run_cli_args)

    entry_point.run_cli([
        "align-put-call-quotes-trades-chain",
        "--exchange", "okex",
        "--date", "2026-04-10",
    ])

    assert called == {"exchange": "okex", "date": "2026-04-10"}
