import pytest
import polars as pl

from tardis import markup_files as mf


def test_mark_up_adds_placeholder_columns_in_place(caplog):
    """Test that unrecognized symbols issue a warning and pass through."""
    df = pl.DataFrame(
        {
            "symbol": ["BTC-USDT", "ETH-30JUN26-2500-C"],
            "timestamp": [1, 2],
            "value": [10.0, 11.0],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="quotes")

    # With retired default, unrecognized symbols pass through unchanged
    assert out.height == 2
    assert "symbol" in out.columns
    assert "timestamp" in out.columns
    assert "value" in out.columns
    # Placeholder columns should NOT be added for unrecognized symbols
    for col in ["CUR1", "CUR2", "exp", "exp_str", "strike", "pc", "type", "ref_sym", "fut_sym", "spot_sym"]:
        assert col not in out.columns
    # Should log the retired warning
    assert "Using default (retired) symbol markup" in caplog.text



def test_mark_up_okex_options_quotes_parses_symbol_fields():
    df = pl.DataFrame(
        {
            "symbol": ["ETH-USD-260529-1600-P"],
            "timestamp": [1],
        }
    )

    out = mf.mark_up(df, exchange="okex-options", data_type="quotes")

    assert out.get_column("CUR1").to_list() == ["ETH"]
    assert out.get_column("CUR2").to_list() == ["USD"]
    assert out.get_column("exp_str").to_list() == ["260529"]
    assert out.get_column("strike").to_list() == [1600.0]
    assert out.get_column("pc").to_list() == ["P"]
    assert out.get_column("ref_sym").to_list() == ["ETHUSD"]
    assert out.get_column("fut_sym").to_list() == ["ETH-USDT-260529"]
    assert out.get_column("spot_sym").to_list() == ["ETH-USDT"]
    assert out.get_column("type").to_list() == ["option"]
    assert out.get_column("inverse").to_list() == [False]


def test_mark_up_okex_options_trades_parses_symbol_fields():
    df = pl.DataFrame(
        {
            "symbol": ["ETH-USD-260529-1600-P"],
            "timestamp": [1],
            "price": [12.5],
        }
    )

    out = mf.mark_up(df, exchange="okex-options", data_type="trades")

    assert out.get_column("CUR1").to_list() == ["ETH"]
    assert out.get_column("CUR2").to_list() == ["USD"]
    assert out.get_column("exp_str").to_list() == ["260529"]
    assert out.get_column("strike").to_list() == [1600.0]
    assert out.get_column("pc").to_list() == ["P"]
    assert out.get_column("type").to_list() == ["option"]
    assert out.get_column("inverse").to_list() == [False]
    # Original columns still present
    assert "timestamp" in out.columns
    assert "price" in out.columns


def test_mark_up_okex_futures_quotes_parses_symbol_fields():
    df = pl.DataFrame(
        {
            "symbol": ["BTC-USD-260925", "ETH-USDC-260925"],
            "timestamp": [1, 2],
        }
    )

    out = mf.mark_up(df, exchange="okex-futures", data_type="quotes")

    assert out.get_column("CUR1").to_list() == ["BTC", "ETH"]
    assert out.get_column("CUR2").to_list() == ["USD", "USDC"]
    assert out.get_column("exp_str").to_list() == ["260925", "260925"]
    assert out.get_column("ref_sym").to_list() == ["BTCUSD", "ETHUSD"]
    assert out.get_column("spot_sym").to_list() == ["BTC-USDT", "ETH-USDC"]
    assert out.get_column("inverse").to_list() == [True, False]
    assert "fut_sym" not in out.columns
    assert "strike" not in out.columns
    assert "pc" not in out.columns


def test_mark_up_okex_spot_quotes_parses_symbol_fields():
    df = pl.DataFrame(
        {
            "symbol": ["ETH-USDC", "BTC-USD"],
            "timestamp": [1, 2],
        }
    )

    out = mf.mark_up(df, exchange="okex", data_type="quotes")

    assert out.get_column("CUR1").to_list() == ["ETH", "BTC"]
    assert out.get_column("CUR2").to_list() == ["USDC", "USD"]
    assert out.get_column("ref_sym").to_list() == ["ETHUSD", "BTCUSD"]
    assert out.get_column("spot_sym").to_list() == ["ETH-USDC", "BTC-USD"]
    assert "exp" not in out.columns
    assert "exp_str" not in out.columns


def test_mark_up_deribit_future_with_explicit_symbol_type_works():
    df = pl.DataFrame(
        {
            "symbol": ["ETH_USDT-6MAR26"],
            "timestamp": [1],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="quotes", symbol_type="futures")

    assert out.get_column("CUR1").to_list() == ["ETH"]
    assert out.get_column("CUR2").to_list() == ["USDT"]
    assert out.get_column("exp_str").to_list() == ["6MAR26"]


def test_mark_up_deribit_option_symbol_type_pattern_mismatch_raises():
    df = pl.DataFrame(
        {
            "symbol": ["ETH-USD"],
            "timestamp": [1],
        }
    )

    with pytest.raises(AssertionError):
        mf.mark_up(df, exchange="deribit", data_type="quotes", symbol_type="option")


def test_mark_up_deribit_trades_future_parses_usd_fallback_and_usdc_root():
    df = pl.DataFrame(
        {
            "symbol": ["ETH_USDC-27MAR26", "ETH-27MAR26"],
            "timestamp": [1, 2],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="trades", symbol_type="future")

    assert out.get_column("CUR1").to_list() == ["ETH", "ETH"]
    assert out.get_column("CUR2").to_list() == ["USDC", "USD"]
    assert out.get_column("exp_str").to_list() == ["27MAR26", "27MAR26"]
    assert out.get_column("ref_sym").to_list() == ["ETHUSD", "ETHUSD"]
    assert out.get_column("spot_sym").to_list() == ["ETH-USDC", "ETH-USD"]
    assert out.get_column("inverse").to_list() == [True, True]


def test_mark_up_deribit_trades_spot_matches_okex_spot_trades_columns():
    df = pl.DataFrame(
        {
            "symbol": ["ETH_USDC"],
            "timestamp": [1],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="trades", symbol_type="spot")

    assert out.get_column("CUR1").to_list() == ["ETH"]
    assert out.get_column("CUR2").to_list() == ["USDC"]
    assert out.get_column("ref_sym").to_list() == ["ETHUSD"]
    assert out.get_column("spot_sym").to_list() == ["ETH_USDC"]


def test_mark_up_deribit_trades_option_parses_d_strike_and_defaults_cur2():
    df = pl.DataFrame(
        {
            "symbol": ["XRP_USDC-2MAR26-1D375-P"],
            "timestamp": [1],
            "price": [0.2],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="trades", symbol_type="option")

    assert out.get_column("CUR1").to_list() == ["XRP"]
    assert out.get_column("CUR2").to_list() == ["USDC"]
    assert out.get_column("ref_sym").to_list() == ["XRPUSD"]
    assert out.get_column("fut_sym").to_list() == ["XRP_USDC-2MAR26"]
    assert out.get_column("spot_sym").to_list() == ["XRP_USDC"]
    assert out.get_column("inverse").to_list() == [True]
    assert out.get_column("strike").to_list() == [1.375]
    assert out.get_column("pc").to_list() == ["P"]
    assert out.get_column("type").to_list() == ["option"]
    # Original columns still present
    assert "timestamp" in out.columns
    assert "price" in out.columns


def test_mark_up_deribit_quotes_future_parses_usd_fallback_and_usdc_root():
    df = pl.DataFrame(
        {
            "symbol": ["ETH_USDC-27MAR26", "ETH-27MAR26"],
            "timestamp": [1, 2],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="quotes", symbol_type="future")

    assert out.get_column("CUR1").to_list() == ["ETH", "ETH"]
    assert out.get_column("CUR2").to_list() == ["USDC", "USD"]
    assert out.get_column("exp_str").to_list() == ["27MAR26", "27MAR26"]
    assert out.get_column("ref_sym").to_list() == ["ETHUSD", "ETHUSD"]
    assert out.get_column("spot_sym").to_list() == ["ETH-USDC", "ETH-USD"]
    assert out.get_column("inverse").to_list() == [True, True]


def test_mark_up_deribit_quotes_future_ref_sym_uses_cur1cur2_for_non_usd_prefix():
    df = pl.DataFrame(
        {
            "symbol": ["ETH_BTC-27MAR26"],
            "timestamp": [1],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="quotes", symbol_type="future")

    assert out.get_column("CUR1").to_list() == ["ETH"]
    assert out.get_column("CUR2").to_list() == ["BTC"]
    assert out.get_column("ref_sym").to_list() == ["ETHBTC"]


def test_mark_up_deribit_trades_future_ref_sym_uses_cur1cur2_for_non_usd_prefix():
    df = pl.DataFrame(
        {
            "symbol": ["ETH_BTC-27MAR26"],
            "timestamp": [1],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="trades", symbol_type="future")

    assert out.get_column("CUR1").to_list() == ["ETH"]
    assert out.get_column("CUR2").to_list() == ["BTC"]
    assert out.get_column("ref_sym").to_list() == ["ETHBTC"]
    assert out.get_column("inverse").to_list() == [True]


def test_mark_up_deribit_quotes_option_parses_d_strike_and_defaults_cur2():
    df = pl.DataFrame(
        {
            "symbol": ["XRP_USDC-2MAR26-1D375-P", "ETH-27MAR26-2D000-C"],
            "timestamp": [1, 2],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="quotes", symbol_type="option")

    assert out.get_column("CUR1").to_list() == ["XRP", "ETH"]
    assert out.get_column("CUR2").to_list() == ["USDC", "USD"]
    assert out.get_column("exp_str").to_list() == ["2MAR26", "27MAR26"]
    assert out.get_column("strike").to_list() == [1.375, 2.0]
    assert out.get_column("pc").to_list() == ["P", "C"]
    assert out.get_column("ref_sym").to_list() == ["XRPUSD", "ETHUSD"]
    assert out.get_column("fut_sym").to_list() == ["XRP_USDC-2MAR26", "ETH-27MAR26"]
    assert out.get_column("spot_sym").to_list() == ["XRP_USDC", "ETH_USDC"]
    assert out.get_column("inverse").to_list() == [True, True]


def test_mark_up_deribit_quotes_option_ref_sym_uses_cur1cur2_for_non_usd_prefix():
    df = pl.DataFrame(
        {
            "symbol": ["XRP_BTC-2MAR26-1D375-P"],
            "timestamp": [1],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="quotes", symbol_type="option")

    assert out.get_column("CUR1").to_list() == ["XRP"]
    assert out.get_column("CUR2").to_list() == ["BTC"]
    assert out.get_column("ref_sym").to_list() == ["XRPBTC"]
    assert out.get_column("inverse").to_list() == [True]


def test_mark_up_deribit_trades_option_ref_sym_uses_cur1cur2_for_non_usd_prefix():
    df = pl.DataFrame(
        {
            "symbol": ["XRP_BTC-2MAR26-1D375-P"],
            "timestamp": [1],
            "price": [0.2],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="trades", symbol_type="option")

    assert out.get_column("CUR1").to_list() == ["XRP"]
    assert out.get_column("CUR2").to_list() == ["BTC"]
    assert out.get_column("ref_sym").to_list() == ["XRPBTC"]
    assert out.get_column("inverse").to_list() == [True]


def test_mark_up_deribit_quotes_spot_matches_okex_shape():
    df = pl.DataFrame(
        {
            "symbol": ["ETH_USDC"],
            "timestamp": [1],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="quotes", symbol_type="spot")

    assert out.get_column("CUR1").to_list() == ["ETH"]
    assert out.get_column("CUR2").to_list() == ["USDC"]
    assert out.get_column("ref_sym").to_list() == ["ETHUSD"]
    assert out.get_column("spot_sym").to_list() == ["ETH_USDC"]


def test_mark_up_deribit_quotes_spot_ref_sym_uses_cur1cur2_for_non_usd_prefix():
    df = pl.DataFrame(
        {
            "symbol": ["ETH_BTC"],
            "timestamp": [1],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="quotes", symbol_type="spot")

    assert out.get_column("CUR1").to_list() == ["ETH"]
    assert out.get_column("CUR2").to_list() == ["BTC"]
    assert out.get_column("ref_sym").to_list() == ["ETHBTC"]
    assert out.get_column("spot_sym").to_list() == ["ETH_BTC"]


def test_mark_up_deribit_trades_spot_ref_sym_uses_cur1cur2_for_non_usd_prefix():
    df = pl.DataFrame(
        {
            "symbol": ["ETH_BTC"],
            "timestamp": [1],
        }
    )

    out = mf.mark_up(df, exchange="deribit", data_type="trades", symbol_type="spot")

    assert out.get_column("CUR1").to_list() == ["ETH"]
    assert out.get_column("CUR2").to_list() == ["BTC"]
    assert out.get_column("ref_sym").to_list() == ["ETHBTC"]
    assert out.get_column("spot_sym").to_list() == ["ETH_BTC"]


def test_mark_up_returns_empty_input_unchanged():
    df = pl.DataFrame(schema={"symbol": pl.String, "timestamp": pl.Int64})

    out = mf.mark_up(df, exchange="deribit", data_type="quotes")

    assert out.is_empty()


def test_mark_up_uses_unique_exchange_from_dataframe_when_exchange_is_none(caplog):
    """Test that mark_up uses exchange from dataframe when not provided."""
    df = pl.DataFrame(
        {
            "exchange": ["deribit", "deribit"],
            "symbol": ["ETH_USDC-27MAR26", "ETH-27MAR26"],
            "timestamp": [1, 2],
        }
    )

    out = mf.mark_up(df, exchange=None, data_type="quotes", symbol_type="future")

    assert out.height == 2
    # Should have parsed the deribit futures symbols
    assert "CUR1" in out.columns
    assert out.get_column("CUR1").to_list() == ["ETH", "ETH"]


def test_mark_up_asserts_on_exchange_mismatch():
    df = pl.DataFrame(
        {
            "exchange": ["deribit", "deribit"],
            "symbol": ["BTC-USDT", "ETH-30JUN26-2500-C"],
        }
    )

    with pytest.raises(AssertionError):
        mf.mark_up(df, exchange="okex", data_type="quotes")


def test_access_files_returns_dataframe_and_symbols(monkeypatch, tmp_path):
    parquet_path = tmp_path / "sample.parquet"
    pl.DataFrame(
        {
            "symbol": ["BTC-USDT", "ETH-30JUN26-2500-C", "BTC-USDT"],
            "timestamp": [1, 2, 3],
            "value": [10.0, 11.0, 12.0],
        }
    ).write_parquet(parquet_path)

    def fake_download_resample(**kwargs):
        return [parquet_path]

    monkeypatch.setattr(mf, "download_resample", fake_download_resample)

    out, symbols = mf.access_files(
        exchange="deribit",
        datatype="quotes",
        date="2026-03-01",
        symbol_list=["BTC-USDT", "ETH-30JUN26-2500-C"],
    )

    assert symbols == ["BTC-USDT", "ETH-30JUN26-2500-C"]
    assert out.height == 3
    assert "CUR1" not in out.columns
