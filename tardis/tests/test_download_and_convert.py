from pathlib import Path
import io

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import pytest
import requests

from tardis import download_files as dac


def _quotes_chunk(exchange: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "exchange": [exchange] * 6,
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "timestamp": [
                1675209660000000,
                1675209720000000,
                1675210320000000,
                1675209780000000,
                1675209840000000,
                1675210080000000,
            ],
            "local_timestamp": [
                1675209660000000,
                1675209720000000,
                1675210320000000,
                1675209780000000,
                1675209840000000,
                1675210080000000,
            ],
            "bid_price": [100.0, 101.0, 103.0, 200.0, 201.0, 202.0],
            "ask_price": [101.0, 102.0, 104.0, 201.0, 202.0, 203.0],
            "volume": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        }
    )


def _trades_chunk(exchange: str, symbol: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "exchange": [exchange, exchange],
            "symbol": [symbol, symbol],
            "timestamp": [1, 2],
            "local_timestamp": [1, 2],
            "price": [100.0, 101.0],
            "amount": [0.1, 0.2],
            "side": ["buy", "sell"],
        }
    )


def test_streaming_resample_creates_daily_parquets(tmp_path, monkeypatch):
    exchange = "binance"
    data_type = "trades"
    symbol = "BTCUSDT"

    def fake_iter(url, api_key=None):
        yield _trades_chunk(exchange, symbol)

    monkeypatch.setattr(dac, "_iter_tardis_csv_rows_streaming", fake_iter)

    parquet_paths = dac.download_resample(
        exchange=exchange,
        data_type=data_type,
        symbols=symbol,
        start_date="2023-02-01",
        end_date="2023-02-02",
        data_dir=str(tmp_path),
        resample_freq="1min",
    )

    assert len(parquet_paths) == 2
    assert all(p.exists() for p in parquet_paths)
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        assert table.num_rows >= 1


def test_streaming_resample_rejects_inverted_date_range(tmp_path):
    with pytest.raises(ValueError):
        dac.download_resample(
            exchange="binance",
            data_type="trades",
            symbols="BTCUSDT",
            start_date="2023-02-05",
            end_date="2023-02-01",
            data_dir=str(tmp_path),
        )


def test_streaming_resample_skips_existing_parquet(tmp_path, monkeypatch):
    exchange = "binance"
    data_type = "trades"
    symbol = "BTCUSDT"
    parquet_path = tmp_path / f"{exchange}_{data_type}_2023-02-01_{symbol}_1min.parquet"

    pd.DataFrame({"symbol": [symbol], "timestamp": [1]}).to_parquet(parquet_path, index=False)

    called = {"count": 0}

    def fake_iter(url, api_key=None):
        called["count"] += 1
        yield _trades_chunk(exchange, symbol)

    monkeypatch.setattr(dac, "_iter_tardis_csv_rows_streaming", fake_iter)

    out = dac.download_resample(
        exchange=exchange,
        data_type=data_type,
        symbols=symbol,
        start_date="2023-02-01",
        end_date="2023-02-01",
        data_dir=str(tmp_path),
        force_reload=False,
        resample_freq="1min",
    )

    assert called["count"] == 0
    assert out == [parquet_path]


def test_streaming_resample_force_reload_overwrites(tmp_path, monkeypatch):
    exchange = "binance"
    data_type = "trades"
    symbol = "BTCUSDT"
    parquet_path = tmp_path / f"{exchange}_{data_type}_2023-02-01_{symbol}_1min.parquet"

    pd.DataFrame({"symbol": [symbol], "timestamp": [1], "trade_amount": [99.0]}).to_parquet(parquet_path, index=False)

    def fake_iter(url, api_key=None):
        yield _trades_chunk(exchange, symbol)

    monkeypatch.setattr(dac, "_iter_tardis_csv_rows_streaming", fake_iter)

    out = dac.download_resample(
        exchange=exchange,
        data_type=data_type,
        symbols=symbol,
        start_date="2023-02-01",
        end_date="2023-02-01",
        data_dir=str(tmp_path),
        force_reload=True,
        resample_freq="1min",
    )

    assert out == [parquet_path]
    table = pq.read_table(parquet_path)
    assert table.num_rows >= 1


def test_streaming_resample_writes_empty_parquet_on_404(tmp_path, monkeypatch):
    def fake_iter(url, api_key=None):
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError("not found", response=response)
        yield

    monkeypatch.setattr(dac, "_iter_tardis_csv_rows_streaming", fake_iter)

    out = dac.download_resample(
        exchange="binance",
        data_type="quotes",
        symbols="OPTIONS",
        start_date="2023-02-01",
        end_date="2023-02-01",
        data_dir=str(tmp_path),
        resample_freq="5min",
    )

    assert len(out) == 1
    assert out[0].exists()
    assert pl.read_parquet(out[0]).is_empty()


def test_streaming_resample_resamples_multi_symbol_chunk_before_writing(tmp_path, monkeypatch):
    exchange = "binance"
    data_type = "quotes"
    symbol = "OPTIONS"

    def fake_iter(url, api_key=None):
        yield _quotes_chunk(exchange)

    monkeypatch.setattr(dac, "_iter_tardis_csv_rows_streaming", fake_iter)

    out = dac.download_resample(
        exchange=exchange,
        data_type=data_type,
        symbols=symbol,
        start_date="2023-02-01",
        end_date="2023-02-01",
        data_dir=str(tmp_path),
        resample_freq="5min",
    )

    assert len(out) == 1
    assert out[0].name == "binance_quotes_2023-02-01_OPTIONS_5min.parquet"

    df = pl.read_parquet(out[0]).sort(["symbol", "timestamp"])
    assert set(df.get_column("symbol").unique().to_list()) == {"A", "B"}
    assert "stale" in df.columns

    df_a = df.filter(pl.col("symbol") == "A")
    assert df_a["timestamp"].head(3).to_list() == [
        pd.Timestamp("2023-02-01 00:00:00"),
        pd.Timestamp("2023-02-01 00:05:00"),
        pd.Timestamp("2023-02-01 00:10:00"),
    ]
    assert df_a["bid_price"].head(3).to_list() == [101.0, 101.0, 103.0]
    assert df_a["volume"].head(3).to_list() == [3.0, None, 3.0]
    assert df_a["stale"].head(3).to_list() == [False, True, False]


def test_download_resample_accepts_symbol_list_and_skips_invalid_entries(tmp_path, monkeypatch, caplog):
    exchange = "binance"
    data_type = "trades"

    def fake_iter(url, api_key=None):
        yield _trades_chunk(exchange, "BTCUSDT")

    monkeypatch.setattr(dac, "_iter_tardis_csv_rows_streaming", fake_iter)

    out = dac.download_resample(
        exchange=exchange,
        data_type=data_type,
        symbols=["BTCUSDT", "", None, "ETHUSDT"],
        start_date="2023-02-01",
        end_date="2023-02-01",
        data_dir=str(tmp_path),
        resample_freq="1min",
    )

    assert len(out) == 2
    assert all(p.exists() for p in out)
    warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("Skipping invalid empty symbol entry" in m for m in warning_messages)
    assert any("Skipping invalid symbol" in m for m in warning_messages)


def test_options_chain_strike_price_is_forced_to_float64_for_integer_like_csv_values():
    csv_data = b"timestamp,strike_price,symbol\n1711929600000000,67000,TEST\n"
    reader = pv.open_csv(
        io.BytesIO(csv_data),
        convert_options=pv.ConvertOptions(
            column_types=dac._temporal_fallback_column_types(dac.TARDIS_COLUMN_TYPES),
            null_values=["", "null", "None", "NULL", "NaN", "nan"],
        ),
    )

    batch = next(iter(reader))
    df = pl.from_arrow(pa.Table.from_batches([batch]))
    df = dac._cast_temporal_fallback_columns(df, dac.TARDIS_COLUMN_TYPES)

    assert df.schema["strike_price"] == pl.Float64
    assert df.get_column("strike_price").to_list() == [67000.0]


def test_peek_streaming_raw_logs_transposed_pandas_preview(monkeypatch, caplog):
    chunk = pl.DataFrame(
        {
            "exchange": ["deribit", "deribit"],
            "symbol": ["OPTIONS", "OPTIONS"],
            "timestamp": [1, 2],
            "bid_price": [100.0, 101.0],
            "ask_price": [101.0, 102.0],
        }
    )

    def fake_iter(url, api_key=None):
        yield chunk

    monkeypatch.setattr(dac, "_iter_tardis_csv_rows_streaming", fake_iter)

    with caplog.at_level("INFO"):
        dac.peek_streaming_raw(
            exchange="deribit",
            data_type="quotes",
            symbol="OPTIONS",
            start_date="2023-02-01",
            head_rows=2,
        )

    log_text = caplog.text
    assert "peek preview (transposed, first 2 rows):" in log_text
    assert "exchange   deribit  deribit" in log_text
    assert "bid_price    100.0    101.0" in log_text
    assert "ask_price    101.0    102.0" in log_text


def test_peek_streaming_raw_can_log_non_transposed_preview(monkeypatch, caplog):
    chunk = pl.DataFrame(
        {
            "exchange": ["deribit", "deribit"],
            "symbol": ["OPTIONS", "OPTIONS"],
            "timestamp": [1, 2],
            "bid_price": [100.0, 101.0],
        }
    )

    def fake_iter(url, api_key=None):
        yield chunk

    monkeypatch.setattr(dac, "_iter_tardis_csv_rows_streaming", fake_iter)

    with caplog.at_level("INFO"):
        dac.peek_streaming_raw(
            exchange="deribit",
            data_type="quotes",
            symbol="OPTIONS",
            start_date="2023-02-01",
            head_rows=2,
            transpose_preview=False,
        )

    log_text = caplog.text
    assert "peek preview (table, first 2 rows):" in log_text
    assert "exchange   symbol  timestamp  bid_price" in log_text
    assert "deribit  OPTIONS          1      100.0" in log_text


def test_download_files_cli_parser_defaults_peek_transposed_to_true():
    parser = dac.build_cli_parser()

    args = parser.parse_args(
        [
            "--exchange", "deribit",
            "--data-type", "quotes",
            "--symbol", "OPTIONS",
            "--start-date", "2023-02-01",
        ]
    )

    assert args.peek_transposed is True


def test_download_files_cli_parser_defaults_resample_freq_to_5min():
    parser = dac.build_cli_parser()

    args = parser.parse_args(
        [
            "--exchange", "deribit",
            "--data-type", "quotes",
            "--symbol", "OPTIONS",
            "--start-date", "2023-02-01",
        ]
    )

    assert args.resample_freq == dac.DEFAULT_RESAMPLE_FREQ


def test_download_files_cli_args_pass_5min_when_resample_freq_omitted(monkeypatch):
    calls: dict[str, object] = {}

    def fake_download_resample(**kwargs):
        calls["resample_freq"] = kwargs["resample_freq"]
        return []

    monkeypatch.setattr(dac, "download_resample", fake_download_resample)

    parser = dac.build_cli_parser()
    args = parser.parse_args(
        [
            "--exchange", "deribit",
            "--data-type", "quotes",
            "--symbol", "OPTIONS",
            "--start-date", "2023-02-01",
        ]
    )

    dac.run_cli_args(args)

    assert calls["resample_freq"] == dac.DEFAULT_RESAMPLE_FREQ
