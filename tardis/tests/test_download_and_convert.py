def test_download_and_convert_vs_streaming_resample_equivalence(tmp_path, monkeypatch):
    """
    Ensure download_and_convert and download_and_convert_streaming_resample produce identical parquet output.
    """
    exchange = "binance"
    data_type = "quotes"
    symbol = "OPTIONS"
    day = "2023-02-01"
    resample_freq = "5min"

    # Use a dedicated test_datasets subdirectory for all test files
    test_root = tmp_path / "test_datasets"
    test_root.mkdir()
    batch_dir = test_root / "batch"
    stream_dir = test_root / "stream"
    batch_dir.mkdir()
    stream_dir.mkdir()

    # Deterministic fake download writes a multi-symbol CSV
    def fake_download(*, exchange, data_types, from_date, to_date, symbols, download_dir, **kwargs):
        csv_path = Path(download_dir) / f"{exchange}_{data_types[0]}_{day}_{symbols[0]}.csv.gz"
        df = pd.DataFrame(
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
        df.to_csv(csv_path, index=False, compression="gzip")

    monkeypatch.setattr(dac.datasets, "download", fake_download)

    # Run batch (chunked) path
    out_batch = dac.download_and_convert(
        exchange=exchange,
        data_type=data_type,
        symbol=symbol,
        start_date=day,
        end_date=day,
        data_dir=str(batch_dir),
        resample_freq=resample_freq,
        cleanup_csv=False,  # Keep CSV for streaming mock
    )

    # Patch _iter_tardis_csv_rows_streaming to read the same CSV as above, but from batch_dir
    def fake_iter_tardis_csv_rows_streaming(url, api_key=None):
        import gzip
        csv_path = batch_dir / f"{exchange}_{data_type}_{day}_{symbol}.csv.gz"
        with gzip.open(csv_path, "rb") as f:
            for batch in pl.scan_csv(f).collect_batches(chunk_size=10_000):
                yield batch

    monkeypatch.setattr(dac, "_iter_tardis_csv_rows_streaming", fake_iter_tardis_csv_rows_streaming)

    out_stream = dac.download_and_convert_streaming_resample(
        exchange=exchange,
        data_type=data_type,
        symbol=symbol,
        start_date=day,
        end_date=day,
        data_dir=str(stream_dir),
        resample_freq=resample_freq,
    )

    # Compare output parquet files
    assert len(out_batch) == len(out_stream) == 1
    df_batch = pl.read_parquet(out_batch[0]).sort(["symbol", "timestamp"])
    df_stream = pl.read_parquet(out_stream[0]).sort(["symbol", "timestamp"])

    # Compare columns and values (ignore metadata differences)
    assert df_batch.columns == df_stream.columns
    pd.testing.assert_frame_equal(
        df_batch.to_pandas().reset_index(drop=True),
        df_stream.to_pandas().reset_index(drop=True),
        check_dtype=False,
        check_like=True,
    )
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

from tardis import download_files as dac


def _write_day_csv(base_dir: Path, exchange: str, data_type: str, day: str, symbol: str) -> Path:
    csv_path = base_dir / f"{exchange}_{data_type}_{day}_{symbol}.csv.gz"
    df = pd.DataFrame(
        {
            "exchange": [exchange, exchange],
            "symbol": [symbol, symbol],
            "timestamp": [1, 2],
            "local_timestamp": [1, 2],
            "price": [100.0, 101.0],
            "quantity": [0.1, 0.2],
            "side": ["buy", "sell"],
        }
    )
    df.to_csv(csv_path, index=False, compression="gzip")
    return csv_path


def test_download_and_convert_creates_daily_parquets(tmp_path, monkeypatch):
    exchange = "binance"
    data_type = "trades"
    symbol = "BTCUSDT"
    start_date = "2023-02-01"
    end_date = "2023-02-02"

    call_log = {}

    def fake_download(*, exchange, data_types, from_date, to_date, symbols, download_dir, **kwargs):
        call_log["exchange"] = exchange
        call_log["data_types"] = data_types
        call_log["from_date"] = from_date
        call_log["to_date"] = to_date
        call_log["symbols"] = symbols
        call_log["download_dir"] = download_dir

        _write_day_csv(Path(download_dir), exchange, data_types[0], "2023-02-01", symbols[0])
        _write_day_csv(Path(download_dir), exchange, data_types[0], "2023-02-02", symbols[0])

    monkeypatch.setattr(dac.datasets, "download", fake_download)

    parquet_paths = dac.download_and_convert(
        exchange=exchange,
        data_type=data_type,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_dir=str(tmp_path),
    )

    assert len(parquet_paths) == 2
    assert all(p.exists() for p in parquet_paths)
    assert [p.name for p in parquet_paths] == [
        "binance_trades_2023-02-01_BTCUSDT.parquet",
        "binance_trades_2023-02-02_BTCUSDT.parquet",
    ]

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        assert table.num_rows == 2
        assert "price" in table.column_names
        assert "quantity" in table.column_names

    assert call_log["exchange"] == "binance"
    assert call_log["data_types"] == ["trades"]
    assert call_log["from_date"] == "2023-02-01"
    assert call_log["to_date"] == "2023-02-03"
    assert call_log["symbols"] == ["BTCUSDT"]
    assert Path(call_log["download_dir"]) == tmp_path


def test_download_and_convert_rejects_inverted_date_range(tmp_path):
    with pytest.raises(ValueError):
        dac.download_and_convert(
            exchange="binance",
            data_type="trades",
            symbol="BTCUSDT",
            start_date="2023-02-05",
            end_date="2023-02-01",
            data_dir=str(tmp_path),
        )


def test_download_and_convert_skips_download_when_parquet_exists(tmp_path, monkeypatch):
    exchange = "binance"
    data_type = "trades"
    symbol = "BTCUSDT"
    start_date = "2023-02-01"
    end_date = "2023-02-01"

    stem = f"{exchange}_{data_type}_2023-02-01_{symbol}"
    parquet_path = tmp_path / f"{stem}.parquet"

    pd.DataFrame(
        {
            "exchange": [exchange],
            "symbol": [symbol],
            "timestamp": [1],
            "local_timestamp": [1],
            "price": [100.0],
            "quantity": [0.1],
            "side": ["buy"],
        }
    ).to_parquet(parquet_path, index=False)

    called = {"count": 0}

    def fake_download(**kwargs):
        called["count"] += 1

    monkeypatch.setattr(dac.datasets, "download", fake_download)

    out = dac.download_and_convert(
        exchange=exchange,
        data_type=data_type,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_dir=str(tmp_path),
        force_reload=False,
    )

    assert called["count"] == 0
    assert out == [parquet_path]


def test_download_and_convert_force_reload_and_cleanup(tmp_path, monkeypatch):
    exchange = "binance"
    data_type = "trades"
    symbol = "BTCUSDT"
    start_date = "2023-02-01"
    end_date = "2023-02-01"

    stem = f"{exchange}_{data_type}_2023-02-01_{symbol}"
    csv_path = tmp_path / f"{stem}.csv.gz"
    parquet_path = tmp_path / f"{stem}.parquet"

    pd.DataFrame(
        {
            "exchange": [exchange],
            "symbol": [symbol],
            "timestamp": [1],
            "local_timestamp": [1],
            "price": [10.0],
            "quantity": [0.1],
            "side": ["buy"],
        }
    ).to_parquet(parquet_path, index=False)

    download_calls = {"count": 0}

    def fake_download(*, exchange, data_types, from_date, to_date, symbols, download_dir, **kwargs):
        download_calls["count"] += 1
        _write_day_csv(Path(download_dir), exchange, data_types[0], "2023-02-01", symbols[0])

    monkeypatch.setattr(dac.datasets, "download", fake_download)

    out = dac.download_and_convert(
        exchange=exchange,
        data_type=data_type,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_dir=str(tmp_path),
        force_reload=True,
        cleanup_csv=True,
    )

    assert download_calls["count"] == 1
    assert out == [parquet_path]
    assert parquet_path.exists()
    assert not csv_path.exists()




def test_download_and_convert_resamples_multi_symbol_csv_before_writing(tmp_path, monkeypatch):
    exchange = "binance"
    data_type = "quotes"
    symbol = "OPTIONS"
    day = "2023-02-01"

    def fake_download(*, exchange, data_types, from_date, to_date, symbols, download_dir, **kwargs):
        csv_path = Path(download_dir) / f"{exchange}_{data_types[0]}_{day}_{symbols[0]}.csv.gz"
        df = pd.DataFrame(
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
        df.to_csv(csv_path, index=False, compression="gzip")

    monkeypatch.setattr(dac.datasets, "download", fake_download)

    out = dac.download_and_convert(
        exchange=exchange,
        data_type=data_type,
        symbol=symbol,
        start_date=day,
        end_date=day,
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
