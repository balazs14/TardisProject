from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

from tardis import download_and_convert as dac


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


def test_resample_midnight_grid_sum_amount_last_others_with_stale(tmp_path):
    src = tmp_path / "binance_trades_2023-02-01_BTCUSDT.parquet"
    df = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "timestamp": [
                1675209660000000,  # 2023-02-01 00:01:00
                1675209840000000,  # 2023-02-01 00:04:00
                1675210320000000,  # 2023-02-01 00:12:00
            ],
            "price": [10.0, 11.0, 13.0],
            "amount": [1.0, 2.0, 5.0],
        }
    )
    df.write_parquet(src)

    out = dac.resample(
        input_parquet_file=str(src),
        freq="5min",
        output_cols=["price", "amount"],
        summed_cols=["amount"],
    )

    assert src.with_name("binance_trades_2023-02-01_BTCUSDT_5min.parquet").exists()
    out = out.sort("timestamp")

    assert out.height >= 3

    first_three = out.head(3)
    ts = first_three["timestamp"].to_list()
    assert ts[0] == pd.Timestamp("2023-02-01 00:00:00")
    assert ts[1] == pd.Timestamp("2023-02-01 00:05:00")
    assert ts[2] == pd.Timestamp("2023-02-01 00:10:00")

    assert first_three["price"].to_list() == [11.0, 11.0, 13.0]
    assert first_three["amount"].to_list() == [3.0, None, 5.0]


def test_resample_chunk_boundary_aggregation_and_stale(tmp_path, monkeypatch):
    src = tmp_path / "binance_trades_2023-02-01_BTCUSDT.parquet"

    # 00:00 bucket has 3 rows and should be split across chunks when chunk_size=2.
    df = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "timestamp": [
                1675209605000000,  # 2023-02-01 00:00:05
                1675209720000000,  # 2023-02-01 00:02:00
                1675209899000000,  # 2023-02-01 00:04:59
                1675210205000000,  # 2023-02-01 00:10:05
            ],
            "price": [10.0, 11.0, 12.0, 20.0],
            "amount": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df.write_parquet(src)

    # Force tiny chunking to exercise cross-chunk merge logic.
    original_iter = dac._iter_parquet_chunks

    def tiny_chunks(path, chunk_size=500_000):
        return original_iter(path, chunk_size=2)

    monkeypatch.setattr(dac, "_iter_parquet_chunks", tiny_chunks)

    out = dac.resample(
        input_parquet_file=str(src),
        freq="5min",
        output_cols=["price", "amount"],
        summed_cols=["amount"],
    )

    out = out.sort("timestamp")
    first_three = out.head(3)

    # Buckets: 00:00, 00:05, 00:10
    assert first_three["timestamp"].to_list() == [
        pd.Timestamp("2023-02-01 00:00:00"),
        pd.Timestamp("2023-02-01 00:05:00"),
        pd.Timestamp("2023-02-01 00:10:00"),
    ]

    # 00:00 bucket: amount sum across chunk boundary and last price retained.
    # 00:05 bucket: no rows in bucket, so summed amount remains null.
    # 00:10 bucket: new row.
    assert first_three["amount"].to_list() == [6.0, None, 4.0]
    assert first_three["price"].to_list() == [12.0, 12.0, 20.0]


def test_resample_exact_bucket_boundary_keeps_summed_values_separate(tmp_path):
    src = tmp_path / "binance_trades_2023-02-01_BTCUSDT.parquet"
    df = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "timestamp": [
                1675209899999999,  # 2023-02-01 00:04:59.999999
                1675209900000000,  # 2023-02-01 00:05:00.000000
                1675209900500000,  # 2023-02-01 00:05:00.500000
            ],
            "price": [10.0, 11.0, 12.0],
            "amount": [1.0, 2.0, 3.0],
        }
    )
    df.write_parquet(src)

    out = dac.resample(
        input_parquet_file=str(src),
        freq="5min",
        output_cols=["price", "amount"],
        summed_cols=["amount"],
    ).sort("timestamp")

    first_two = out.head(2)
    assert first_two["timestamp"].to_list() == [
        pd.Timestamp("2023-02-01 00:00:00"),
        pd.Timestamp("2023-02-01 00:05:00"),
    ]

    # 00:00 bucket contains only the pre-boundary tick.
    # 00:05 bucket contains both ticks at and after the boundary.
    assert first_two["amount"].to_list() == [1.0, 5.0]
    assert first_two["price"].to_list() == [10.0, 12.0]


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
