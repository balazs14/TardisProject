from pathlib import Path

import pandas as pd
import polars as pl

from tardis import day_sample_options as dso


def test_sample_day_options_uses_post_resample_freq_before_alignment(tmp_path, monkeypatch):
    exchange = "testex"
    day = "2023-02-01"
    symbol = "TESTSYM"

    quote_path = tmp_path / f"{exchange}_quotes_{day}_{symbol}_5min.parquet"
    pl.DataFrame({"symbol": [symbol], "timestamp": [1]}).write_parquet(quote_path)

    monkeypatch.setattr(
        dso,
        "EXCHANGE_DOWNLOADS",
        {
            exchange: {
                "static": [(exchange, "quotes", symbol)],
            }
        },
    )

    calls: dict[str, object] = {}

    def fake_download_resample(**kwargs):
        calls["download_resample_freq"] = kwargs["resample_freq"]
        return [quote_path]

    def fake_normalize_quotes(_exchange, _df, _timestamp_unit):
        return pl.DataFrame(
            {
                "ts": [pd.Timestamp("2023-02-01 00:00:00")],
                "symbol": [symbol],
                "pc": ["C"],
            }
        )

    def fake_resample_normalized_quotes(df, freq, _timestamp_unit):
        calls["post_resample_freq"] = freq
        calls["normalized_rows"] = df.height
        return pl.DataFrame(
            {
                "ts": [pd.Timestamp("2023-02-01 00:00:00")],
                "symbol": [symbol],
                "pc": ["C"],
            }
        )

    def fake_align_calls_puts_with_legs(df, trade_df):
        calls["sampled_rows"] = df.height
        calls["trade_rows"] = trade_df.height
        return pl.DataFrame(
            {
                "ts": [pd.Timestamp("2023-02-01 00:00:00")],
                "symbol": [symbol],
                "pc": ["C"],
                "value": [1.0],
            }
        )

    monkeypatch.setattr(dso, "download_resample", fake_download_resample)
    monkeypatch.setattr(dso, "_normalize_quotes", fake_normalize_quotes)
    monkeypatch.setattr(dso, "_resample_normalized_quotes", fake_resample_normalized_quotes)
    monkeypatch.setattr(dso, "_align_calls_puts_with_legs", fake_align_calls_puts_with_legs)

    config = dso.SampleConfig(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path),
        freq="5min",
        post_resample_freq="15min",
        force_reload=True,
        cleanup_intermediate_parquet=False,
    )

    out = dso.day_sample(day=day, exchange=exchange, config=config)

    assert calls["download_resample_freq"] == "5min"
    assert calls["post_resample_freq"] == "15min"
    assert calls["normalized_rows"] == 1
    assert calls["sampled_rows"] == 1
    assert calls["trade_rows"] == 0

    output_path = Path(config.output_dir) / f"{exchange}_aligned_options_{day}_15min.parquet"
    assert output_path.exists()
    assert out.height == 1
