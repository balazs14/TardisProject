from pathlib import Path

import polars as pl

from tardis import align_put_call_data as apc


def test_align_put_call_cli_parser_defaults_no_recreate_existing():
    parser = apc.build_cli_parser()

    args = parser.parse_args(
        [
            "--exchange", "okex",
            "--date", "2026-01-01",
        ]
    )

    assert args.recreate_existing is False


def test_align_put_call_cli_parser_accepts_recreate_existing():
    parser = apc.build_cli_parser()

    args = parser.parse_args(
        [
            "--exchange", "okex",
            "--date", "2026-01-01",
            "--recreate-existing",
        ]
    )

    assert args.recreate_existing is True


def test_run_cli_args_skips_when_output_exists_and_no_recreate(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "aligned.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("already here", encoding="utf-8")

    args = apc.build_cli_parser().parse_args(
        [
            "--exchange", "okex",
            "--date", "2026-01-01",
            "--output", str(output_path),
        ]
    )

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("alignment pipeline should not run when output exists and recreate is disabled")

    monkeypatch.setattr(apc, "create_okex_aligned_options", _should_not_run)

    apc.run_cli_args(args)

    captured = capsys.readouterr()
    assert f"skipped_existing={output_path}" in captured.out


def test_run_cli_args_recreates_when_output_exists_and_flag_enabled(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "aligned.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("already here", encoding="utf-8")

    args = apc.build_cli_parser().parse_args(
        [
            "--exchange", "okex",
            "--date", "2026-01-01",
            "--output", str(output_path),
            "--recreate-existing",
        ]
    )

    def _fake_create(*_args, **_kwargs):
        return pl.DataFrame({"x": [1]})

    monkeypatch.setattr(apc, "create_okex_aligned_options", _fake_create)

    apc.run_cli_args(args)

    captured = capsys.readouterr()
    assert f"wrote={output_path}" in captured.out
    assert Path(output_path).exists()
