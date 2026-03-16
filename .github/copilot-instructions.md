# Copilot instructions for TardisProject

## Big picture

- This repo mixes package code (`TardisProject/`) with script-style pipelines at repo root (notably `okex.py`, `deribit.py`, `deribit_polars.py`).
- Core reusable ingestion/resampling utilities live in `tardis/lib/*.py`.
- Strategy analytics (put-call parity break calculations) are implemented in both pandas and polars paths in `okex.py` and 'deribit_polars.py' Expecting to add files and refactor these into the lib.
- Data is file-based (CSV `.csv.gz` -> parquet), not service-based; most workflows are batch/day oriented.

## Primary data flow

1. Download from Tardis via `tardis_dev.datasets.download(...)`.
2. Convert CSV to typed parquet (pyarrow streaming) with microsecond timestamps (`TIMESTAMP_UNIT = 'us'`).
3. Resample/forward-fill per symbol on a time grid.
4. Join option/future/spot legs and compute PCP metrics.

Reference implementations:

- Generic downloader/converter: `tardis/lib/download_and_convert.py`
- OKX enrichment + PCP metrics: `okex.py` (`augment_quotes`, `merge_data`, `pcp_breaking_*`)
- Snapshot tests for PCP outputs: `tests/test_pcp_breaking.py`

## Developer workflows

- Use Python 3.12 (`pyproject.toml` requires `>=3.12`).
- Main commands:
  - `pytest -q` (also VS Code task: **Run tests (pytest)**)
  - `black .` (also VS Code task: **Format with black**)
- VS Code is configured to use `venv/bin/python` in `.vscode/settings.json`.

## Project-specific coding patterns

- Preserve dual interfaces where present: functions may accept pandas input and internally switch to polars, then optionally convert back (see `align_calls_puts`, `pcp_breaking_polars`).
- Keep timestamp handling explicit and in microseconds; many tests depend on exact bucket/timestamp behavior.
- In resampling code, aggregation is column-sensitive:
  - explicit map in `RESAMPLE_AGGREGATION_BY_COLUMN`
  - fallback: columns containing `amount` are summed, others use last value.
- Existing tests often validate full dataframe string snapshots via `tardis/lib/test_utils.py`; avoid reformatting output columns/order without updating tests.

## Testing and safe changes

- Prefer targeted tests first:
  - `tests/test_pcp_breaking.py`
  - `tardis/tests/test_download_and_convert.py`
- For data-download logic, tests monkeypatch `datasets.download`; keep new tests offline/deterministic.
- Scripts in `attic/` are exploratory/archive; avoid treating them as canonical unless explicitly requested.

## External integration points

- Tardis API integration: `tardis_dev.datasets` (and `tardis_client` in some scripts).
- Storage format stack: `pyarrow` + `polars` + `pandas` interop.
- Environment variable used by download scripts: `TARDIS_API_KEY`.

## When editing

- Make minimal, local changes; this repo has many ad-hoc scripts with overlapping logic.
- If you change shared pipeline behavior, run both PCP and download/convert test suites.
- Prefer updating existing functions over introducing new abstractions unless repetition is blocking correctness.
