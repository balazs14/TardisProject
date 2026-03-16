# TardisProject Context

## What this project is

TardisProject is a Python 3.12 codebase for crypto market data workflows, focused on downloading historical exchange datasets (via Tardis), converting CSVs to parquet efficiently, and running downstream analytics such as put-call-parity break calculations.

## Primary goals

- Download exchange data reliably by date range and symbol.
- Convert large CSV inputs to typed parquet files with stable schema handling.
- Resample time-series data for analysis-ready intervals.
- Compute strategy/arb diagnostics (e.g., PCP breaking metrics) with both pandas and polars paths.

## Main code locations

- `tardis/lib/download_and_convert.py`
  - Core download + conversion + resample utilities.
  - Defines rich column type mapping (`TARDIS_COLUMN_TYPES`) and aggregation policies.
- `okex.py`
  - Contains PCP breaking logic used by tests (`pcp_breaking_pandas`, `pcp_breaking_polars`).
- `tests/test_pcp_breaking.py`
  - Snapshot-style validation of PCP metric outputs.
- `tardis/tests/test_download_and_convert.py`
  - Behavioral tests for download/convert/resample paths.

## Data and output directories

- Raw and generated datasets appear in folders like:
  - `datasets/`
  - `deribit_data/`
  - `okex/`
- Scripts and notebooks in repo root and `attic/` support ad-hoc analysis.

## Tooling and runtime

- Python requirement: `>=3.12` (`pyproject.toml`).
- Key libraries: `pandas`, `polars`, `pyarrow`, `pytest`, `tardis-dev`, `tardis-client`, `duckdb`.
- Test task: `Run tests (pytest)` → `pytest -q`.
- Formatting task: `Format with black` → `black .`.

## Current testing shape

- Unit/integration-style tests are present for:
  - Download date handling and force reload behavior.
  - CSV→parquet conversion expectations.
  - Resampling semantics (bucket boundaries, stale fills, sums vs last-value columns).
  - PCP computation snapshots against expected numeric outputs.

## Conventions and implementation notes

- Timestamps are normalized with microsecond resolution (`TIMESTAMP_UNIT = 'us'`).
- Conversion logic includes fallback parsing when timestamp fields arrive as integers.
- Aggregation defaults:
  - Explicit per-column rules in `RESAMPLE_AGGREGATION_BY_COLUMN`.
  - Fallback rule: columns containing `amount` are summed; others take last value.

## Known project shape

- Mixed style workspace:
  - Package code under `TardisProject/`
  - Standalone scripts at root (e.g., `deribit.py`, `okex.py`).
  - Historical/experimental scripts under `attic/`.
- This suggests production and exploratory workflows coexist in one repository.

## Suggested near-term TODOs

- Add a top-level `README.md` that links to this context.
- Define one canonical data directory policy (raw vs processed).
- Add CLI entry points for the most common download/convert/resample workflow.
- Expand test coverage around edge schemas and missing columns.

## Quick start commands

```bash
# Run tests
pytest -q

# Auto-format code
black .
```

## Notes to keep updated

When project behavior changes, update this document first in these sections:

1. Main code locations
2. Tooling and runtime
3. Current testing shape
4. Suggested near-term TODOs
