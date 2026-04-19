# === Importable entry points matching CLI subcommands ===
def universe_exchanges(exchange_regex=None):
    """Return a list of exchange dicts, optionally filtered by regex on id."""
    import re
    exs = _list_exchanges(datasets_only=True)
    if exchange_regex:
        pat = re.compile(exchange_regex, re.IGNORECASE)
        exs = [e for e in exs if pat.search(e["id"])]
    return exs


def universe_symbols(
    exchange=None,
    exchange_regex=None,
    symbol_type_regex=None,
    symbol_regex=None,
    stream_type_regex=None,
    date=None,
    head=None,
):
    """Return all symbols for matching exchanges, with all CLI-style filters."""
    import re
    exs = universe_exchanges(exchange_regex) if exchange_regex else ([exchange] if exchange else [])
    if not exs:
        return []
    out = []
    for exch in exs if isinstance(exs[0], str) else [e["id"] for e in exs]:
        syms = _filtered_symbols(
            exchange=exch,
            symbol_type_regex=symbol_type_regex,
            symbol_regex=symbol_regex,
            stream_type_regex=stream_type_regex,
            date=date,
            head=head,
        )
        for s in syms:
            s2 = dict(s)
            s2["exchange"] = exch
            out.append(s2)
    return out


def universe_data_types(exchange=None, exchange_regex=None):
    """Return a dict mapping exchange id to {symbol_type: [data_types]} for all matches."""
    import re
    exs = universe_exchanges(exchange_regex) if exchange_regex else ([exchange] if exchange else [])
    if not exs:
        return {}
    result = {}
    for exch in exs if isinstance(exs[0], str) else [e["id"] for e in exs]:
        result[exch] = _data_types_for_exchange(exch)
    return result


def universe_columns(
    exchange=None,
    exchange_regex=None,
    symbol_type_regex=None,
    symbol_regex=None,
    stream_type_regex=None,
    date=None,
    head=None,
):
    """Return column previews for matched stream files.

    Supports the same filters as the CLI columns command, including
    symbol_type_regex to keep only symbols whose type matches the regex.
    Output items include: exchange, symbol, stream_type, symbol_type,
    columns=[{column_name, polars_inferred_type}, ...].
    """
    import re
    exs = universe_exchanges(exchange_regex) if exchange_regex else ([exchange] if exchange else [])
    if not exs:
        return []
    out = []
    for exch in exs if isinstance(exs[0], str) else [e["id"] for e in exs]:
        targets = _matched_stream_targets(
            exchange=exch,
            symbol_type_regex=symbol_type_regex,
            symbol_regex=symbol_regex,
            stream_type_regex=stream_type_regex,
            date=date,
            head=head,
        )
        for t in targets:
            preview = _stream_columns_head(
                exchange=t["exchange"],
                symbol=t["symbol"],
                stream_type=t["stream_type"],
                sample_day=t["sample_day"],
                sample_rows=0,
            )
            if preview.get("error"):
                continue
            cols = [
                {"column_name": r["column_name"], "polars_inferred_type": r["polars_inferred_type"]}
                for r in preview.get("rows", [])
            ]
            out.append({
                "exchange": exch,
                "symbol": t["symbol"],
                "stream_type": t["stream_type"],
                "symbol_type": t["symbol_type"],
                "columns": cols,
            })
    return out
"""
tardis_universe.py
------------------
Helpers for exploring what data is downloadable from Tardis.

Key API endpoints used:
  GET https://api.tardis.dev/v1/exchanges
      -> list of all exchanges with supportsDatasets flag

  GET https://api.tardis.dev/v1/exchanges/:exchange
      -> per-exchange detail: datasets.symbols[] with dataTypes, availableSince, availableTo

  GET https://api.tardis.dev/v1/api-key-info
      -> what your API key can actually access (exchanges, date ranges)

Usage examples
--------------
  # Show all dataset-capable exchanges
  python tardis_universe.py exchanges

  # Show downloadable symbols + data-types for one exchange
  python tardis_universe.py symbols okex-options

  # Show what your API key covers
  python tardis_universe.py key-info

  # Filter symbols by type, e.g. only options
  python tardis_universe.py symbols deribit --type option

  # Filter symbols with a substring match on id
  python tardis_universe.py symbols okex-options --match BTC

  # Show symbols available on a specific date
  python tardis_universe.py symbols okex-options --date 2025-11-01
"""

import os
import gzip
import pathlib
import logging
import requests
from datetime import date as date_type
from urllib.parse import urlencode

from tardis.api_cache import cached_get

# ---------------------------------------------------------------------------
# Column-preview disk cache
# ---------------------------------------------------------------------------

# Root for cached CSV previews; resolved relative to this file's package dir.
_CACHE_DIR = pathlib.Path(__file__).parent.parent / "datasets" / "columns_cache"
_CACHE_MAX_ROWS = 100
logger = logging.getLogger(__name__)
_LOG_LEVEL = os.environ.get("TARDIS_UNIVERSE_LOGLEVEL")
if _LOG_LEVEL:
    logging.basicConfig(level=getattr(logging, _LOG_LEVEL.upper(), logging.INFO))


def _cache_path(exchange: str, stream_type: str, symbol: str) -> pathlib.Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return _CACHE_DIR / f"{exchange}-{stream_type}-{safe}-preview.csv"


def _empty_cache_marker_path(exchange: str, stream_type: str, symbol: str) -> pathlib.Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return _CACHE_DIR / f"{exchange}-{stream_type}-{safe}-preview.empty"


def _read_cache(exchange: str, stream_type: str, symbol: str):
    """Return a polars.DataFrame from cache, or None if not present."""
    import polars as pl
    empty_marker = _empty_cache_marker_path(exchange, stream_type, symbol)
    if empty_marker.exists():
        logger.debug("Cache hit (empty marker) for preview file: %s", empty_marker)
        return pl.DataFrame()

    p = _cache_path(exchange, stream_type, symbol)
    if p.exists():
        logger.debug("Cache hit for preview file: %s", p)
        return pl.read_csv(p, infer_schema_length=_CACHE_MAX_ROWS)
    logger.debug("Cache miss for preview file: %s", p)
    return None


def _write_cache(exchange: str, stream_type: str, symbol: str, df) -> pathlib.Path:
    """Write up to _CACHE_MAX_ROWS rows to the cache CSV and return the path."""
    p = _cache_path(exchange, stream_type, symbol)
    empty_marker = _empty_cache_marker_path(exchange, stream_type, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    if empty_marker.exists():
        empty_marker.unlink()
    df.head(_CACHE_MAX_ROWS).write_csv(p)
    logger.debug("Wrote preview cache file: %s", p)
    return p


def _write_empty_cache_marker(exchange: str, stream_type: str, symbol: str) -> pathlib.Path:
    p = _empty_cache_marker_path(exchange, stream_type, symbol)
    csv_path = _cache_path(exchange, stream_type, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()
    p.write_text("empty\n", encoding="utf-8")
    logger.debug("Wrote empty preview cache marker: %s", p)
    return p

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE = "https://api.tardis.dev/v1"
API_KEY = os.environ.get("TARDIS_API_KEY")


def _headers():
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    return {}


def _get(url, params=None):
    logger.debug("About to hit endpoint: %s params=%s", url, params)
    payload = cached_get(url, params=params, headers=_headers(), timeout=30)
    logger.debug("Returned from endpoint (maybe cached): %s", url)
    return payload


# ---------------------------------------------------------------------------
# Core fetch functions
# ---------------------------------------------------------------------------

def _list_exchanges(datasets_only=True):
    """Return list of exchange dicts from /v1/exchanges.

    Each entry has: id, name, enabled, supportsDatasets, availableSince, [availableTo]
    """
    data = _get(f"{BASE}/exchanges")
    if datasets_only:
        data = [e for e in data if e.get("supportsDatasets")]
    return data


def _exchange_details(exchange: str):
    """Return full detail dict for one exchange, including datasets.symbols."""
    return _get(f"{BASE}/exchanges/{exchange}")


def _dataset_symbols(exchange: str, symbol_type_regex=None, symbol_regex=None, date=None):
    """Return the datasets.symbols list for an exchange.

    Parameters
    ----------
    exchange          : exchange id, e.g. 'okex-options', 'deribit'
    symbol_type_regex : regex matched against the instrument type field, e.g. 'opt' or 'future|combo'
    symbol_regex      : regex matched against the symbol id, e.g. 'BTC' or '^ETH-USD-\\d{6}-C$'
    date              : ISO date string 'YYYY-MM-DD'; keep only symbols available on that date
    """
    import re
    details = _exchange_details(exchange)
    symbols = details.get("datasets", {}).get("symbols", [])

    if symbol_type_regex:
        pat = re.compile(symbol_type_regex, re.IGNORECASE)
        symbols = [s for s in symbols if pat.search(s.get("type", ""))]

    if symbol_regex:
        pat = re.compile(symbol_regex, re.IGNORECASE)
        symbols = [s for s in symbols if pat.search(s["id"])]

    if date:
        from datetime import date as date_type
        d = date_type.fromisoformat(date)
        def _available(s):
            since = date_type.fromisoformat(s["availableSince"][:10])
            to_raw = s.get("availableTo")
            to = date_type.fromisoformat(to_raw[:10]) if to_raw else date_type(9999, 12, 31)
            return since <= d <= to
        symbols = [s for s in symbols if _available(s)]

    return symbols


def _filter_data_types(symbols, stream_type_regex):
    """Filter each symbol's dataTypes list by a regex; symbols with no match are dropped."""
    import re
    pat = re.compile(stream_type_regex, re.IGNORECASE)
    result = []
    for s in symbols:
        matched = [dt for dt in s.get("dataTypes", []) if pat.search(dt)]
        if matched:
            result.append({**s, "dataTypes": matched})
    return result


def _filtered_symbols(
    exchange: str,
    symbol_type_regex=None,
    symbol_regex=None,
    stream_type_regex=None,
    date=None,
    head=None,
):
    """Return symbols filtered by type/symbol/date and optional stream type + head."""
    symbols = _dataset_symbols(
        exchange=exchange,
        symbol_type_regex=symbol_type_regex,
        symbol_regex=symbol_regex,
        date=date,
    )
    if stream_type_regex:
        symbols = _filter_data_types(symbols, stream_type_regex)
    if head is not None:
        if head < 0:
            raise ValueError("head must be >= 0")
        symbols = symbols[:head]
    return symbols


def _matched_stream_targets(
    exchange: str,
    symbol_type_regex=None,
    symbol_regex=None,
    stream_type_regex=None,
    date=None,
    head=None,
):
    """Return a list of stream targets with a sample_day computed from the date filter or latest available day."""
    import datetime
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    symbols = _filtered_symbols(
        exchange=exchange,
        symbol_type_regex=symbol_type_regex,
        symbol_regex=symbol_regex,
        stream_type_regex=stream_type_regex,
        date=date,
    )
    targets = []
    for s in symbols:
        # Pick the sample day: if caller specified a date use it, else use the
        # symbol's latest available day (availableTo strips one day so use the
        # raw value), or fall back to today.
        if date:
            sample_day = date
        else:
            avail_to = s.get("availableTo", "")
            raw = avail_to[:10] if avail_to else yesterday
            # clamp to yesterday — today's data is not yet published
            sample_day = min(raw, yesterday)
        for stream_type in s.get("dataTypes", []):
            targets.append(
                {
                    "exchange": exchange,
                    "symbol": s["id"],
                    "symbol_type": s.get("type", ""),
                    "stream_type": stream_type,
                    "sample_day": sample_day,
                }
            )

    if head is not None:
        if head < 0:
            raise ValueError("head must be >= 0")
        targets = targets[:head]
    return targets


def _safe_cell(v):
    if v is None:
        return ""
    text = str(v)
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _stream_columns_head(
    exchange: str,
    symbol: str,
    stream_type: str,
    sample_day: str,
    sample_rows: int = 2,
):
    """Return a transposed column preview for one downloadable stream file.

    Checks the disk cache first (datasets/columns_cache/).  On a cache miss,
    uses PyArrow streaming CSV reader to fetch only the first batch from Tardis
    and caches it, so subsequent calls are free. This avoids loading the entire
    file into memory.
    """
    from .download_files import _tardis_csv_url, _temporal_fallback_column_types, _cast_temporal_fallback_columns, TARDIS_COLUMN_TYPES
    import polars as pl
    import pyarrow.csv as pv
    import pyarrow as pa

    sample_rows = max(0, sample_rows)
    day = date_type.fromisoformat((sample_day or "1970-01-01")[:10])
    url = _tardis_csv_url(exchange=exchange, data_type=stream_type, day=day, symbol=symbol)

    out = {
        "exchange": exchange,
        "symbol": symbol,
        "stream_type": stream_type,
        "symbol_type": None,
        "url": url,
        "sample_day": day.isoformat(),
        "from_cache": False,
        "cache_path": None,
        "rows": [],
        "error": None,
    }

    try:
        full = None
        # --- cache hit ---
        cached_df = _read_cache(exchange, stream_type, symbol)
        if cached_df is not None:
            full = cached_df
            out["from_cache"] = True
            out["cache_path"] = str(_cache_path(exchange, stream_type, symbol))
        else:
            # --- cache miss: streaming fetch from HTTP ---
            # Use PyArrow streaming reader to read only the first batch without
            # loading the entire file into memory.
            api_key = os.environ.get("TARDIS_API_KEY", None)
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            
            # Setup PyArrow CSV reader options (same as in download_files)
            read_options = pv.ReadOptions(block_size=8 * 1024 * 1024)
            convert_options = pv.ConvertOptions(
                column_types=_temporal_fallback_column_types(TARDIS_COLUMN_TYPES),
                null_values=["", "null", "None", "NULL", "NaN", "nan"],
            )
            
            logger.debug("About to hit streaming endpoint: %s", url)
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                logger.debug("Attached to streaming endpoint: %s status=%s", url, response.status_code)
                try:
                    response.raise_for_status()
                    response.raw.decode_content = True
                    with gzip.GzipFile(fileobj=response.raw) as gz:
                        reader = pv.open_csv(gz, read_options=read_options, convert_options=convert_options)
                        try:
                            # Read only the first streamed batch using the same
                            # pattern as tardis/download_files.py.
                            first_batch = None
                            for batch in reader:
                                first_batch = batch
                                break

                            if first_batch is None:
                                logger.debug("Streaming endpoint produced no CSV rows: %s", url)
                                full = pl.DataFrame()
                            else:
                                arrow_table = pa.Table.from_batches([first_batch])
                                full = pl.from_arrow(arrow_table)
                                full = _cast_temporal_fallback_columns(full, TARDIS_COLUMN_TYPES)
                                logger.debug(
                                    "Materialized streamed preview chunk for %s/%s/%s with rows=%s cols=%s",
                                    exchange,
                                    stream_type,
                                    symbol,
                                    full.height,
                                    len(full.columns),
                                )
                        finally:
                            reader.close()
                finally:
                    logger.debug("Detached from streaming endpoint: %s", url)

            if full is None:
                logger.debug("No batch returned from streaming endpoint; treating as empty stream: %s", url)
                full = pl.DataFrame()

            if full.is_empty():
                out["error"] = "empty stream"
                out["cache_path"] = str(_write_empty_cache_marker(exchange, stream_type, symbol))
                return out

            # Keep only the first min(streamed chunk rows, _CACHE_MAX_ROWS) rows
            # for both current previewing and disk cache.
            full = full.head(_CACHE_MAX_ROWS)
            logger.debug(
                "Persisting preview cache for %s/%s/%s with rows=%s (limit=%s)",
                exchange,
                stream_type,
                symbol,
                full.height,
                _CACHE_MAX_ROWS,
            )
            out["cache_path"] = str(_write_cache(exchange, stream_type, symbol, full))

        sample = full.head(max(1, sample_rows))
        if sample.is_empty():
            out["error"] = "empty (cache)"
            return out

        dict_rows = sample.to_dicts()
        schema = sample.schema
        transposed_rows = []
        for col in sample.columns:
            vals = [r.get(col) for r in dict_rows]
            transposed_rows.append({
                "column_name": col,
                "polars_inferred_type": str(schema[col]),
                "row_1": _safe_cell(vals[0]) if len(vals) > 0 else "",
                "row_2": _safe_cell(vals[1]) if len(vals) > 1 else "",
            })
        out["rows"] = transposed_rows
        return out
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        out["error"] = f"http error: {status}"
        return out
    except pa.ArrowInvalid as exc:
        # Empty CSV payloads can surface as Arrow parse errors before any batch exists.
        msg = str(exc)
        logger.debug("Arrow parse error while streaming %s: %s", url, msg)
        if "empty" in msg.lower():
            out["error"] = "empty stream"
            out["cache_path"] = str(_write_empty_cache_marker(exchange, stream_type, symbol))
            return out
        out["error"] = msg
        return out
    except Exception as exc:
        logger.exception("Unexpected error while building preview for %s/%s/%s", exchange, stream_type, symbol)
        out["error"] = str(exc)
        return out


def _matched_stream_columns_head(
    exchange: str,
    symbol_type_regex=None,
    symbol_regex=None,
    stream_type_regex=None,
    date=None,
    head=None,
    sample_rows: int = 2,
):
    """Return transposed column previews for matched downloadable stream files."""
    targets = _matched_stream_targets(
        exchange=exchange,
        symbol_type_regex=symbol_type_regex,
        symbol_regex=symbol_regex,
        stream_type_regex=stream_type_regex,
        date=date,
        head=head,
    )
    previews = []
    for t in targets:
        preview = _stream_columns_head(
            exchange=t["exchange"],
            symbol=t["symbol"],
            stream_type=t["stream_type"],
            sample_day=t["sample_day"],
            sample_rows=sample_rows,
        )
        preview["symbol_type"] = t["symbol_type"]
        previews.append(preview)
    return previews


def _key_info():
    """Return API key coverage info from /v1/api-key-info."""
    return _get(f"{BASE}/api-key-info")


def _data_types_for_exchange(exchange: str):
    """Summarise which data-types are available per symbol type for an exchange."""
    symbols = _dataset_symbols(exchange)
    coverage = {}   # type -> set of dataTypes
    for s in symbols:
        t = s.get("type", "unknown")
        for dt in s.get("dataTypes", []):
            coverage.setdefault(t, set()).add(dt)
    return {t: sorted(v) for t, v in coverage.items()}


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_exchanges(exchanges):
    fmt = "{:<30} {:<12} {:<26} {}"
    print(fmt.format("ID", "ENABLED", "AVAILABLE SINCE", "DATASETS"))
    print("-" * 80)
    for e in exchanges:
        avail = e.get("availableSince", "")[:10]
        avail_to = e.get("availableTo", "")
        if avail_to:
            avail += f"  ..  {avail_to[:10]}"
        print(fmt.format(e["id"], str(e.get("enabled", "")), avail, "csv" if e.get("supportsDatasets") else ""))


def _print_symbols(symbols, show_data_types=True):
    if show_data_types:
        fmt = "  {:<45} {:<15} {:<12} {:<12} {}"
        print(fmt.format("SYMBOL", "SYMBOL_TYPE", "AVAILABLE_FROM", "AVAILABLE_TO", "STREAM_TYPES"))
    else:
        fmt = "  {:<45} {:<15} {:<12} {:<12}"
        print(fmt.format("SYMBOL", "SYMBOL_TYPE", "AVAILABLE_FROM", "AVAILABLE_TO"))
    print("  " + "-" * 120)

    for s in symbols:
        dts = "  [" + ", ".join(s.get("dataTypes", [])) + "]" if show_data_types else ""
        avail_to = s.get("availableTo", "")[:10] or "now"
        print(f"  {s['id']:<45} {s.get('type',''):<15} {s['availableSince'][:10]}  ..  {avail_to}{dts}")


def _print_columns_transposed(previews):
    fmt = "  {:<36} {:<22} {:<30} {}"
    for i, p in enumerate(previews, start=1):
        cached_note = "  [cached]" if p.get("from_cache") else "  [fetched]"
        print(
            f"\n[{i}] {p['exchange']}  symbol={p['symbol']}  "
            f"symbol-type={p.get('symbol_type','')}  stream-type={p['stream_type']}  "
            f"day={p['sample_day']}{cached_note}"
        )
        if p.get("error"):
            print(f"  ERROR: {p['error']}")
            print(f"  URL: {p['url']}")
            continue

        print(fmt.format("COLUMN_NAME", "POLARS_INFERRED_TYPE", "ROW_1", "ROW_2"))
        print("  " + "-" * 120)
        for r in p.get("rows", []):
            print(
                fmt.format(
                    r.get("column_name", ""),
                    r.get("polars_inferred_type", ""),
                    r.get("row_1", ""),
                    r.get("row_2", ""),
                )
            )


def _print_key_info(info):
    for entry in info:
        to_str = entry.get("to", "")[:10]
        syms = entry.get("symbols", [])
        sym_note = f"  symbols: {syms}" if syms else "  all symbols"
        print(f"  {entry['exchange']:<30} {entry.get('accessType',''):<20} {entry.get('from','')[:10]}  ..  {to_str}  plan={entry.get('dataPlan','')}{sym_note}")


# NOTE:
# CLI parser/dispatch was moved to tardis.entry_point so this module remains
# a pure library of universe/data-inspection helpers.
