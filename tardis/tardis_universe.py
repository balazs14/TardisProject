# === Importable entry points matching CLI subcommands ===
def universe_exchanges(exchange_regex=None):
    """Return a list of exchange dicts, optionally filtered by regex on id."""
    import re
    exs = list_exchanges(datasets_only=True)
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
        syms = filtered_symbols(
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
        result[exch] = data_types_for_exchange(exch)
    return result


def universe_columns(
    exchange=None,
    exchange_regex=None,
    symbol_regex=None,
    stream_type_regex=None,
    date=None,
    head=None,
):
    """Return a list of dicts for each matched stream file: exchange, symbol, stream_type, symbol_type, columns=[{column_name, polars_inferred_type}, ...]"""
    import re
    exs = universe_exchanges(exchange_regex) if exchange_regex else ([exchange] if exchange else [])
    if not exs:
        return []
    out = []
    for exch in exs if isinstance(exs[0], str) else [e["id"] for e in exs]:
        targets = matched_stream_targets(
            exchange=exch,
            symbol_regex=symbol_regex,
            stream_type_regex=stream_type_regex,
            date=date,
            head=head,
        )
        for t in targets:
            preview = stream_columns_head(
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
import sys
import json
import argparse
import gzip
import pathlib
import requests
from datetime import date as date_type
from urllib.parse import urlencode

# ---------------------------------------------------------------------------
# Column-preview disk cache
# ---------------------------------------------------------------------------

# Root for cached CSV previews; resolved relative to this file's package dir.
_CACHE_DIR = pathlib.Path(__file__).parent.parent / "datasets" / "columns_cache"
_CACHE_MAX_ROWS = 100


def _cache_path(exchange: str, stream_type: str, symbol: str) -> pathlib.Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return _CACHE_DIR / f"{exchange}-{stream_type}-{safe}-preview.csv"


def _read_cache(exchange: str, stream_type: str, symbol: str):
    """Return a polars.DataFrame from cache, or None if not present."""
    import polars as pl
    p = _cache_path(exchange, stream_type, symbol)
    if p.exists():
        return pl.read_csv(p, infer_schema_length=_CACHE_MAX_ROWS)
    return None


def _write_cache(exchange: str, stream_type: str, symbol: str, df) -> pathlib.Path:
    """Write up to _CACHE_MAX_ROWS rows to the cache CSV and return the path."""
    p = _cache_path(exchange, stream_type, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.head(_CACHE_MAX_ROWS).write_csv(p)
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
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


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
    details = exchange_details(exchange)
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
    symbols = dataset_symbols(
        exchange=exchange,
        symbol_type_regex=symbol_type_regex,
        symbol_regex=symbol_regex,
        date=date,
    )
    if stream_type_regex:
        symbols = filter_data_types(symbols, stream_type_regex)
    if head is not None:
        if head < 0:
            raise ValueError("head must be >= 0")
        symbols = symbols[:head]
    return symbols


def _matched_stream_targets(exchange: str, symbol_regex=None, stream_type_regex=None, date=None, head=None):
    """Return a list of stream targets with a sample_day computed from the date filter or latest available day."""
    import datetime
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    symbols = filtered_symbols(
        exchange=exchange,
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

    Checks the disk cache first (datasets/columns_cache/).  On a cache miss
    fetches up to _CACHE_MAX_ROWS rows from Tardis and writes them to cache
    so subsequent calls are free.
    """
    from .download_files import _tardis_csv_url
    import polars as pl

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
        # --- cache hit ---
        cached_df = _read_cache(exchange, stream_type, symbol)
        if cached_df is not None:
            full = cached_df
            out["from_cache"] = True
            out["cache_path"] = str(_cache_path(exchange, stream_type, symbol))
        else:
            # --- cache miss: fetch from HTTP ---
            api_key = os.environ.get("TARDIS_API_KEY", None)
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                response.raise_for_status()
                response.raw.decode_content = True
                with gzip.GzipFile(fileobj=response.raw) as gz:
                    full = pl.read_csv(gz, n_rows=_CACHE_MAX_ROWS, ignore_errors=True)

            if full.is_empty():
                out["error"] = "empty stream"
                return out

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
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _matched_stream_columns_head(
    exchange: str,
    symbol_regex=None,
    stream_type_regex=None,
    date=None,
    head=None,
    sample_rows: int = 2,
):
    """Return transposed column previews for matched downloadable stream files."""
    targets = matched_stream_targets(
        exchange=exchange,
        symbol_regex=symbol_regex,
        stream_type_regex=stream_type_regex,
        date=date,
        head=head,
    )
    previews = []
    for t in targets:
        preview = stream_columns_head(
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
    symbols = dataset_symbols(exchange)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        prog="tardis_universe",
        description="Explore the Tardis downloadable CSV universe",
        epilog=(
            "Examples:\n"
            "  python -m tardis.tardis_universe exchanges\n"
            "  python -m tardis.tardis_universe symbols okex-options\n"
            "  python -m tardis.tardis_universe symbols deribit --symbol-type-regex option --symbol-regex BTC\n"
            "  python -m tardis.tardis_universe symbols okex-options --date 2025-11-01\n"
            "  python -m tardis.tardis_universe symbols deribit --stream-type-regex 'quotes|trades'\n"
            "  python -m tardis.tardis_universe data-types deribit\n"
            "  python -m tardis.tardis_universe columns deribit --symbol-regex BTC --stream-type-regex quotes --head 3\n"
            "  python -m tardis.tardis_universe key-info\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser(
        "exchanges",
        help="List all dataset-capable exchanges",
        description=(
            "Fetch and print all exchanges that support downloadable CSV datasets.\n\n"
            "Example:\n"
            "  python -m tardis.tardis_universe exchanges"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sym_p = sub.add_parser(
        "symbols",
        help="List downloadable symbols for one or more exchanges",
        description=(
            "Print every symbol Tardis has CSV data for the given exchange(s),\n"
            "along with its instrument type, availability window and data-types.\n\n"
            "Examples:\n"
            "  python -m tardis.tardis_universe symbols --exchange okex-options\n"
            "  python -m tardis.tardis_universe symbols --exchange deribit --symbol-type-regex option\n"
            "  python -m tardis.tardis_universe symbols --exchange deribit --symbol-type-regex 'future|combo' --symbol-regex BTC\n"
            "  python -m tardis.tardis_universe symbols --exchange okex-options --date 2025-11-01\n"
            "  python -m tardis.tardis_universe symbols --exchange deribit --stream-type-regex 'quotes|trades'\n"
            "  python -m tardis.tardis_universe symbols --exchange deribit --head 20\n"
            "  python -m tardis.tardis_universe symbols --exchange deribit --symbol-regex '^ETH-USD-\\d{6}-C$'\n"
            "  python -m tardis.tardis_universe symbols --exchange deribit --symbol-regex BTC --json\n"
            "  python -m tardis.tardis_universe symbols --exchange-regex 'okex|deribit' --symbol-type-regex option\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = sym_p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--exchange",
        help="Exchange id, e.g. 'okex-options', 'deribit', 'binance'. Use the 'exchanges' sub-command to see all valid ids.",
    )
    group.add_argument(
        "--exchange-regex",
        help="Regex to match one or more exchanges by id.",
    )
    sym_p.add_argument(
        "--symbol-type-regex", dest="symbol_type_regex", default=None, metavar="REGEX",
        help=(
            "Regex matched against the instrument type field (case-insensitive). "
            "Common type values: option, future, perpetual, spot, combo. "
            "Examples: --symbol-type-regex option   --symbol-type-regex 'future|combo'"
        ),
    )
    sym_p.add_argument(
        "--symbol-regex", dest="symbol_regex", default=None, metavar="REGEX",
        help=(
            "Regex matched against the symbol id (case-insensitive). "
            "Examples: --symbol-regex BTC   --symbol-regex '^ETH-USD-\\d{6}-C$'"
        ),
    )
    sym_p.add_argument(
        "--stream-type-regex", dest="stream_type_regex", default=None, metavar="REGEX",
        help=(
            "Regex matched against each data-type (stream) name; only matching data-types are shown, "
            "and symbols with no matching data-type are dropped. "
            "Examples: --stream-type-regex quotes   --stream-type-regex 'quotes|trades'"
        ),
    )
    sym_p.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help=(
            "Only show symbols whose availability window includes this date. "
            "Example: --date 2025-11-01"
        ),
    )
    sym_p.add_argument(
        "--show-available-types", dest="show_available_types", action="store_true", default=True,
        help="Show the list of data-types available for each symbol (default: on).",
    )
    sym_p.add_argument(
        "--no-show-available-types", dest="show_available_types", action="store_false",
        help="Hide the data-types column for a more compact view.",
    )
    sym_p.add_argument(
        "--json", action="store_true",
        help="Dump the filtered symbol list as raw JSON (useful for scripting).",
    )
    sym_p.add_argument(
        "--head", type=int, default=None, metavar="N",
        help="Only show the first N results after filters are applied.",
    )

    dt_p = sub.add_parser(
        "data-types",
        help="Summarise data-types available per symbol type for an exchange",
        description=(
            "For each instrument type on the exchange (option, future, perpetual, …)\n"
            "print the union of all data-types any symbol of that type carries.\n\n"
            "Example:\n"
            "  python -m tardis.tardis_universe data-types deribit\n"
            "  python -m tardis.tardis_universe data-types okex-options"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group2 = dt_p.add_mutually_exclusive_group(required=True)
    group2.add_argument(
        "--exchange",
        help="Exchange id, e.g. 'deribit', 'okex-options'.",
    )
    group2.add_argument(
        "--exchange-regex",
        help="Regex to match one or more exchanges by id.",
    )

    col_p = sub.add_parser(
        "columns",
        help="Preview columns + 2 sample rows from matched downloadable stream files",
        description=(
            "Match symbol/data stream combinations using regex filters, then fetch the\n"
            "first chunk of each matching CSV stream and print a transposed schema preview.\n\n"
            "Examples:\n"
            "  python -m tardis.tardis_universe columns deribit --symbol-regex BTC --stream-type-regex quotes --head 3\n"
            "  python -m tardis.tardis_universe columns okex-options --stream-type-regex derivative_ticker --head 2\n"
            "  python -m tardis.tardis_universe columns deribit --symbol-regex '^ETH-USD-\\\\d{6}-C$' --stream-type-regex trades --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group3 = col_p.add_mutually_exclusive_group(required=True)
    group3.add_argument(
        "--exchange",
        help="Exchange id, e.g. 'deribit', 'okex-options'.",
    )
    group3.add_argument(
        "--exchange-regex",
        help="Regex to match one or more exchanges by id.",
    )
    col_p.add_argument(
        "--symbol-regex", dest="symbol_regex", default=None, metavar="REGEX",
        help="Regex matched against the symbol id.",
    )
    col_p.add_argument(
        "--stream-type-regex", dest="stream_type_regex", default=None, metavar="REGEX",
        help="Regex matched against stream/data-type names (e.g. quotes, trades).",
    )
    col_p.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help="Only consider symbols whose availability window includes this date.",
    )
    col_p.add_argument(
        "--head", type=int, default=None, metavar="N",
        help="Only inspect the first N matched stream files (default: None ).",
    )
    col_p.add_argument(
        "--json", action="store_true",
        help="Dump column previews as JSON instead of pretty printed table.",
    )

    sub.add_parser(
        "key-info",
        help="Show what your API key covers (exchanges, date ranges, data plan)",
        description=(
            "Calls /v1/api-key-info and prints which exchanges and date ranges\n"
            "your TARDIS_API_KEY is authorised to download.\n\n"
            "Example:\n"
            "  python -m tardis.tardis_universe key-info"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "exchanges":
        exs = list_exchanges(datasets_only=True)
        print_exchanges(exs)

    if args.cmd == "symbols":
        import re
        all_exs = [e["id"] for e in list_exchanges(datasets_only=True)]
        if args.exchange:
            exchanges = [args.exchange]
        else:
            pat = re.compile(args.exchange_regex, re.IGNORECASE)
            exchanges = [e for e in all_exs if pat.search(e)]
        for exch in exchanges:
            total = len(
                filtered_symbols(
                    exchange=exch,
                    symbol_type_regex=args.symbol_type_regex,
                    symbol_regex=args.symbol_regex,
                    stream_type_regex=args.stream_type_regex,
                    date=args.date,
                )
            )
            syms = filtered_symbols(
                exchange=exch,
                symbol_type_regex=args.symbol_type_regex,
                symbol_regex=args.symbol_regex,
                stream_type_regex=args.stream_type_regex,
                date=args.date,
                head=args.head,
            )
            if args.json:
                print(json.dumps(syms, indent=2))
            else:
                print(f"\n{len(syms)} symbols for {exch}"
                      + (f"  (symbol-type-regex={args.symbol_type_regex})" if args.symbol_type_regex else "")
                      + (f"  (symbol-regex={args.symbol_regex})" if args.symbol_regex else "")
                      + (f"  (stream-type-regex={args.stream_type_regex})" if args.stream_type_regex else "")
                      + (f"  (head={args.head})" if args.head is not None else "")
                      + (f"  (date={args.date})" if args.date else ""))
                print_symbols(syms, show_data_types=args.show_available_types)
                if args.head is not None and total > len(syms):
                    print(f"\nShowing first {len(syms)} of {total} matching symbols.")

    elif args.cmd == "columns":
        import re
        all_exs = [e["id"] for e in list_exchanges(datasets_only=True)]
        if args.exchange:
            exchanges = [args.exchange]
        else:
            pat = re.compile(args.exchange_regex, re.IGNORECASE)
            exchanges = [e for e in all_exs if pat.search(e)]
        for exch in exchanges:
            try:
                previews = matched_stream_columns_head(
                    exchange=exch,
                    symbol_regex=args.symbol_regex,
                    stream_type_regex=args.stream_type_regex,
                    date=args.date,
                    head=args.head,
                    sample_rows=2,
                )
            except ValueError as exc:
                parser.error(str(exc))

            if args.json:
                print(json.dumps(previews, indent=2))
            else:
                print(
                    f"\n{len(previews)} matched stream files for {exch}"
                    + (f"  (symbol-regex={args.symbol_regex})" if args.symbol_regex else "")
                    + (f"  (stream-type-regex={args.stream_type_regex})" if args.stream_type_regex else "")
                    + (f"  (date={args.date})" if args.date else "")
                    + (f"  (head={args.head})" if args.head is not None else "")
                )
                print_columns_transposed(previews)

    elif args.cmd == "data-types":
        import re
        all_exs = [e["id"] for e in list_exchanges(datasets_only=True)]
        if args.exchange:
            exchanges = [args.exchange]
        else:
            pat = re.compile(args.exchange_regex, re.IGNORECASE)
            exchanges = [e for e in all_exs if pat.search(e)]
        for exch in exchanges:
            coverage = data_types_for_exchange(exch)
            print(f"\nData-types available on {exch}:")
            for t, dts in sorted(coverage.items()):
                print(f"  {t:<20} {dts}")

    elif args.cmd == "key-info":
        info = key_info()
        print("\nAPI key coverage:")
        print_key_info(info)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
