import logging
from pathlib import Path

import polars as pl

from tardis.download_files import download_resample


logger = logging.getLogger(__name__)


def _default_symbol_markup() -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Retired: issue warning and pass dataframe through unchanged."""
    logger.warning("Using default (retired) symbol markup; no columns will be added")
    return {}, ()


# ============================================================================
# OKEX CONVERTERS (options, futures, spot)
# ============================================================================

def _okex_options_quotes_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert OKEX options quote/trade symbols to markup expressions.
    
    Pattern: CUR1-CUR2-YYMMDD-STRIKE-PC
    Example: ETH-USD-260529-1600-P
    """
    assert df.select(pl.col("symbol").cast(pl.String).str.contains(r"^[A-Z0-9]+-[A-Z0-9]+-\d{6}-\d+(?:\.\d+)?-[CP]$").all()).item(), \
        "okex-options option pattern expected: ETH-USD-260529-1600-P"
    
    markup_columns = ("CUR1", "CUR2", "exp", "exp_str", "strike", "pc", "type", "ref_sym", "fut_sym", "spot_sym", "inverse")
    symbol = pl.col("symbol").cast(pl.String)
    parts = symbol.str.split_exact("-", 4)
    cur1 = parts.struct.field("field_0")
    cur2 = parts.struct.field("field_1")
    quote_ccy = pl.when(cur2.str.to_uppercase() == pl.lit("USD")).then(pl.lit("USDT")).otherwise(cur2)
    exp_str = parts.struct.field("field_2")
    strike = parts.struct.field("field_3").cast(pl.Float64)
    pc = parts.struct.field("field_4")

    expr_map = {
        "CUR1": cur1,
        "CUR2": cur2,
        "exp": exp_str.str.strptime(pl.Datetime("us"), format="%y%m%d", strict=True),
        "exp_str": exp_str,
        "strike": strike,
        "pc": pc,
        "type": pl.lit("option"),
        "ref_sym": pl.when(cur2.str.to_uppercase().str.starts_with("USD")).then(pl.concat_str([cur1, pl.lit("USD")])).otherwise(pl.concat_str([cur1, pl.lit("-"), cur2])),
        "fut_sym": pl.concat_str([cur1, pl.lit("-"), quote_ccy, pl.lit("-"), exp_str]),
        "spot_sym": pl.concat_str([cur1, pl.lit("-"), quote_ccy]),
        "inverse": pl.lit(False),
    }
    return expr_map, markup_columns


def _okex_options_trades_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert OKEX options trades symbols to markup expressions."""
    return _okex_options_quotes_markup(df)


def _okex_futures_quotes_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert OKEX futures quote/trade symbols to markup expressions.
    
    Pattern: CUR1-CUR2-YYMMDD
    Example: BTC-USD-260925
    """
    assert df.select(pl.col("symbol").cast(pl.String).str.contains(r"^[A-Z0-9]+-[A-Z0-9]+-\d{6}$").all()).item(), \
        "okex-futures future pattern expected: ETH-USDT-240613"
    
    markup_columns = ("CUR1", "CUR2", "exp", "exp_str", "ref_sym", "spot_sym", "inverse")
    symbol = pl.col("symbol").cast(pl.String)
    parts = symbol.str.split_exact("-", 2)
    cur1 = parts.struct.field("field_0")
    cur2 = parts.struct.field("field_1")
    exp_str = parts.struct.field("field_2")
    quote_ccy = pl.when(cur2.str.to_uppercase() == pl.lit("USD")).then(pl.lit("USDT")).otherwise(cur2)

    expr_map = {
        "CUR1": cur1,
        "CUR2": cur2,
        "exp": exp_str.str.strptime(pl.Datetime("us"), format="%y%m%d", strict=True),
        "exp_str": exp_str,
        "ref_sym": pl.when(cur2.str.to_uppercase().str.starts_with("USD")).then(pl.concat_str([cur1, pl.lit("USD")])).otherwise(pl.concat_str([cur1, pl.lit("-"), cur2])),
        "spot_sym": pl.concat_str([cur1, pl.lit("-"), quote_ccy]),
        "inverse": cur2.str.to_uppercase() == pl.lit("USD"),
    }
    return expr_map, markup_columns


def _okex_futures_trades_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert OKEX futures trades symbols to markup expressions."""
    return _okex_futures_quotes_markup(df)


def _okex_spot_quotes_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert OKEX spot quote/trade symbols to markup expressions.
    
    Pattern: CUR1-CUR2
    Example: ETH-USDC
    """
    assert df.select(pl.col("symbol").cast(pl.String).str.contains(r"^[A-Z0-9]+-[A-Z0-9]+$").all()).item(), \
        "okex spot pattern expected: ETH-USD"
    
    markup_columns = ("CUR1", "CUR2", "ref_sym", "spot_sym")
    symbol = pl.col("symbol").cast(pl.String)
    parts = symbol.str.split_exact("-", 1)
    cur1 = parts.struct.field("field_0")
    cur2 = parts.struct.field("field_1")

    expr_map = {
        "CUR1": cur1,
        "CUR2": cur2,
        "ref_sym": pl.when(cur2.str.to_uppercase().str.starts_with("USD")).then(pl.concat_str([cur1, pl.lit("USD")])).otherwise(pl.concat_str([cur1, pl.lit("-"), cur2])),
        "spot_sym": symbol,
    }
    return expr_map, markup_columns


def _okex_spot_trades_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert OKEX spot trades symbols to markup expressions."""
    return _okex_spot_quotes_markup(df)


# ============================================================================
# DERIBIT CONVERTERS (options, futures, spot)
# ============================================================================

def _deribit_quotes_options_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert Deribit options quote/trade symbols to markup expressions.
    
    Pattern: ROOT-DDMMMYY-STRIKETOKEN-PC
    Example: XRP_USDC-27MAR26-1D375-P or AVAX_USDC-16APR26-9-P
    Where ROOT can be CUR1_CUR2 or just CUR1 (defaults CUR2 to USD)
    And STRIKETOKEN is either an integer (9) or uses D as decimal separator (1D375 = 1.375)
    """
    symbol = pl.col("symbol").cast(pl.String)
    assert df.select(symbol.str.contains(r"^[A-Z_0-9]+-\d{1,2}[A-Z]{3}\d{2}-\d+(?:D\d+)?-[CP]$").all()).item(), \
        "deribit option pattern expected: XRP_USDC-27MAR26-1D375-P or AVAX_USDC-16APR26-9-P"
    
    markup_columns = ("CUR1", "CUR2", "exp", "exp_str", "strike", "pc", "type", "ref_sym", "fut_sym", "spot_sym", "inverse")

    # Split on "-" to get root, date, strike token, and pc
    parts = symbol.str.split("-")
    root = parts.list.get(0)
    exp_str = parts.list.get(1)
    strike_token = parts.list.get(2)  # e.g., "1D375"
    pc = parts.list.get(3)  # e.g., "P"

    # Parse strike token: integer "9" or decimal token "1D375" -> "1.375".
    strike = strike_token.str.replace("D", ".").cast(pl.Float64)

    # Parse CUR1_CUR2 from root: split on "_", default CUR2 to USD if not present
    root_parts = root.str.split_exact("_", n=1)
    cur1 = root_parts.struct.field("field_0")
    cur2 = pl.coalesce([root_parts.struct.field("field_1"), pl.lit("USD")])

    expr_map = {
        "CUR1": cur1,
        "CUR2": cur2,
        "exp": exp_str.str.strptime(pl.Datetime("us"), format="%d%b%y", strict=True),
        "exp_str": exp_str,
        "strike": strike,
        "pc": pc,
        "type": pl.lit("option"),
        "ref_sym": pl.when(cur2.str.to_uppercase().str.starts_with("USD")).then(pl.concat_str([cur1, pl.lit("USD")])).otherwise(pl.concat_str([cur1, cur2])),
        "fut_sym": pl.concat_str([root, pl.lit("-"), exp_str]),
        "spot_sym": pl.when(root.str.contains("_"))
        .then(root)
        .when(cur1.is_in(["BTC", "ETH"]))
        .then(pl.concat_str([cur1, pl.lit("_USDC")]))
        .otherwise(cur1),
        "inverse": pl.lit(True),
    }
    return expr_map, markup_columns


def _deribit_trades_option_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert Deribit trades options symbols to markup expressions."""
    return _deribit_quotes_options_markup(df)


def _deribit_quotes_futures_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert Deribit futures quote/trade symbols to markup expressions.
    
    Pattern: ROOT-DDMMMYY
    Example: ETH_USDC-27MAR26 or ETH-6MAR26
    Where ROOT can be CUR1_CUR2 or just CUR1 (defaults CUR2 to USD)
    And DDMMMYY is 1-2 digit day + 3-letter month + 2-digit year
    """
    symbol = pl.col("symbol").cast(pl.String)
    assert df.select(symbol.str.contains(r"^[A-Z_0-9]+-\d{1,2}[A-Z]{3}\d{2}$").all()).item(), \
        "deribit future pattern expected: ETH_USDC-6MAR26 or ETH-6MAR26"
    
    markup_columns = ("CUR1", "CUR2", "exp", "exp_str", "ref_sym", "spot_sym", "inverse")

    # Split on "-" to get root and date
    parts = symbol.str.split("-")
    root = parts.list.get(0)
    exp_str = parts.list.get(1)

    # Parse CUR1_CUR2 from root: split_exact defaults missing parts to null, coalesce provides USD fallback
    root_parts = root.str.split_exact("_", n=1)
    cur1 = root_parts.struct.field("field_0")
    cur2 = pl.coalesce([root_parts.struct.field("field_1"), pl.lit("USD")])

    expr_map = {
        "CUR1": cur1,
        "CUR2": cur2,
        "exp": exp_str.str.strptime(pl.Datetime("us"), format="%d%b%y", strict=True),
        "exp_str": exp_str,
        "ref_sym": pl.when(cur2.str.to_uppercase().str.starts_with("USD")).then(pl.concat_str([cur1, pl.lit("USD")])).otherwise(pl.concat_str([cur1, cur2])),
        "spot_sym": pl.concat_str([cur1, pl.lit("-"), cur2]),
        "inverse": pl.lit(True),
    }
    return expr_map, markup_columns


def _deribit_trades_future_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert Deribit trades futures symbols to markup expressions."""
    return _deribit_quotes_futures_markup(df)


def _deribit_derivative_ticker_future_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert Deribit derivative_ticker futures symbols to markup expressions.

    Uses the same symbol parsing as Deribit futures quotes and sets top-of-book
    fields from mark_price.
    """
    symbol = pl.col("symbol").cast(pl.String)
    assert df.select(symbol.str.contains(r"^[A-Z_0-9]+-\d{1,2}[A-Z]{3}\d{2}$").all()).item(), \
        "deribit future pattern expected: ETH_USDC-6MAR26 or ETH-6MAR26"

    markup_columns = (
        "CUR1",
        "CUR2",
        "exp",
        "exp_str",
        "ref_sym",
        "spot_sym",
        "inverse",
        "bid_price",
        "ask_price",
        "bid_amount",
        "ask_amount",
    )

    parts = symbol.str.split("-")
    root = parts.list.get(0)
    exp_str = parts.list.get(1)

    root_parts = root.str.split_exact("_", n=1)
    cur1 = root_parts.struct.field("field_0")
    cur2 = pl.coalesce([root_parts.struct.field("field_1"), pl.lit("USD")])

    expr_map = {
        "CUR1": cur1,
        "CUR2": cur2,
        "exp": exp_str.str.strptime(pl.Datetime("us"), format="%d%b%y", strict=True),
        "exp_str": exp_str,
        "ref_sym": pl.when(cur2.str.to_uppercase().str.starts_with("USD")).then(pl.concat_str([cur1, pl.lit("USD")])).otherwise(pl.concat_str([cur1, cur2])),
        "spot_sym": pl.concat_str([cur1, pl.lit("-"), cur2]),
        "inverse": pl.lit(True),
        "bid_price": pl.col("mark_price"),
        "ask_price": pl.col("mark_price"),
        "bid_amount": pl.lit(None, dtype=pl.Float64),
        "ask_amount": pl.lit(None, dtype=pl.Float64),
    }
    return expr_map, markup_columns


def _deribit_quotes_spot_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert Deribit spot quote/trade symbols to markup expressions.
    
    Pattern: CUR1-CUR2 or CUR1_CUR2
    Example: ETH-USD or ETH_USDC
    """
    symbol = pl.col("symbol").cast(pl.String)
    assert df.select(symbol.str.contains(r"^[A-Z0-9]+[-_][A-Z0-9]+$").all()).item(), \
        "deribit spot pattern expected: ETH-USD or ETH_USDC"
    
    markup_columns = ("CUR1", "CUR2", "ref_sym", "spot_sym")

    # Split on either separator and preserve the original symbol in spot_sym.
    parts = symbol.str.replace("_", "-", literal=True).str.split("-")
    cur1 = parts.list.get(0)
    cur2 = parts.list.get(1)

    expr_map = {
        "CUR1": cur1,
        "CUR2": cur2,
        "ref_sym": pl.when(cur2.str.to_uppercase().str.starts_with("USD")).then(pl.concat_str([cur1, pl.lit("USD")])).otherwise(pl.concat_str([cur1, cur2])),
        "spot_sym": symbol,
    }
    return expr_map, markup_columns


def _deribit_trades_spot_markup(df: pl.DataFrame) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Convert Deribit trades spot symbols to markup expressions."""
    return _deribit_quotes_spot_markup(df)


# ============================================================================
# DISPATCH AND PUBLIC API
# ============================================================================

def _normalize_symbol_type(symbol_type: str | None) -> str | None:
    if symbol_type is None:
        return None
    st = symbol_type.lower()
    if st == "futures":
        return "future"
    if st == "options":
        return "option"
    return st


def _infer_symbol_markup(
    exchange: str,
    data_type: str | None,
    symbol_type: str | None,
    df: pl.DataFrame,
) -> tuple[dict[str, pl.Expr], tuple[str, ...]]:
    """Dispatch vectorized symbol markup expressions by (exchange, data_type, symbol_type)."""
    exch = (exchange or "").lower()
    dt = (data_type or "").lower()
    st = _normalize_symbol_type(symbol_type)

    # OKEX family dispatches purely on (exchange, data_type).
    key2 = (exch, dt)
    if key2 == ("okex-options", "quotes"):
        return _okex_options_quotes_markup(df)
    if key2 == ("okex-options", "trades"):
        return _okex_options_trades_markup(df)
    if key2 == ("okex-futures", "quotes"):
        return _okex_futures_quotes_markup(df)
    if key2 == ("okex-futures", "trades"):
        return _okex_futures_trades_markup(df)
    if key2 == ("okex", "quotes"):
        return _okex_spot_quotes_markup(df)
    if key2 == ("okex", "trades"):
        return _okex_spot_trades_markup(df)

    if st is not None:
        key3 = (exch, dt, st)
        if key3 in {
            ("deribit", "quotes", "spot"),
        }:
            return _deribit_quotes_spot_markup(df)
        if key3 in {
            ("deribit", "quotes", "future"),
        }:
            return _deribit_quotes_futures_markup(df)
        if key3 in {
            ("deribit", "quotes", "option"),
        }:
            return _deribit_quotes_options_markup(df)
        if key3 in {
            ("deribit", "trades", "option"),
        }:
            return _deribit_trades_option_markup(df)
        if key3 in {
            ("deribit", "trades", "future"),
        }:
            return _deribit_trades_future_markup(df)
        if key3 in {
            ("deribit", "trades", "spot"),
        }:
            return _deribit_trades_spot_markup(df)
        if key3 in {
            ("deribit", "derivative_ticker", "future"),
        }:
            return _deribit_derivative_ticker_future_markup(df)
        return _default_symbol_markup()
    return _default_symbol_markup()


def mark_up(
    df: pl.DataFrame,
    exchange: str | None,
    data_type: str | None,
    symbol_type: str | None = None,
) -> pl.DataFrame:
    """Add placeholder symbol-markup columns using vectorized Polars expressions.

    When `data_type` is provided, markup is applied to rows matching the
    selected exchange and data_type. If symbol_type is provided, dispatch is
    performed on (exchange, data_type, symbol_type).
    """
    if df.is_empty():
        return df
    if "symbol" not in df.columns:
        logger.warning("mark_up received dataframe without symbol column")
        return df

    resolved_exchange = exchange

    if "exchange" in df.columns:
        unique_exchanges = [x for x in df.get_column("exchange").drop_nulls().unique().to_list()]
        assert len(unique_exchanges) == 1, "expected one unique exchange in dataframe"
        df_exchange = str(unique_exchanges[0])
        if resolved_exchange is None:
            resolved_exchange = df_exchange
        else:
            assert df_exchange == resolved_exchange, "dataframe exchange does not match exchange argument"
    else:
        assert resolved_exchange is not None, "exchange must be provided when dataframe has no exchange column"

    markup_expr, markup_columns = _infer_symbol_markup(resolved_exchange, data_type, symbol_type, df)
    return df.with_columns([markup_expr[name].alias(name) for name in markup_columns])


def access_files(
    exchange,
    datatype,
    date,
    symbol_list,
    sample_freq: str = "5min",
    raw_data_dir: str | None = None,
):
    """Return concatenated downloaded data and discovered symbols.

    Returns
    -------
    (tuple[pl.DataFrame, list[str]])
        A tuple of:
        - concatenated dataframe from produced parquet file(s)
        - symbols discovered in the produced parquet file(s)
    """

    data_dir = raw_data_dir if raw_data_dir is not None else f"datasets/{exchange}_raw/"

    parquet_paths = download_resample(
        exchange=exchange,
        data_type=datatype,
        symbols=symbol_list,
        start_date=date,
        end_date=date,
        resample_freq=sample_freq,
        data_dir=data_dir,
    )

    frames: list[pl.DataFrame] = []
    discovered_symbols: list[str] = []
    for p in parquet_paths:
        path = Path(p)
        if not path.exists():
            logger.warning("Skipping missing parquet path returned by download_resample: %s", path)
            continue

        df = pl.read_parquet(path)
        if "symbol" not in df.columns:
            logger.warning("Skipping parquet with no symbol column: %s", path)
            continue

        file_symbols = [str(sym) for sym in df.get_column("symbol").to_list()]
        discovered_symbols.extend(file_symbols)

        frames.append(df)

    # Preserve appearance order while deduplicating symbols found across files.
    unique_symbols = list(dict.fromkeys(discovered_symbols))

    if not frames:
        return pl.DataFrame(), unique_symbols
    return pl.concat(frames, rechunk=False), unique_symbols
