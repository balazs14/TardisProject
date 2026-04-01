#from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

DEFAULT_JOIN_KEYS = ("timestamp", "symbol", "exchange")


def _qident(name: str) -> str:
    assert name, "empty identifier"
    return '"' + name.replace('"', '""') + '"'


def _qlit(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _read_parquet_select(path: str | Path, cols: list[str]) -> str:
    col_sql = ", ".join(_qident(c) for c in cols)
    return f"SELECT {col_sql} FROM read_parquet({_qlit(str(path))})"


def _const_sql(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return _qlit(str(value))


def parquet_columns(path: str | Path) -> list[str]:
    con = duckdb.connect()
    try:
        rows = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def parquet_symbols(path: str | Path, limit: int | None = 100) -> list[str]:
    con = duckdb.connect()
    try:
        sql = "SELECT DISTINCT symbol FROM read_parquet(?) ORDER BY 1"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = con.execute(sql, [str(path)]).fetchall()
        return [r[0] for r in rows if r[0] is not None]
    finally:
        con.close()


def parquet_timestamps(path: str | Path, limit: int | None = 100) -> list[object]:
    con = duckdb.connect()
    try:
        sql = "SELECT DISTINCT timestamp FROM read_parquet(?) ORDER BY 1"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = con.execute(sql, [str(path)]).fetchall()
        return [r[0] for r in rows if r[0] is not None]
    finally:
        con.close()


def parquet_summary(path: str | Path) -> dict[str, object]:
    con = duckdb.connect()
    try:
        row = con.execute(
            """
            SELECT
              COUNT(*) AS n_rows,
              COUNT(DISTINCT symbol) AS n_symbols,
              MIN(timestamp) AS min_ts,
              MAX(timestamp) AS max_ts
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
        return {
            "path": str(path),
            "n_rows": row[0],
            "n_symbols": row[1],
            "min_ts": row[2],
            "max_ts": row[3],
        }
    finally:
        con.close()


def inspect_inputs(
    sources: dict[str, str | Path],
    *,
    symbol_limit: int = 3,
    ts_limit: int = 25,
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for source, path in sources.items():
        p = Path(path)
        assert p.exists(), f"missing parquet file: {p}"
        info = parquet_summary(p)
        info["columns"] = parquet_columns(p)
        info["symbols_preview"] = parquet_symbols(p, limit=symbol_limit)
        info["timestamps_preview"] = parquet_timestamps(p, limit=ts_limit)
        report[source] = info
        logger.info(
            "source=%s rows=%s symbols=%s ts=[%s, %s]"
            "\n    columns=%s"
            "\n    symbols_preview=%s",
            source,
            info["n_rows"],
            info["n_symbols"],
            info["min_ts"],
            info["max_ts"],
            info['columns'],
            info['symbols_preview'],
        )
    return report


def zip_join_parquets(
    *,
    sources: dict[str, str | Path],
    column_map: dict[str, object],
    output_path: str | Path,
    join_keys: tuple[str, ...] = DEFAULT_JOIN_KEYS,
    join_type: str = "inner",
    where_sql: str | None = None,
) -> Path:
    assert sources, "sources must not be empty"
    assert column_map, "column_map must not be empty"
    assert join_type.lower() in {"inner", "left", "right", "full", "full outer"}
    assert len(set(join_keys)) == len(join_keys), "duplicate join keys"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    needed: dict[str, set[str]] = {src: set(join_keys) for src in sources}
    source_mappings: dict[str, tuple[str, str]] = {}
    for out_col, spec in column_map.items():
        assert out_col not in join_keys, f"output column collides with join key: {out_col}"
        if isinstance(spec, (tuple, list)):
            assert len(spec) == 2, f"column_map[{out_col}] tuple/list must have 2 items"
            src, src_col = spec
            assert src in sources, f"unknown source in column_map: {src}"
            source_mappings[out_col] = (src, src_col)
            needed[src].add(src_col)

    for src, path in sources.items():
        p = Path(path)
        assert p.exists(), f"missing parquet file: {p}"
        cols = set(parquet_columns(p))
        missing = sorted(c for c in needed[src] if c not in cols)
        assert not missing, f"source={src} missing columns: {missing}"

    aliases = list(sources.keys())
    base = aliases[0]
    join_type_sql = "FULL OUTER" if join_type.lower() == "full" else join_type.upper()

    from_sql = f"({_read_parquet_select(sources[base], sorted(needed[base]))}) AS {_qident(base)}"
    for src in aliases[1:]:
        right = f"({_read_parquet_select(sources[src], sorted(needed[src]))}) AS {_qident(src)}"
        on_sql = " AND ".join(
            f"{_qident(base)}.{_qident(k)} = {_qident(src)}.{_qident(k)}" for k in join_keys
        )
        from_sql += f" {join_type_sql} JOIN {right} ON {on_sql}"

    select_sql = [f"{_qident(base)}.{_qident(k)} AS {_qident(k)}" for k in join_keys]
    for out_col, spec in column_map.items():
        if isinstance(spec, (tuple, list)):
            src, src_col = source_mappings[out_col]
            select_sql.append(f"{_qident(src)}.{_qident(src_col)} AS {_qident(out_col)}")
        else:
            select_sql.append(f"{_const_sql(spec)} AS {_qident(out_col)}")

    order_keys = ["timestamp"]
    if "symbol" in join_keys:
        order_keys.append("symbol")
    if "exchange" in join_keys:
        order_keys.append("exchange")
    order_sql = ", ".join(_qident(k) for k in order_keys)

    query = (
        f"SELECT {', '.join(select_sql)} FROM {from_sql}"
        + (f" WHERE {where_sql}" if where_sql else "")
        + f" ORDER BY {order_sql}"
    )
    copy_sql = (
        f"COPY ({query}) TO {_qlit(str(output))} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )

    logger.info(
        "zip-join start: sources=%s mapped_cols=%d join=%s -> %s",
        len(sources),
        len(column_map),
        join_type_sql,
        output,
    )
    con = duckdb.connect()
    try:
        con.execute(copy_sql)
    finally:
        con.close()

    assert output.exists(), f"expected output parquet missing: {output}"
    logger.info("zip-join done: %s", output)
    return output
