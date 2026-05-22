import argparse
import json
import logging
import os

import argcomplete

from tardis import _CliHelpFormatter
from tardis import align_derivative_ticker_trades_quotes
from tardis import align_option_chain_trades_quotes
from tardis import align_put_call_data
from tardis import download_files
from tardis import tardis_universe

UNIVERSE_COMMANDS = {"exchanges", "symbols", "columns", "data-types", "key-info"}


def _add_loglevel_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loglevel",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=(
            "Logging level for CLI execution. "
            "When omitted, uses TARDIS_UNIVERSE_LOGLEVEL if set, otherwise WARNING."
        ),
    )


def _add_universe_subparsers(sub: argparse._SubParsersAction) -> None:
    ex_p = sub.add_parser("exchanges", help="List all dataset-capable exchanges", formatter_class=_CliHelpFormatter, allow_abbrev=False)
    _add_loglevel_arg(ex_p)

    sym_p = sub.add_parser("symbols", help="List downloadable symbols for one or more exchanges", formatter_class=_CliHelpFormatter, allow_abbrev=False)
    _add_loglevel_arg(sym_p)
    group = sym_p.add_mutually_exclusive_group(required=True)
    group.add_argument("--exchange", help="Exchange id, e.g. 'okex-options', 'deribit', 'binance'.")
    group.add_argument("--exchange-regex", help="Regex to match one or more exchanges by id.")
    sym_p.add_argument("--symbol-type-regex", dest="symbol_type_regex", default=None, metavar="REGEX")
    sym_p.add_argument("--symbol-regex", dest="symbol_regex", default=None, metavar="REGEX")
    sym_p.add_argument("--stream-type-regex", dest="stream_type_regex", default=None, metavar="REGEX")
    sym_p.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    sym_p.add_argument("--show-available-types", dest="show_available_types", action="store_true", default=True)
    sym_p.add_argument("--no-show-available-types", dest="show_available_types", action="store_false")
    sym_p.add_argument("--json", action="store_true")
    sym_p.add_argument("--head", type=int, default=None, metavar="N")

    dt_p = sub.add_parser("data-types", help="Summarise data-types available per symbol type for an exchange", formatter_class=_CliHelpFormatter, allow_abbrev=False)
    _add_loglevel_arg(dt_p)
    group2 = dt_p.add_mutually_exclusive_group(required=True)
    group2.add_argument("--exchange", help="Exchange id, e.g. 'deribit', 'okex-options'.")
    group2.add_argument("--exchange-regex", help="Regex to match one or more exchanges by id.")

    col_p = sub.add_parser("columns", help="Preview columns + 2 sample rows from matched downloadable stream files", formatter_class=_CliHelpFormatter, allow_abbrev=False)
    _add_loglevel_arg(col_p)
    group3 = col_p.add_mutually_exclusive_group(required=True)
    group3.add_argument("--exchange", help="Exchange id, e.g. 'deribit', 'okex-options'.")
    group3.add_argument("--exchange-regex", help="Regex to match one or more exchanges by id.")
    col_p.add_argument("--symbol-type-regex", dest="symbol_type_regex", default=None, metavar="REGEX")
    col_p.add_argument("--symbol-regex", dest="symbol_regex", default=None, metavar="REGEX")
    col_p.add_argument("--stream-type-regex", dest="stream_type_regex", default=None, metavar="REGEX")
    col_p.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    col_p.add_argument("--head", type=int, default=None, metavar="N")
    col_p.add_argument("--json", action="store_true")

    ki_p = sub.add_parser("key-info", help="Show what your API key covers (exchanges, date ranges, data plan)", formatter_class=_CliHelpFormatter, allow_abbrev=False)
    _add_loglevel_arg(ki_p)


def _add_download_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "download-files",
        help="Download+resample Tardis stream files",
        description="Run tardis.download_files download_resample pipeline.",
        formatter_class=_CliHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--exchange", required=True, help="Exchange name (e.g. deribit, okex, binance)")
    p.add_argument("--data-type", required=True, dest="data_type", help="Tardis data type (e.g. trades, quotes, options_chain)")
    p.add_argument("--symbol", required=True, help="Instrument symbol in exchange notation")
    p.add_argument("--start-date", required=True, dest="start_date", help="Inclusive start date (YYYY-MM-DD)")
    p.add_argument("--end-date", default=None, dest="end_date", help="Inclusive end date (YYYY-MM-DD). Defaults to --start-date when omitted.")
    p.add_argument("--data-dir", default=".", help="Directory where files are downloaded and stored")
    p.add_argument("--force-reload", action="store_true", help="Force conversion even if target parquet already exists")
    p.add_argument("--resample-freq", default=download_files.DEFAULT_RESAMPLE_FREQ, help="Resample frequency before parquet write (e.g. 1min, 5min)")
    p.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    p.add_argument("--peek", action="store_true", help="Stream one raw chunk and print head + inferred schema, then exit")
    p.add_argument("--peek-rows", type=int, default=10, help="Number of rows to print in --peek mode")
    p.add_argument(
        "--peek-transposed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display --peek preview transposed so all columns are visible",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Download and save raw .csv.gz files only (no resample/parquet conversion)",
    )


def _add_align_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "align-put-call",
        help="Build aligned call/put/future/spot data",
        description="Run tardis.align_put_call_data alignment pipeline.",
        formatter_class=_CliHelpFormatter,
        allow_abbrev=False,
    )
    _add_loglevel_arg(p)
    p.add_argument("--exchange", required=True, choices=["okex", "deribit"], help="Exchange to process")
    p.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    p.add_argument("--sample-freq", default="5min", help="Sampling frequency label for output naming")
    recreate_group = p.add_mutually_exclusive_group()
    recreate_group.add_argument(
        "--recreate-existing",
        dest="recreate_existing",
        action="store_true",
        help="Recreate output parquet even when it already exists.",
    )
    recreate_group.add_argument(
        "--no-recreate-existing",
        dest="recreate_existing",
        action="store_false",
        help="Skip processing when the output parquet path already exists (default).",
    )
    p.set_defaults(recreate_existing=False)
    p.add_argument(
        "--raw-data-dir",
        default="datasets/{exchange}_raw/",
        help="Directory for download_resample parquet outputs",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output parquet path. Derived from exchange/date/sample-freq when omitted.",
    )


def _add_align_option_chain_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "align-option-chain-trades-quotes",
        help="Build aligned option-chain/trades/quotes data",
        description="Run tardis.align_option_chain_trades_quotes alignment pipeline.",
        formatter_class=_CliHelpFormatter,
        allow_abbrev=False,
    )
    _add_loglevel_arg(p)
    p.add_argument("--exchange", required=True, choices=["okex", "deribit"], help="Exchange to process")
    p.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    p.add_argument("--symbol", default="OPTIONS", help="Instrument symbol filter (defaults to OPTIONS)")
    p.add_argument("--sample-freq", default="5min", help="Sampling frequency label for output naming")
    recreate_group = p.add_mutually_exclusive_group()
    recreate_group.add_argument(
        "--recreate-existing",
        dest="recreate_existing",
        action="store_true",
        help="Recreate output parquet even when it already exists.",
    )
    recreate_group.add_argument(
        "--no-recreate-existing",
        dest="recreate_existing",
        action="store_false",
        help="Skip processing when the output parquet path already exists (default).",
    )
    p.set_defaults(recreate_existing=False)
    p.add_argument(
        "--raw-data-dir",
        default="datasets/{exchange}_raw/",
        help="Directory for download_resample parquet outputs",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output parquet path. Derived from exchange/date/sample-freq when omitted.",
    )


def _add_align_derivative_ticker_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "align-derivative-ticker-trades-quotes",
        help="Build aligned derivative-ticker/trades/quotes data",
        description="Run tardis.align_derivative_ticker_trades_quotes alignment pipeline.",
        formatter_class=_CliHelpFormatter,
        allow_abbrev=False,
    )
    _add_loglevel_arg(p)
    p.add_argument("--exchange", required=True, choices=["okex", "deribit"], help="Exchange to process")
    p.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    p.add_argument("--symbol", required=True, help="Instrument symbol to align (e.g. BTC-24APR26)")
    p.add_argument("--sample-freq", default="5min", help="Sampling frequency label for output naming")
    recreate_group = p.add_mutually_exclusive_group()
    recreate_group.add_argument(
        "--recreate-existing",
        dest="recreate_existing",
        action="store_true",
        help="Recreate output parquet even when it already exists.",
    )
    recreate_group.add_argument(
        "--no-recreate-existing",
        dest="recreate_existing",
        action="store_false",
        help="Skip processing when the output parquet path already exists (default).",
    )
    p.set_defaults(recreate_existing=False)
    p.add_argument(
        "--raw-data-dir",
        default="datasets/{exchange}_raw/",
        help="Directory for download_resample parquet outputs",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output parquet path. Derived from exchange/date/sample-freq when omitted.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tardisctl",
        description="Explore the Tardis downloadable CSV universe",
        formatter_class=_CliHelpFormatter,
        allow_abbrev=False,
    )
    _add_loglevel_arg(parser)
    sub = parser.add_subparsers(dest="cmd")

    _add_universe_subparsers(sub)
    _add_download_subparser(sub)
    _add_align_subparser(sub)
    _add_align_option_chain_subparser(sub)
    _add_align_derivative_ticker_subparser(sub)
    return parser


def _resolve_effective_loglevel(args: argparse.Namespace) -> str:
    return (args.loglevel or os.environ.get("TARDIS_UNIVERSE_LOGLEVEL") or "WARNING").upper()


def _configure_global_logging(loglevel: str) -> None:
    level = getattr(logging, loglevel.upper(), logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _run_universe_cmd(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.cmd == "exchanges":
        exs = tardis_universe.universe_exchanges()
        tardis_universe._print_exchanges(exs)
        return

    if args.cmd == "symbols":
        import re

        all_exs = [e["id"] for e in tardis_universe.universe_exchanges()]
        if args.exchange:
            exchanges = [args.exchange]
        else:
            pat = re.compile(args.exchange_regex, re.IGNORECASE)
            exchanges = [e for e in all_exs if pat.search(e)]

        for exch in exchanges:
            total = len(
                tardis_universe._filtered_symbols(
                    exchange=exch,
                    symbol_type_regex=args.symbol_type_regex,
                    symbol_regex=args.symbol_regex,
                    stream_type_regex=args.stream_type_regex,
                    date=args.date,
                )
            )
            syms = tardis_universe._filtered_symbols(
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
                print(
                    f"\n{len(syms)} symbols for {exch}"
                    + (f"  (symbol-type-regex={args.symbol_type_regex})" if args.symbol_type_regex else "")
                    + (f"  (symbol-regex={args.symbol_regex})" if args.symbol_regex else "")
                    + (f"  (stream-type-regex={args.stream_type_regex})" if args.stream_type_regex else "")
                    + (f"  (head={args.head})" if args.head is not None else "")
                    + (f"  (date={args.date})" if args.date else "")
                )
                tardis_universe._print_symbols(syms, show_data_types=args.show_available_types)
                if args.head is not None and total > len(syms):
                    print(f"\nShowing first {len(syms)} of {total} matching symbols.")
        return

    if args.cmd == "columns":
        import re

        all_exs = [e["id"] for e in tardis_universe.universe_exchanges()]
        if args.exchange:
            exchanges = [args.exchange]
        else:
            pat = re.compile(args.exchange_regex, re.IGNORECASE)
            exchanges = [e for e in all_exs if pat.search(e)]
        for exch in exchanges:
            try:
                previews = tardis_universe._matched_stream_columns_head(
                    exchange=exch,
                    symbol_type_regex=args.symbol_type_regex,
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
                    + (f"  (symbol-type-regex={args.symbol_type_regex})" if args.symbol_type_regex else "")
                    + (f"  (symbol-regex={args.symbol_regex})" if args.symbol_regex else "")
                    + (f"  (stream-type-regex={args.stream_type_regex})" if args.stream_type_regex else "")
                    + (f"  (date={args.date})" if args.date else "")
                    + (f"  (head={args.head})" if args.head is not None else "")
                )
                tardis_universe._print_columns_transposed(previews)
        return

    if args.cmd == "data-types":
        import re

        all_exs = [e["id"] for e in tardis_universe.universe_exchanges()]
        if args.exchange:
            exchanges = [args.exchange]
        else:
            pat = re.compile(args.exchange_regex, re.IGNORECASE)
            exchanges = [e for e in all_exs if pat.search(e)]
        for exch in exchanges:
            coverage = tardis_universe.universe_data_types(exchange=exch)[exch]
            print(f"\nData-types available on {exch}:")
            for t, dts in sorted(coverage.items()):
                print(f"  {t:<20} {dts}")
        return

    if args.cmd == "key-info":
        info = tardis_universe._key_info()
        print("\nAPI key coverage:")
        tardis_universe._print_key_info(info)
        return


def run_cli(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    _configure_global_logging(_resolve_effective_loglevel(args))

    if args.cmd in UNIVERSE_COMMANDS:
        _run_universe_cmd(args, parser)
        return
    if args.cmd == "download-files":
        download_files.run_cli_args(args)
        return
    if args.cmd == "align-put-call":
        align_put_call_data.run_cli_args(args)
        return
    if args.cmd == "align-option-chain-trades-quotes":
        align_option_chain_trades_quotes.run_cli_args(args)
        return
    if args.cmd == "align-derivative-ticker-trades-quotes":
        align_derivative_ticker_trades_quotes.run_cli_args(args)
        return

    parser.print_help()


if __name__ == "__main__":
    run_cli()
