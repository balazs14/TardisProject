import argparse
import logging
import sys
import os


class _CliHelpFormatter(argparse.HelpFormatter):
    """Show [required] prefix for required options; show (default: X) for non-None defaults."""

    def _get_help_string(self, action):
        help_str = action.help or ""
        if getattr(action, "required", False):
            return "[required]  " + help_str
        if (
            action.default not in (None, argparse.SUPPRESS)
            and "%(default)" not in help_str
            and "default" not in help_str.lower()
        ):
            if action.option_strings or action.nargs in (argparse.OPTIONAL, argparse.ZERO_OR_MORE):
                help_str += " (default: %(default)s)"
        return help_str

global_simulation_ts = None

class GlobalContextAdapter(logging.LoggerAdapter):
    def set_global_simulation_ts(self, ts):
        global global_simulation_ts
        global_simulation_ts = ts
        return
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra["simts"] = global_simulation_ts   # inject global variable
        kwargs["extra"] = extra
        return msg, kwargs

def get_my_logger(name):
    return GlobalContextAdapter(logging.getLogger(name), {})

def config_logging(fname):
    pkg_logger = logging.getLogger(__name__) # root logger for package
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # Remove all handlers
    for h in pkg_logger.handlers[:]: 
        pkg_logger.removeHandler(h)
        #h.close()
    
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            if not hasattr(record, "simts"):
                record.simts = "---- -- -- -- -- --"
            return super().format(record)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = SafeFormatter("%(simts)s %(name)-20s: %(message)s")
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)

    handler = logging.FileHandler(fname, mode='w')
    handler.setLevel(logging.INFO)
    formatter = SafeFormatter("%(simts)s | %(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s: %(message)s")
    
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)


def package_set_log_level(level=logging.INFO, file_level=logging.DEBUG):
    pkg_logger = logging.getLogger(__name__) # root logger for package
    pkg_logger.setLevel(logging.DEBUG) # logger's level is applied first, then handlers filter
    for h in pkg_logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(level)
        if isinstance(h, logging.FileHandler):
            h.setLevel(file_level)
    
