import logging
import sys
import os

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
    
    # for name, logger in logging.root.manager.loggerDict.items():
    #     if isinstance(logger, logging.Logger) and name.startswith(__package__):
    #         #import pdb; pdb.set_trace()
    #         for h in logger.handlers:
    #             if isinstance(h, logging.StreamHandler):
    #                 h.setFormatter(fmt)
    #                 h.setLevel(level)
    #             if isinstance(h, logging.FileHandler):
    #                 fmt = logging.Formatter("(%asctime)s %(levelname)-8s | %(name)-15s | %(message)s")
    #                 h.setFormatter(fmt)
    #                 h.setLevel(file_level)
