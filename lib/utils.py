from TardisProject import get_my_logger
logger = get_my_logger(__name__)

import asyncio
import functools
import inspect
import os
import contextlib
import pymysql
import pandas as pd
import numpy as np
import pickle
from .my_concurrent import futures as mfc
from collections.abc import Sequence

class ParameterError(Exception):
    pass
class InterruptError(Exception):
    pass

local_bucket = None
s3_bucket = None

def set_layer(layer='test'):
    global local_bucket
    global s3_bucket
    logger.info(f'setting layer to "{layer}"')
    local_bucket = '/my/data/to-be-determined/' + layer
    s3_bucket = 's3://to-be-determined/' + layer

def interruptible():
    def wrapper(func):
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            global stop_event
            stop_event = asyncio.Event()
            def inner_ctrl_c_signal_handler(sig, frame):
                '''                                                                                    
                function that gets called when the user issues a                                       
                keyboard interrupt (ctrl+c)                                                            
                '''
                logger.info("SIGINT caught!")
                stop_event.set()
            def inner_sigterm_signal_handler(loop):
                logger.info('received SIGINT')
                for task in asyncio.Task.all_tasks():
                    task.cancel()
                logger.info('raised CancelledError')
            # signal.signal(signal.SIGINT, inner_ctrl_c_signal_handler)                                
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGINT, functools.partial(
                inner_sigterm_signal_handler, loop))
            return await func(*args, **kwargs)

        return wrapped
    return wrapper

def wrap_in_sync(afn):
    def wrapped(*args, **kwargs):
        return asyncio.run(afn(*args, **kwargs))
    return wrapped

def run_in_parallel(coros, run_async_version=True, labels=None, nproc=None):

    if isinstance(coros,dict):
        labels = list(coros.keys())
        coros = [coros[k] for k in labels]
    if labels is None:
        labels = range(len(coros))
    if not isinstance(labels, (list,tuple)):
        labels = [labels]

    if len(labels) < len(coros):
        labels.extend(range(len(labels), len(coros)))

    fns = [ (wrap_in_sync(coro) if run_async_version else coro)
            for coro in coros ]
    
    res = {}
    if nproc is None: nproc = int(max(os.cpu_count()-2, (os.cpu_count()+1)/2))
    with mcf.ProcessPoolExecutor(max_workers=nproc) as executor:
        retdict = { executor.submit(fn):label for fn,label in zip(fns,labels)} 
        for future in mcf.as_completed(retdict):
            res[retdict[future]] = future.result()

    return res
    
def partialize_fnstring(fnstring, fndict):
    fnstring = fnstring.strip('`')
    fns = fnstring.split(',')
    fn = fndict[fns[0]]
    args = fns[1:]
    def partialized(mktstate, runledger, **params):
        return fn(mktstate, runledger, *args, **params)
    return partialized 

def connection_context(target_database='default', **kwargs):
    if target_database == 'default':
        dbhost='to-be-determined'
        dbport=3306
        dbuser='user'
        dbpassword='password'
    else:
        raise NotImplementedError
    return contextlib.closing(pymysql.connect(host=dbhost, port=dbport, user=dbuser,
                                              password=dbpassword))

def dbquery(sql, target):
    with connection_context(target) as conn:
        df = pd.read_sql_query(sql, conn)
    return df

def sql_list(series, numeric=False):
    series = pd.Series(series)
    if numeric:
        ls = [f"{s}" for s in series.to_list()]
    else:
        ls = [f"'{s}'" for s in series.to_list()]
    return ','.join(ls)

def prev_date(day, days, steps=1):
    ser = pd.DataFrame(index=days, data=days)
    shifted = ser.shift(steps).reindex(index=[day], method='bfill').iloc[0,0]
    return shifted

def is_array_like(var):
    return isinstance(var, (Sequence, np.ndarray,pd.Series)) and not isinstance(var, str)

def ensure_array_like(var):
    if is_array_like(var):
        return var
    else:
        return [var]

def ensure_correctly_sized_arrays(*args):
    inputs = []
    maxlen = 1
    for a in args:
        arr = np.array(ensure_array_like(a))
        inputs.append(arr)
        maxlen = max(maxlen, len(arr))
    for i, inp in enumerate(inputs):
        if len(inp) == 1:
            inputs[i] = np.hstack([inp]*maxlen)
        elif len(inp) == maxlen:
            pass
        else:
            raise ParameterError('arguments must be same length or singles')
    return tuple(inputs)

class Persistable():
    # at init time, the directory and the naming convention get fixed
    # persist_dir comes from tha params
    # at persist time a runid (subdirectory name) and a
    # tag (completes the naming convention) is specified
    def __init__(self, naming_convention=None, **params):
        self.persist_dir = params.get('persist_dir','./')
        if naming_convention is not None:
            self.naming_convention = naming_convention
        else:
            self.naming_convention = 'what_is_this.pkl'
        self.persist_path = None
        self.params = params
        
    def _fname(self, tag):
        if tag is None:
            return self.naming_convention
        else:
            return self.naming_convention.format(tag).replace(' ','_')

    @classmethod
    def format_path(cls, persist_dir, tag):
        # this only works if there is a class attribute naming_convention
        if tag is None:
            fname = cls.naming_convention
        else:
            fname = cls.naming_convention.format(tag).replace(' ','_')

        path = '/'.join([persist_dir, fname])
        return path
        
    def _fdir(self):
        
        return self.persist_dir
    
    def persist(self, fname=None, tag=None):
        if self.params.get('do_persist_pkl', False) == False:
            return
        if fname is not None: # overwrite convention
            path = '/'.join([self._fdir(), fname])
        else:
            path = '/'.join([self._fdir(), self._fname(tag)])
        # remember where self was saved, for later reporting
        self.persist_path = path
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.debug(f'writing {path}')
        pickle.dump(self,open(path,'wb'))
        return

    @classmethod
    def load_persisted(cls, fn):
        fns = sorted(glob.glob(fn))
        if len(fns) == 0:
            return None
        if len(fns) == 1:
            return pickle.load(open(fns[0],'rb'))
        if len(fns) > 1:
            return [pickle.load(open(f,'rb')) for f in fns]

    @classmethod
    def load_persisted_by_tag(cls, persist_dir, tag):
        fn = cls.format_path(persist_dir, tag)
        return pickle.load(open(fn,'rb'))

    def persist_df_to_csv(self, df, fname):
        if self.params.get('do_persist_csv', False) == False:
            return
        path = '/'.join([self._fdir(), fname])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.debug(f'writing {path}')
        df.to_csv(path)
        return

# timing
import time
from contextlib import contextmanager

@contextmanager
def timer(name, logger=None, collector_dict=None):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        duration = end - start
        if collector_dict is not None:
            if name in collector_dict:
                collector_dict[name] += duration
            else:
                collector_dict[name] = duration
        msg = f"[TIMER] {name}: {duration:.4f} s"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
    
# memory logging
import time, os, psutil, tracemalloc

process = psutil.Process(os.getpid())
tracemalloc.start()

def log_process_memory():
    res = ""
    mem = process.memory_info().rss / 1024**2
    res = res + f"[MEMORY] Memory usage: {mem:.0f} MB\n"

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")
    # res = res + "[MEMORY] Top allocations:\n"
    # for stat in top_stats[:3]:
    #     res = res + f'\t{stat}\n'
    return res

from pympler import asizeof
def compute_df_memory_mb(df):
    return df.memory_usage(deep=True).sum() / 1024**2
def compute_object_memory_mb(obj):
    return asizeof.asizeof(obj) / 1024**2
def log_df_memory(df, name):
    return f'[MEMORY] df {name}: {compute_df_memory_mb(df):.0f} MB'
def log_object_memory(df, name):
    return f'[MEMORY] obj {name}: {compute_object_memory_mb(df):.0f} MB'

