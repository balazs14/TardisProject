import pandas as pd
import numpy as np
from pandas.core.base import PandasObject
from . import utils

# def cleanup_index_columns(df):
#     '''remove columns that are named the same as index levels'''
#     index_levels = set([level for level in df.index.names if level is not None])
#     if isinstance (df, pd.Series):
#         columns = set([df.name]) if df.name is not None else set([])
#     elif isinstance(df, pd.DataFrame):
#         columns = set(df.columns)
#     extra_columns = set.intersection(index_levels, columns)
#     if not extra_columns:
#         return df
#     else:
#         if isinstance (df, pd.Series):
#             return df
#         else:
#             if 'index' in df.columns:
#                 extra_columns = list(extra_columns) + ['index']
#             return df.drop(extra_columns, axis=1)

# def complete_index_columns(df):
#     if set(df.index.names).issubset(set(df.columns)):
#         return df
#     if isinstance(df.index, pd.MultiIndex):
#         lev = df.index.levels[-1].name
#         if lev in df.columns:
#             return complete_index_columns(df.reset_index(lev, drop=True))
#         else:
#             return complete_index_columns(df.reset_index(lev, drop=False))
#     else:
#         # just a simple index
#         lev = df.index.name
#         if lev in df.columns:
#             return df.reset_index(drop=True)
#         elif lev : # not empty
#             return df.reset_index(drop=False)
#         else:
#             return df.reset_index(drop=True)

# def reindex(df_in, levels=[], sort_index=False, copy=False, keep_columns=False):
#     if keep_columns:
#         return reindex_keep_columns(df_in, levels=levels, sort_index=sort_index, copy=copy)
#     if copy:
#         df = df_in.copy()
#     else:
#         df = df_in
#     df = cleanup_index_columns(df)
#     if not levels:
#         return df.reset_index()
#     else:
#         df = df.reset_index(allow_duplicates=True).set_index(levels)
#         if sort_index:
#             return df.sort_index()
#         else:
#             return df

# def reindex_keep_columns(df_in, levels=[], sort_index=False, copy=False):
#     if copy:
#         df =  df_in.copy()
#     else:
#         df = df_in

#     if not isinstance(levels, (list,tuple)):
#         levels = [levels]

#     df = complete_index_columns(df)

#     # shortcut for performance
#     if df.index.names != levels:
#         # we make sure index is also available as a column
#         if len(levels) == 0:
#             if sort_index:
#                 return df.reset_index(drop=True).sort_index()
#             else:
#                 return df.reset_index(drop=True)

#         if len(levels) == 1:
#             df.index = pd.MultiIndex.from_frame(df[levels])
#             if levels[0] in df.columns:
#                 df = df.reset_index(drop=True)
#             df.index = df[levels[0]]

#         if len(levels) > 1:
#             df.index = pd.MultiIndex.from_frame(df[levels])
#     else:
#         if len(levels) == 1 and isinstance(df.index, pd.MultiIndex):
#             df.index = pd.Index(df.index.to_frame()[levels[0]])


#     if sort_index:
#         return df.sort_index()
#     else:
#         return df

# def merge_assign(dfa, dfb, levels, col, fillna, how='outer'):
#     left = reindex(dfa)
#     if col in left:
#         left.drop(columns=[col], inplace=True)
#     right = reindex(dfb)[levels + [col]]
#     res = left.merge(right, on=levels, how=how)
#     res[col].fillna(fillna, inplace=True)
#     return reindex(res, levels)



def stringify_index(index):
    def _stringify(tup):
        if not isinstance(tup,(tuple,list)):
            return str(tup)
        tup2 = [str(x) for x in tup]
        return '_'.join(tup2).replace(' ','_')
    if isinstance(index, pd.Index):
        return [_stringify(x) for x in index.values]
    else:
        raise ValueError('index object expected')

def is_good_friday(date):
    from dateutil.easter import easter
    good_fri_date = pd.Timestamp(easter(date.year))-pd.Timedelta('2D')
    return date == good_fri_date

def is_christmas_day(date):
    return date.month == 12 and date.day == 25

def is_newyears_day(date):
    return date.month == 1 and date.day == 1

def is_weekend(date):
    return date.weekday() == 5 or date.weekday() == 6

def is_holiday(date):
    return is_good_friday(date) or is_newyears_day(date) \
        or is_weekend(date) or is_christmas_day(date)

def is_bday(date):
    return is_holiday(date) == False

def date_plus_offset(date, offset, businessday=False, round_to_prev=True):
    oneday = pd.Timedelta('1D')
    sign = 1 if offset >= 0 else -1
    date = pd.Timestamp(date)
    if businessday:
        while is_holiday(date):
            if round_to_prev or offset<0:
                date = date - oneday
            elif round_to_prev==False or offset>0:
                date = date + oneday
        cnt = abs(offset)
        while cnt:
            date = date + sign * oneday
            if not is_holiday(date):
                cnt = cnt - 1
        return date
    else:
        return date + pd.Timedelta(f'{offset}D')

def prev_bday(date, offset=0):
    assert offset <= 0
    return date_plus_offset(date, offset, businessday=True, round_to_prev=True)

def next_bday(date, offset=0):
    assert offset >= 0
    return date_plus_offset(date, offset, businessday=True, round_to_prev=False)

def timestampize(df, cols=None):
    if isinstance(df, pd.Series):
        df = pd.to_datetime(df)
    else:
        if not isinstance(cols, list):
            cols = [cols]
        for col in cols:
            df[col] = pd.to_datetime(df[col])
    return df
PandasObject.timestampize = timestampize

def floatize(df, cols=None):
    if isinstance(df, pd.Series):
        df = df.astype(float)
    else:
        if not isinstance(cols, list):
            cols = [cols]
        for col in cols:
            df[col] = df[col].astype(float)
    return df
PandasObject.floatize = floatize

# def join_update_old(df1, ser, on):
#     column = ser.name
#     df = df1.drop(columns=[column], errors='ignore')
#     df = df.join(ser, on=on)
#     return df
#     # df = join_update(df, multiplier_inferred.rename('multiplier'),  on=['ref_symbol','sectype']).set_index('symbol').multiplier


def join_update(df1, ser, on=None):
    # copies df...
    column = ser.name
    df = df1.drop(columns=[column], errors='ignore')
    if isinstance(ser.index, pd.MultiIndex):
        if on is None:
            on = ser.index.names
        elif on!=ser.index.names :
            ser.index.names = on
    else:
        if on is None:
            on = ser.index.name
        elif on!=ser.index.name :
            ser.index.name = on
    try:
        # deal with the case where there is a column named the same as the index
        cols_to_rename = {col:f'{col}_tmp' for col in df.columns if col in df.index.names}
        cols_to_rename_back = {f'{col}_tmp':col for col in df.columns if col in df.index.names}
        df = df.rename(columns=cols_to_rename)
        df = df.join(ser, on=on)
        df = df.rename(columns=cols_to_rename_back)
    except:
        # if we have a multiindex on ser, we need to get rid of the degenerate indices

        import pdb; pdb.set_trace()
        if isinstance(ser.index, pd.MultiIndex):
            assert (on in ser.index.names)
        else:
            assert (on == ser.index.name)
        ser = ser.reset_index().set_index(on)[column]
        df = df.join(ser, on=on)

    return df

PandasObject.join_update = join_update

def map_update(df, src_df, on=None, col=None, target_col=None, prefill=None):
    '''
    For each column in src_df, join on "on" to update (inplace)
    or extend the columns in df. Only those rows are updated that
    are present in src_df.

    Importantly, the update happens in place, with minimal copying.

    params:
         - on: single or multiple index columns to join on
               if None, use the index or src_df to join on.
         - col: if None, col = src_df.columns (or series name if src is series)
         - target_col: used to rename the cols. by default don't rename
         - copy: if True, don't do it in place, instead return a copy of the updated
                 df. This is important wher df is a slice.
    '''

    if isinstance(src_df, pd.Series):
        src_df = src_df.to_frame()

    # deal with default cases for col
    if col is None:
        col = list(set(src_df.columns)-set(utils.ensure_array_like(on)))

    if not is_iterable(col):
        col = [col]

    if target_col is None:
        target_col = col
    else:
        if not is_iterable(target_col):
            target_col = [target_col]
    assert len(target_col) == len(col)

    if prefill is None:
        prefill = ['ignore'] * len(target_col)
    if not is_iterable(prefill):
        prefill = [prefill]
    assert len(prefill) == len(target_col)

    # deal with default case for on
    if on is None:
        if isinstance(src_df.index, pd.MultiIndex):
            on = src_df.index.names
        else:
            on = src_df.index.name

    # shortcut if src_df is empty:
    if src_df.empty:
        for tc,pf in zip(target_col,prefill):
            if tc not in df.columns:
                df[tc] = pf
        return df

    # add temporary columns to df and src to make "map" work
    if not isinstance(on, list):
        # if on is not a list, then either on is the index on src_df
        # or one of the columns
        if not isinstance(src_df.index, pd.MultiIndex) \
           and on == src_df.index.name:
            src_already_indexed = True
            tmp_src_df = src_df
        else:
            assert on in src_df.columns
            if isinstance(src_df.index, pd.RangeIndex):
                tmp_src_df = src_df.set_index(on)
            else:
                # this might copy, but I need indexing for the map
                tmp_src_df = src_df.reset_index().set_index(on)

        if not isinstance(df.index, pd.MultiIndex) \
           and on == df.index.name:
            df['tmp_on'] = df.index
        else:
            if on in df.columns:
                df['tmp_on'] = df[on]
            else:
                assert on in df.index.names
                df['tmp_on'] = df.index.to_frame()[on]
    else:
        # if on is a list
        # assert that the either on is the same as the src_df index or
        # all of "on" is in the columns
        if isinstance(src_df.index, pd.MultiIndex) \
           and on==src_df.index.names:
            src_df['tmp_on'] = src_df.index
        else:
            assert set(on).issubset(src_df.columns)
            src_df['tmp_on'] = list(zip(*[src_df[c] for c in on]))

        if isinstance(src_df.index, pd.RangeIndex):
            tmp_src_df = src_df.set_index('tmp_on')
        else:
            # this might copy, but I need indexing for the map
            tmp_src_df = src_df.reset_index().set_index('tmp_on')

        if isinstance(df.index, pd.MultiIndex) \
           and on==df.index.names:
            df['tmp_on'] = df.index
        else:
            from_columns = set(on).intersection(df.columns)
            from_index = set(on).intersection(df.index.names)
            assert set(on).issubset(from_columns.union(from_index))
            clist = list(from_columns)
            ilist = list(from_index)
            tmp_df = pd.concat([df.index.to_frame()[ilist], df[clist]], axis=1)
            df['tmp_on'] = list(zip(*[tmp_df[c] for c in on]))

    tmp_src_df['tmp_present'] = True
    rows_to_overwrite = df['tmp_on'].map(tmp_src_df['tmp_present'])
    rows_to_overwrite = rows_to_overwrite.fillna(False)
    for co, tc, fi in zip(col, target_col, prefill):
        # if co is not in tmp_src_df, it means it is in "on"
        # Therefor no need to copy it
        if co not in tmp_src_df.columns: continue

        # only overwrite rows in tc for which there is something
        # in src.
        if fi != 'ignore':
            df[tc] = fi
        tgt = df['tmp_on'].map(tmp_src_df[co])
        df.loc[rows_to_overwrite, tc] = tgt.loc[rows_to_overwrite]

    df.drop(columns=['tmp_on'], errors='ignore', inplace=True)
    # when src_df is a slice we might trigger a copy here. I am
    # leaving this in -- the user should make sure that src is not a slice
    # -- after all I remove it, too annoying.
    #src_df.drop(columns=['tmp_on','tmp_present'], errors='ignore', inplace=True)
    return df # works inplace, this is just for convenience

PandasObject.map_update = map_update

def amounts_formatter(val, dummy=None):
    sign = '' if val>=0 else '-'
    val = abs(val)
    val = int(val)
    res = ''
    while val > 0:
        valk = int(val/1000)
        valn = val - valk * 1000
        val = valk
        res = ',{:03.0f}'.format(valn) + res
    while res.startswith(('0',',')):
        res = res[1:]
    if len(res) == 0:
        res = '--'
    else:
        res = sign + res
    return res

def is_iterable(obj):
    if isinstance(obj, str):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False


import contextlib
@contextlib.contextmanager
def display_context(kind=None):

    def eng_format_sigfigs(x, sig=3):
        if pd.isna(x):
            return "NaN"
        if x == 0:
            return "0"

        # exponent multiple of 3
        exp3 = int(np.floor(np.log10(abs(x)) / 3) * 3)
        scaled = x / (10**exp3)

        # format with sig figs
        s = f"{scaled:.{sig}g}"

        # SI postfixes
        si_prefixes = {
            -24: "y", -21: "z", -18: "a", -15: "f", -12: "p", -9: "n",
            -6: "u", -3: "m", 0: "",
             3: "k", 6: "M", 9: "G", 12: "T", 15: "P", 18: "E", 21: "Z", 24: "Y"
        }
        unit = si_prefixes.get(exp3, f"e{exp3}")
        return f"{s}{unit}"

    if kind is None:
        pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)
        with pd.option_context('display.max_columns',None,
                               'display.width',200,
                               'display.max_colwidth',30):
            yield
        pd.set_option('display.float_format',None)
    elif kind == '3 decimals':
        with pd.option_context('display.max_columns',None,
                               'display.width',200,
                               'display.max_colwidth',30,
                               'display.float_format','{:.3f}'.format):
            yield
    elif kind == 'many rows':
        pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)
        with pd.option_context('display.max_columns',None,
                               'display.width',200,
                               'display.max_rows',500,
                               'display.max_colwidth',30):
            yield
        pd.set_option('display.float_format',None)
    elif kind == 'wide columns':
        with pd.option_context('display.max_columns',None,
                               'display.width',200,
                               'display.max_rows',50,
                               'display.max_colwidth',50):
            yield
    elif kind == 'narrow columns':
        with pd.option_context('display.max_columns',None,
                               'display.width',200,
                               'display.max_rows',50,
                               'display.max_colwidth',11,
                               'display.float_format', lambda x: eng_format_sigfigs(x, sig=3)):
            yield
    else:
        raise utils.ParameterError

