import pandas as pd
from io import StringIO


def df_from_string(string, sep=r'\s+', **kwargs):
    return pd.read_csv(StringIO(string), sep=sep, **kwargs)

def assert_df_equal(df,df_good):
    if isinstance(df, str):
        str_df = df
    else:
        str_df = df.to_string() + '\n'

    if isinstance(df_good,pd.DataFrame):
        str_good =  df_good.to_string() + '\n'
    else:
        str_good = df_good

    comp_df = str_df.replace(' ','').replace('\n','')
    comp_good = str_good.replace(' ','').replace('\n','')

    message = '\n\nRESULT SHOULD HAVE BEEN: \n'
    message += str_good
    message += '\nBUT IT WAS: \n'
    message += str_df

    if not comp_df==comp_good:
        with open('/tmp/tmp_df','w') as f:
            f.write(str_df)

    assert comp_df==comp_good, message

