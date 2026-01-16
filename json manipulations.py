import duckdb
con = duckdb.connect()
print(con.execute("SELECT count(*), count(f_ask) FROM 'deribit_data/deribit_pcp_results_2025-12-05_1min.parquet'").df())