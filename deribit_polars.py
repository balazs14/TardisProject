import os
import sys
import logging
import glob
import gzip
import polars as pl
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tardis_dev import datasets
import pandas as pd

# ---------------------------------------------------------
# 1. KONFIGURÁCIÓ
# ---------------------------------------------------------
load_dotenv()
API_KEY = os.environ.get("TARDIS_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DATADIR = 'deribit_data'
os.makedirs(DATADIR, exist_ok=True)

# Polars optimalizáció
GRID_FREQ = "1s"       

# ---------------------------------------------------------
# 2. KONVERZIÓ: PURE POLARS (STREAMING)
# ---------------------------------------------------------
def convert_to_parquet_polars(csv_file, pq_file):
    """
    CSV -> Parquet konverzió tisztán Polars-szal (sink_parquet).
    Memóriakímélő és kezeli a hiányzó árakat (Coalesce logika).
    """
    logger.info(f"Converting {os.path.basename(csv_file)} -> Parquet (Polars Engine)...")

    # 1. Fejléc ellenőrzése (hogy tudjuk, miből gazdálkodhatunk)
    try:
        with gzip.open(csv_file, 'rt') as f:
            header = f.readline().strip().split(',')
    except Exception as e:
        logger.error(f"Cannot read header: {e}")
        return

    # 2. Schema definíció (hogy a számok tényleg számok legyenek)
    # Csak azokat definiáljuk, amik léteznek a fájlban
    dtypes = {
        'symbol': pl.String,
        'timestamp': pl.Int64,
        'local_timestamp': pl.Int64,
    }
    
    possible_floats = [
        'ask_price', 'bid_price', 
        'best_ask_price', 'best_bid_price', 
        'mark_price', 'last_price', 'index_price',
        'ask_amount', 'bid_amount',
        'best_ask_amount', 'best_bid_amount'
    ]
    
    for col in possible_floats:
        if col in header:
            dtypes[col] = pl.Float64

    # 3. Lazy Scan indítása
    q = pl.scan_csv(csv_file, dtypes=dtypes)

    # 4. Kifejezések felépítése (Expressions)
    select_exprs = []

    # --- IDŐBÉLYEG ---
    # Preferáljuk a local_timestamp-et
    if 'local_timestamp' in header:
        select_exprs.append(pl.col('local_timestamp').cast(pl.Datetime(time_unit='us')).alias('ts'))
    elif 'timestamp' in header:
        select_exprs.append(pl.col('timestamp').cast(pl.Datetime(time_unit='us')).alias('ts'))
    else:
        logger.error("No timestamp column found!")
        return

    # --- SZIMBÓLUM ---
    select_exprs.append(pl.col('symbol'))

    # --- ÁRAK (A PANDAS LOGIKA IMPLEMENTÁLÁSA POLARS-BAN) ---
    # Logika: Best Ask -> Mark Price -> Last Price
    # A 'coalesce' az első NEM NULL értéket veszi a listából.
    
    # ASK ÁR
    ask_sources = []
    if 'best_ask_price' in header: ask_sources.append(pl.col('best_ask_price'))
    if 'ask_price' in header: ask_sources.append(pl.col('ask_price'))
    if 'mark_price' in header: ask_sources.append(pl.col('mark_price'))
    if 'last_price' in header: ask_sources.append(pl.col('last_price'))
    
    if ask_sources:
        select_exprs.append(pl.coalesce(ask_sources).alias('ask_price'))
    else:
        # Ha semmilyen ár nincs, akkor NaN
        select_exprs.append(pl.lit(None, dtype=pl.Float64).alias('ask_price'))

    # BID ÁR (Ugyanaz a logika)
    bid_sources = []
    if 'best_bid_price' in header: bid_sources.append(pl.col('best_bid_price'))
    if 'bid_price' in header: bid_sources.append(pl.col('bid_price'))
    if 'mark_price' in header: bid_sources.append(pl.col('mark_price')) # Bid-nek is jó a mark, ha nincs más
    if 'last_price' in header: bid_sources.append(pl.col('last_price'))
    
    if bid_sources:
        select_exprs.append(pl.coalesce(bid_sources).alias('bid_price'))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Float64).alias('bid_price'))

    # --- MENNYISÉGEK ---
    # Ask Amount
    ask_amt_sources = []
    if 'best_ask_amount' in header: ask_amt_sources.append(pl.col('best_ask_amount'))
    elif 'ask_amount' in header: ask_amt_sources.append(pl.col('ask_amount'))
    
    if ask_amt_sources:
        select_exprs.append(pl.coalesce(ask_amt_sources).fill_null(0.0).alias('ask_amount'))
    else:
        select_exprs.append(pl.lit(0.0).alias('ask_amount'))

    # Bid Amount
    bid_amt_sources = []
    if 'best_bid_amount' in header: bid_amt_sources.append(pl.col('best_bid_amount'))
    elif 'bid_amount' in header: bid_amt_sources.append(pl.col('bid_amount'))
    
    if bid_amt_sources:
        select_exprs.append(pl.coalesce(bid_amt_sources).fill_null(0.0).alias('bid_amount'))
    else:
        select_exprs.append(pl.lit(0.0).alias('bid_amount'))

    # 5. Konverzió és Mentés (Streaming)
    # A sink_parquet a memóriát kímélve írja a lemezre a végeredményt
    try:
        q.select(select_exprs).sink_parquet(pq_file)
    except Exception as e:
        logger.error(f"Streaming conversion failed: {e}")

def prepare_data_for_day(date_str):
    target_dt = pd.to_datetime(date_str)
    from_date = target_dt.strftime('%Y-%m-%d %H:%M:%S')
    to_date = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')

    files = {
        'OPTIONS': {'type': 'quotes', 'symbol': 'OPTIONS', 'pq': f"{DATADIR}/deribit_quotes_{date_str}_OPTIONS.parquet"},
        'FUTURES': {'type': 'derivative_ticker', 'symbol': 'FUTURES', 'pq': f"{DATADIR}/deribit_derivative_ticker_{date_str}_FUTURES.parquet"}
    }

    for key, info in files.items():
        if not os.path.exists(info['pq']):
            csv_pattern = f"{DATADIR}/deribit_{info['type']}_{date_str}_{info['symbol']}.csv.gz"
            if not glob.glob(csv_pattern): 
                logger.info(f"Downloading {key} data ({info['type']})...")
                datasets.download(
                    exchange="deribit",
                    data_types=[info['type']], 
                    from_date=from_date,
                    to_date=to_date,
                    symbols=[info['symbol']],
                    download_dir=DATADIR,
                    api_key=API_KEY
                )
            found_csvs = glob.glob(f"{DATADIR}/deribit_{info['type']}*{date_str}*{info['symbol']}*.csv.gz")
            if found_csvs:
                convert_to_parquet_polars(found_csvs[0], info['pq'])
            else:
                logger.error(f"Download failed for {info['symbol']}")
    
    return files['OPTIONS']['pq'], files['FUTURES']['pq']

# ---------------------------------------------------------
# 3. POLARS ELEMZÉS (GYORS ÉS PONTOS)
# ---------------------------------------------------------
def analyze_day(date_str):
    opt_file, fut_file = prepare_data_for_day(date_str)
    
    logger.info("Loading data pointers...")
    lf_opt_raw = pl.scan_parquet(opt_file)
    lf_fut = pl.scan_parquet(fut_file)

    logger.info("Identifying active expiries (Fast Method)...")
    
    # 1. Struktúra felépítése (csak a szimbólumokból)
    unique_symbols = lf_opt_raw.select("symbol").unique().collect()
    
    structure_df = unique_symbols.with_columns([
        pl.col("symbol").str.split("-").list.get(0).alias("underlying"),
        pl.col("symbol").str.split("-").list.get(1).alias("expiry"),
        pl.col("symbol").str.split("-").list.get(2).cast(pl.Float64, strict=False).alias("strike"),
        pl.col("symbol").str.split("-").list.get(3).alias("type"),
    ]).filter(
        pl.col("strike").is_not_null() & pl.col("type").is_not_null()
    ).select(['underlying', 'expiry']).unique()

    # Future előkészítés
    lf_fut = lf_fut.with_columns([
        pl.col("symbol").str.split("-").list.get(0).alias("underlying"),
        pl.col("symbol").str.split("-").list.get(1).alias("expiry"),
    ])

    results = []
    count = 0
    total = len(structure_df)

    for row in structure_df.iter_rows(named=True):
        und = row['underlying']
        exp = row['expiry']
        count += 1
        
        fut_symbol_expected = f"{und}-{exp}"
        prefix = f"{und}-{exp}"
        is_linear = "_USDC" in und
        
        # 1. Future szűrés
        q_fut = lf_fut.filter(pl.col("symbol") == fut_symbol_expected)
        
        # Van-e future adat?
        if q_fut.head(1).collect().is_empty():
            continue

        # 2. Option szűrés (Prefix alapú = Gyors)
        q_opts_raw = lf_opt_raw.filter(pl.col("symbol").str.starts_with(prefix))
        
        # Részletes parse csak a szűrt adaton
        q_opts = q_opts_raw.with_columns([
            pl.col("symbol").str.split("-").list.get(2).cast(pl.Float64, strict=False).alias("strike"),
            pl.col("symbol").str.split("-").list.get(3).alias("type")
        ]).filter(pl.col("strike").is_not_null())

        # Grid építése
        fut_times = q_fut.select([pl.col("ts").min().alias("min"), pl.col("ts").max().alias("max")]).collect()
        t_start, t_end = fut_times["min"][0], fut_times["max"][0]
        if t_start is None: continue

        grid_df = pl.DataFrame({"ts": pl.datetime_range(t_start, t_end, interval=GRID_FREQ, eager=True)}).lazy()
        
        q_fut_clean = q_fut.select([pl.col("ts"), pl.col("ask_price").alias("f_ask"), pl.col("bid_price").alias("f_bid")]).sort("ts")
        grid_with_fut = grid_df.join_asof(q_fut_clean, on="ts", strategy="backward")

        strikes = q_opts.select("strike").unique().collect().get_column("strike").to_list()
        strikes.sort()

        for K in strikes:
            # Opciók szétválogatása
            q_c = q_opts.filter((pl.col("strike") == K) & (pl.col("type") == "C")).select([
                pl.col("ts"), pl.col("ask_price").alias("c_ask"), pl.col("bid_price").alias("c_bid")]).sort("ts")
            q_p = q_opts.filter((pl.col("strike") == K) & (pl.col("type") == "P")).select([
                pl.col("ts"), pl.col("ask_price").alias("p_ask"), pl.col("bid_price").alias("p_bid")]).sort("ts")
            
            # Párosítás
            merged = grid_with_fut.join_asof(q_c, on="ts", strategy="backward").join_asof(q_p, on="ts", strategy="backward")
            
            # PCP Képletek
            diff_fwd = pl.col("c_ask") - pl.col("p_bid")
            diff_bwd = pl.col("c_bid") - pl.col("p_ask")
            
            if is_linear:
                calc_fwd = diff_fwd - (pl.col("f_bid") - pl.lit(K))
                calc_bwd = diff_bwd - (pl.col("f_ask") - pl.lit(K))
            else:
                calc_fwd = (diff_fwd * pl.col("f_bid")) - (pl.col("f_bid") - pl.lit(K))
                calc_bwd = (diff_bwd * pl.col("f_ask")) - (pl.col("f_ask") - pl.lit(K))

            res = merged.select([
                pl.col("ts"),
                pl.lit(und).alias("symbol_base"),
                pl.lit(exp).alias("expiry"),
                pl.lit(K).alias("strike"),
                calc_fwd.alias("pcpb_forward"),
                calc_bwd.alias("pcpb_backward"),
                pl.col("c_ask"), pl.col("c_bid"), pl.col("p_ask"), pl.col("p_bid"), pl.col("f_ask"), pl.col("f_bid")
            ])
            results.append(res)
            
        if count % 10 == 0:
            logger.info(f"Processed {count}/{total} expiries...")

    if results:
        final_lf = pl.concat(results).drop_nulls(subset=["pcpb_forward"])
        output_file = f"{DATADIR}/deribit_pcp_{date_str}.parquet"
        
        # Mentés engine=streaming módban (nagyon fontos a nagy fájloknál)
        try:
            final_lf.collect(engine="streaming").write_parquet(output_file)
        except Exception as e:
            logger.warning(f"Streaming write failed ({e}), falling back to standard...")
            final_lf.collect().write_parquet(output_file)
            
        logger.info(f"Saved: {output_file}")
    else:
        logger.warning("No valid triplets found.")

if __name__ == "__main__":
    day = sys.argv[1] if len(sys.argv) > 1 else "2024-05-01"
    analyze_day(day)