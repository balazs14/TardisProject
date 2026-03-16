import math
import importlib
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl

try:
    _numba = importlib.import_module("numba")
    njit = _numba.njit
    prange = _numba.prange
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional runtime fallback
    _HAS_NUMBA = False
    njit = None
    prange = range


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    inv_sqrt_2pi = 0.3989422804014327
    return inv_sqrt_2pi * math.exp(-0.5 * x * x)


def _inverse_black_price_vega_xs(
    fwd: float,
    strike: float,
    tte: float,
    rate: float,
    vol: float,
    is_call: bool,
    scale: float,
) -> tuple[float, float]:
    if fwd <= 0.0 or strike <= 0.0 or tte <= 0.0 or vol <= 0.0 or scale <= 0.0:
        return math.nan, math.nan

    sqrt_t = math.sqrt(tte)
    vsqrt_t = vol * sqrt_t
    d1 = (math.log(fwd / strike) + 0.5 * vol * vol * tte) / vsqrt_t
    d2 = d1 - vsqrt_t
    discount = math.exp(-rate * tte)
    m = strike / fwd

    if is_call:
        price_coin = discount * (_norm_cdf(d1) - m * _norm_cdf(d2))
    else:
        price_coin = discount * (m * _norm_cdf(-d2) - _norm_cdf(-d1))

    # Coin-margined inverse option premium is in base coin units.
    # The pipeline stores *_price_xS in quote terms, so scale by spot/index.
    price_xs = scale * price_coin
    vega_xs = scale * discount * sqrt_t * _norm_pdf(d1)
    return price_xs, vega_xs


def _inverse_black_bounds_xs(
    fwd: float,
    strike: float,
    tte: float,
    rate: float,
    is_call: bool,
    scale: float,
) -> tuple[float, float]:
    if fwd <= 0.0 or scale <= 0.0:
        return math.nan, math.nan

    discount = math.exp(-rate * tte)
    m = strike / fwd
    if is_call:
        lower_coin, upper_coin = discount * max(1.0 - m, 0.0), discount
    else:
        lower_coin, upper_coin = discount * max(m - 1.0, 0.0), discount * m
    return scale * lower_coin, scale * upper_coin


if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _norm_cdf_nb(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


    @njit(cache=True, fastmath=True)
    def _norm_pdf_nb(x: float) -> float:
        inv_sqrt_2pi = 0.3989422804014327
        return inv_sqrt_2pi * math.exp(-0.5 * x * x)


    @njit(cache=True, fastmath=True)
    def _inverse_black_price_vega_xs_nb(
        fwd: float,
        strike: float,
        tte: float,
        rate: float,
        vol: float,
        is_call: bool,
        scale: float,
    ):
        if fwd <= 0.0 or strike <= 0.0 or tte <= 0.0 or vol <= 0.0 or scale <= 0.0:
            return math.nan, math.nan

        sqrt_t = math.sqrt(tte)
        vsqrt_t = vol * sqrt_t
        d1 = (math.log(fwd / strike) + 0.5 * vol * vol * tte) / vsqrt_t
        d2 = d1 - vsqrt_t
        discount = math.exp(-rate * tte)
        m = strike / fwd

        if is_call:
            price_coin = discount * (_norm_cdf_nb(d1) - m * _norm_cdf_nb(d2))
        else:
            price_coin = discount * (m * _norm_cdf_nb(-d2) - _norm_cdf_nb(-d1))

        price_xs = scale * price_coin
        vega_xs = scale * discount * sqrt_t * _norm_pdf_nb(d1)
        return price_xs, vega_xs


    @njit(cache=True, fastmath=True)
    def _inverse_black_bounds_xs_nb(
        fwd: float,
        strike: float,
        tte: float,
        rate: float,
        is_call: bool,
        scale: float,
    ):
        if fwd <= 0.0 or scale <= 0.0:
            return math.nan, math.nan

        discount = math.exp(-rate * tte)
        m = strike / fwd
        if is_call:
            lower_coin, upper_coin = discount * max(1.0 - m, 0.0), discount
        else:
            lower_coin, upper_coin = discount * max(m - 1.0, 0.0), discount * m
        return scale * lower_coin, scale * upper_coin


    @njit(cache=True, fastmath=True, parallel=True)
    def _implied_vol_newton_vectorized_nb(
        fwd: np.ndarray,
        strike: np.ndarray,
        tte: np.ndarray,
        scale: np.ndarray,
        obs_price: np.ndarray,
        is_call: np.ndarray,
        rate: float,
        init_vol: float,
        tol: float,
        max_iter: int,
        min_vol: float,
        max_vol: float,
    ) -> np.ndarray:
        n = fwd.shape[0]
        out = np.empty(n, dtype=np.float64)
        out[:] = np.nan

        for i in prange(n):
            f = fwd[i]
            k = strike[i]
            t = tte[i]
            s = scale[i]
            p = obs_price[i]
            cflag = is_call[i]

            if not np.isfinite(f) or not np.isfinite(k) or not np.isfinite(t) or not np.isfinite(s) or not np.isfinite(p):
                continue
            if f <= 0.0 or k <= 0.0 or t <= 0.0 or s <= 0.0 or p <= 0.0:
                continue

            lower, upper = _inverse_black_bounds_xs_nb(f, k, t, rate, cflag, s)
            eps = 1e-14
            if p < (lower - eps) or p > (upper + eps):
                continue

            sigma = init_vol
            if sigma < min_vol:
                sigma = min_vol
            if sigma > max_vol:
                sigma = max_vol

            for _ in range(max_iter):
                model, vega = _inverse_black_price_vega_xs_nb(f, k, t, rate, sigma, cflag, s)
                if not np.isfinite(model) or not np.isfinite(vega) or vega <= 1e-14:
                    break

                diff = model - p
                if abs(diff) < tol:
                    out[i] = sigma
                    break

                step = diff / vega
                sigma_new = sigma - step
                if sigma_new < min_vol:
                    sigma_new = min_vol
                elif sigma_new > max_vol:
                    sigma_new = max_vol

                if abs(sigma_new - sigma) < tol:
                    sigma = sigma_new
                    out[i] = sigma
                    break

                sigma = sigma_new

        return out


else:
    def _implied_vol_newton_vectorized_nb(
        fwd: np.ndarray,
        strike: np.ndarray,
        tte: np.ndarray,
        scale: np.ndarray,
        obs_price: np.ndarray,
        is_call: np.ndarray,
        rate: float,
        init_vol: float,
        tol: float,
        max_iter: int,
        min_vol: float,
        max_vol: float,
    ) -> np.ndarray:
        n = fwd.shape[0]
        out = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            f = float(fwd[i])
            k = float(strike[i])
            t = float(tte[i])
            s = float(scale[i])
            p = float(obs_price[i])
            cflag = bool(is_call[i])

            if not (np.isfinite(f) and np.isfinite(k) and np.isfinite(t) and np.isfinite(s) and np.isfinite(p)):
                continue
            if f <= 0.0 or k <= 0.0 or t <= 0.0 or s <= 0.0 or p <= 0.0:
                continue

            lower, upper = _inverse_black_bounds_xs(f, k, t, rate, cflag, s)
            eps = 1e-14
            if p < (lower - eps) or p > (upper + eps):
                continue

            sigma = min(max(init_vol, min_vol), max_vol)
            for _ in range(max_iter):
                model, vega = _inverse_black_price_vega_xs(f, k, t, rate, sigma, cflag, s)
                if not (np.isfinite(model) and np.isfinite(vega)) or vega <= 1e-14:
                    break

                diff = model - p
                if abs(diff) < tol:
                    out[i] = sigma
                    break

                sigma_new = sigma - diff / vega
                sigma_new = min(max(sigma_new, min_vol), max_vol)

                if abs(sigma_new - sigma) < tol:
                    out[i] = sigma_new
                    break

                sigma = sigma_new

        return out


def _ensure_tte_column(df: pl.DataFrame) -> pl.DataFrame:
    if "tte" in df.columns:
        return df
    if "exp" in df.columns and "ts" in df.columns:
        return df.with_columns(((pl.col("exp") - pl.col("ts")) / pl.duration(days=365)).alias("tte"))
    raise ValueError("Input needs either 'tte' or both 'exp' and 'ts' columns")


def _select_price_columns(df: pl.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    default_cols = ["call_bid_price", "call_ask_price", "put_bid_price", "put_ask_price"]
    cols = list(requested) if requested is not None else default_cols
    return [c for c in cols if c in df.columns]


def compute_black_implied_vols(
    aligned_df: pl.DataFrame | pd.DataFrame,
    r: float,
    price_columns: Iterable[str] | None = None,
    fwd_col: str = "fut_mid_price",
    scale_col: str = "index",
    strike_col: str = "strike",
    tte_col: str = "tte",
    output_suffix: str = "_iv_black",
    output_columns: list[str] | None = None,
    init_vol: float = 0.8,
    tol: float = 1e-8,
    max_iter: int = 100,
    min_vol: float = 1e-6,
    max_vol: float = 10.0,
) -> pl.DataFrame | pd.DataFrame:
    """Compute inverse (coin-margined) Black implied vols.

    Parameters
    ----------
    aligned_df
        Aligned options dataframe (Polars or pandas), typically downstream of compute_pcp_metrics.
    r
        Continuously compounded annual risk-free rate.
    price_columns
        Option price columns to invert. Defaults to call/put bid/ask columns that exist.
    fwd_col
        Futures forward/mid column. Defaults to ``fut_mid_price``.
    scale_col
        Price scaling column used only when inverting *_price_xS columns
        (typically spot/index). Defaults to ``index``.
    strike_col
        Strike column name.
    tte_col
        Time-to-expiry in years column. If absent, attempts to compute from exp-ts.
    output_suffix
        Added to each price column name for IV outputs. Ignored when output_columns is given.
    output_columns
        Explicit output column names, one per entry in price_columns. When provided,
        output_suffix is ignored and results are written directly into these columns.
    """
    pandas_interface = isinstance(aligned_df, pd.DataFrame)
    df = pl.from_pandas(aligned_df) if pandas_interface else aligned_df

    if fwd_col not in df.columns:
        if "fut_bid_price" in df.columns and "fut_ask_price" in df.columns:
            df = df.with_columns(((pl.col("fut_bid_price") + pl.col("fut_ask_price")) * 0.5).alias(fwd_col))
        else:
            raise ValueError(f"Missing forward column '{fwd_col}' and cannot derive from fut_bid_price/fut_ask_price")

    df = _ensure_tte_column(df)
    if tte_col != "tte" and "tte" in df.columns and tte_col not in df.columns:
        df = df.rename({"tte": tte_col})

    if strike_col not in df.columns:
        raise ValueError(f"Missing strike column '{strike_col}'")
    if tte_col not in df.columns:
        raise ValueError(f"Missing tte column '{tte_col}'")

    chosen_price_cols = _select_price_columns(df, price_columns)
    if not chosen_price_cols:
        raise ValueError("No requested option price columns found in dataframe")

    needs_scale = any(pcol.endswith("_xS") for pcol in chosen_price_cols)
    if needs_scale and scale_col not in df.columns:
        if "spot_ask_price" in df.columns and "spot_bid_price" in df.columns:
            df = df.with_columns(((pl.col("spot_bid_price") + pl.col("spot_ask_price")) * 0.5).alias(scale_col))
        elif "spot_bid_price" in df.columns:
            df = df.with_columns(pl.col("spot_bid_price").alias(scale_col))
        elif "spot_ask_price" in df.columns:
            df = df.with_columns(pl.col("spot_ask_price").alias(scale_col))
        else:
            raise ValueError(f"Missing scale column '{scale_col}' and cannot derive from spot prices")

    fwd = df.get_column(fwd_col).cast(pl.Float64, strict=False).to_numpy()
    scale_base = (
        df.get_column(scale_col).cast(pl.Float64, strict=False).to_numpy()
        if needs_scale
        else np.ones(df.height, dtype=np.float64)
    )
    strike = df.get_column(strike_col).cast(pl.Float64, strict=False).to_numpy()
    tte = df.get_column(tte_col).cast(pl.Float64, strict=False).to_numpy()

    new_cols = []
    for pcol in chosen_price_cols:
        prices = df.get_column(pcol).cast(pl.Float64, strict=False).to_numpy()
        is_call = np.full(prices.shape[0], pcol.startswith("call_"), dtype=np.bool_)
        scale = scale_base if pcol.endswith("_xS") else np.ones(prices.shape[0], dtype=np.float64)

        iv = _implied_vol_newton_vectorized_nb(
            fwd=fwd,
            strike=strike,
            tte=tte,
            scale=scale,
            obs_price=prices,
            is_call=is_call,
            rate=float(r),
            init_vol=float(init_vol),
            tol=float(tol),
            max_iter=int(max_iter),
            min_vol=float(min_vol),
            max_vol=float(max_vol),
        )
        if output_columns is not None:
            out_name = output_columns[len(new_cols)]
        else:
            out_name = f"{pcol}{output_suffix}"
        new_cols.append(pl.Series(out_name, iv))

    out = df.with_columns(new_cols)
    return out.to_pandas() if pandas_interface else out
