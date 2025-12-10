# =========================================
# XGB Hyperparameter Tuning (Time-series correct) — PARALLEL VERSION
# - Data: NB FX panel 1980–1999 (daily, multiple FX series)
# - Target for tuning: panel FX series (univariate per series)
# - Split:
#     Train (inner tuning): 1980-12-10 → 1995-12-31
#     Validation (outer eval): 1996-01-01 → 1999-12-31
# - Inner loop:
#     Expanding monthly walk-forward on train
#     For each month m:
#         cut = last business day of previous month
#         fit one-step daily XGB on history up to cut
#         recursive daily forecast for next month
#         aggregate to business-day monthly mean
# - Objective: minimize average RMSE across FX series
# - Parallelization:
#     (A) Candidates in outer loop (process-level)
#     (B) Series inside inner scoring (optional)
#     (C) Months inside walk-forward (optional)
#   Guards prevent nested oversubscription by auto-disabling (B,C)
#   when (A) uses multiple jobs.
# =========================================

from __future__ import annotations
import io, time, random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import requests, certifi
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed


# -----------------------------
# Configuration
# -----------------------------
FINETUNE_FX_COLS = [
    "AUD", "CAD", "CHF", "DKK", "GBP",
    "ISK", "JPY", "NZD", "SEK", "USD", "XDR",
]

NB_PANEL_URL = (
    "https://raw.githubusercontent.com/bredeespelid/"
    "Data_MasterOppgave/refs/heads/main/FineTuneData/NB1980-1999.csv"
)

@dataclass
class Cfg:
    # Walk-forward settings
    m_freq: str = "M"
    max_lags: int = 20
    max_horizon: int = 64
    min_hist_days: int = 40

    # Parallel settings
    n_jobs_outer: int = -1     # parallel over parameter candidates
    n_jobs_series: int = 1     # parallel over FX series inside scoring
    n_jobs_months: int = 1     # parallel over months inside walk-forward
    backend_outer: str = "loky"
    backend_inner: str = "loky"

    # Repro + verbosity
    random_state: int = 42
    verbose: bool = True

    # Inner WF tuning window (within train)
    inner_start_period: str = "1985-01"
    inner_end_period: str = "1995-12"

    # Output
    fig_png: str = "XGB_Tuned_OuterVal_FXPanel.png"

CFG = Cfg()


# -----------------------------
# Download + load NB FX panel
# -----------------------------
def download_csv_text(url: str, retries: int = 3, timeout: int = 60) -> str:
    """Download CSV with basic retry/backoff."""
    last_err = None
    for k in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, verify=certifi.where())
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if k < retries:
                time.sleep(1.5 * k)
    raise RuntimeError(f"Download failed: {last_err}")


def load_nb_fx_panel(url: str) -> pd.DataFrame:
    """Load NB panel, coerce numeric, build daily calendar index, ffill."""
    text = download_csv_text(url)
    raw = pd.read_csv(
        io.StringIO(text),
        sep=";",
        decimal=".",
        encoding="utf-8-sig",
    )
    required = ["ds"] + FINETUNE_FX_COLS
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in NB panel CSV: {missing}")

    df = (
        raw[required]
        .rename(columns={"ds": "DATE"})
        .assign(DATE=lambda x: pd.to_datetime(x["DATE"], dayfirst=True, errors="coerce"))
        .dropna(subset=["DATE"])
        .sort_values("DATE")
        .set_index("DATE")
    )
    for c in FINETUNE_FX_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all", subset=FINETUNE_FX_COLS)

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df_d = df.reindex(full_idx).ffill()
    df_d.index.name = "DATE"
    return df_d


# -----------------------------
# Parameter sanitation
# -----------------------------
def sanitize_params(p: Dict) -> Dict:
    """Ensure correct dtypes for XGBoost."""
    p = p.copy()
    if "n_estimators" in p:
        p["n_estimators"] = int(round(p["n_estimators"]))
    if "max_depth" in p:
        p["max_depth"] = int(round(p["max_depth"]))
    if "min_child_weight" in p:
        p["min_child_weight"] = float(p["min_child_weight"])
    if "learning_rate" in p:
        p["learning_rate"] = float(p["learning_rate"])
    if "subsample" in p:
        p["subsample"] = float(p["subsample"])
    if "colsample_bytree" in p:
        p["colsample_bytree"] = float(p["colsample_bytree"])
    if "gamma" in p:
        p["gamma"] = float(p["gamma"])
    if "reg_alpha" in p:
        p["reg_alpha"] = float(p["reg_alpha"])
    if "reg_lambda" in p:
        p["reg_lambda"] = float(p["reg_lambda"])
    return p


# -----------------------------
# Business-day utilities
# -----------------------------
def to_business_series(S_d: pd.Series) -> pd.Series:
    """Convert daily series to business-day series with forward fill."""
    return S_d.asfreq("B").ffill()


def last_trading_day(S_b: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the last business day in [start, end]."""
    sl = S_b.loc[start:end]
    return sl.index[-1] if not sl.empty else None


# -----------------------------
# Feature engineering: univariate lag matrix
# -----------------------------
def make_lag_matrix_univariate(
    s_hist: pd.Series,
    max_lags: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, y for one-step ahead univariate forecasting.

    X_t = [y_{t-1}, ..., y_{t-L}]
    y_t = y_t
    """
    work = pd.DataFrame({"y": s_hist.astype(float)})
    for k in range(1, max_lags + 1):
        work[f"lag{k}"] = work["y"].shift(k)
    work = work.dropna()
    X = work[[f"lag{k}" for k in range(1, max_lags + 1)]].values
    y = work["y"].values
    return X, y


# -----------------------------
# XGB one-step model (univariate)
# -----------------------------
def fit_xgb_one_step_univariate(
    s_hist_daily: pd.Series,
    params: Dict
) -> XGBRegressor:
    """Fit XGB regressor to predict next-day level from own lags."""
    params = sanitize_params(params)

    X, y = make_lag_matrix_univariate(s_hist_daily, CFG.max_lags)
    if len(X) < 5:
        raise ValueError("Too few observations after lagging.")

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=1,  # always 1 to control nested parallelism
        random_state=CFG.random_state,
        **params
    )
    model.fit(X, y)
    return model


def recursive_daily_forecast_univariate(
    model: XGBRegressor,
    s_hist_daily: pd.Series,
    H: int
) -> pd.Series:
    """
    Produce H daily forecasts recursively.

    The series is updated with predicted values after each step.
    """
    hist = s_hist_daily.astype(float).copy()
    preds = []

    for _ in range(H):
        if len(hist) < CFG.max_lags:
            raise ValueError("Not enough history for lags.")

        lags = hist.iloc[-CFG.max_lags:][::-1].values
        x = lags.reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)

        new_idx = hist.index[-1] + pd.Timedelta(days=1)
        hist = pd.concat([hist, pd.Series([yhat], index=[new_idx])])

    return pd.Series(preds)


# -----------------------------
# Monthly walk-forward for one series (optional parallel over months)
# -----------------------------
def walk_forward_monthly_one_series(
    S_d: pd.Series,
    params: Dict,
    start_period: Optional[pd.Period] = None,
    end_period: Optional[pd.Period] = None,
    n_jobs_months: int = 1,
    backend: str = "loky"
) -> pd.DataFrame:
    """
    Monthly walk-forward:
      - cut at last business day of previous month
      - fit one-step model on daily history up to cut
      - recursive daily forecast for next month
      - aggregate forecast to business-day monthly mean
    """
    params = sanitize_params(params)
    S_b = to_business_series(S_d)

    first_m = pd.Period(S_b.index.min(), freq=CFG.m_freq)
    last_m  = pd.Period(S_b.index.max(), freq=CFG.m_freq)
    if start_period is not None:
        first_m = max(first_m, start_period)
    if end_period is not None:
        last_m = min(last_m, end_period)

    months = pd.period_range(first_m, last_m, freq=CFG.m_freq)

    def _process_one_month(m: pd.Period):
        prev_m = m - 1
        m_start, m_end = m.start_time, m.end_time
        prev_start, prev_end = prev_m.start_time, prev_m.end_time

        cut = last_trading_day(S_b, prev_start, prev_end)
        if cut is None:
            return None

        hist_d = S_d.loc[:cut]
        if hist_d.size < CFG.min_hist_days:
            return None
        if hist_d.size <= CFG.max_lags:
            return None

        idx_m_b = S_b.index[(S_b.index >= m_start) & (S_b.index <= m_end)]
        if idx_m_b.size < 1:
            return None
        y_true = float(S_b.loc[idx_m_b].mean())

        H = (m_end.date() - m_start.date()).days + 1
        if H <= 0 or H > CFG.max_horizon:
            return None

        model = fit_xgb_one_step_univariate(hist_d, params)
        pf = recursive_daily_forecast_univariate(model, hist_d, H)

        f_idx = pd.date_range(cut + pd.Timedelta(days=1), periods=H, freq="D")
        pred_daily = pd.Series(pf.values, index=f_idx)

        pred_b = pred_daily.reindex(idx_m_b, method=None)
        if pred_b.isna().all():
            return None
        y_pred = float(pred_b.dropna().mean())

        return {"month": m, "y_true": y_true, "y_pred": y_pred}

    if n_jobs_months == 1:
        out_rows = [_process_one_month(m) for m in months]
    else:
        out_rows = Parallel(n_jobs=n_jobs_months, backend=backend)(
            delayed(_process_one_month)(m) for m in months
        )

    out_rows = [r for r in out_rows if r is not None]
    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df.set_index("month").sort_index()
    return df


def rmse_mae(eval_df: pd.DataFrame) -> Tuple[float, float]:
    """Compute RMSE and MAE on monthly evaluation frame."""
    core = eval_df.dropna()
    if core.empty:
        return np.nan, np.nan
    err = core["y_true"] - core["y_pred"]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(mean_absolute_error(core["y_true"], core["y_pred"]))
    return rmse, mae


# -----------------------------
# Inner-loop score (optional parallel over series)
# -----------------------------
def score_params_inner(
    params: Dict,
    train_panel_d: pd.DataFrame,
    n_jobs_series: int = 1,
    n_jobs_months: int = 1,
    backend: str = "loky"
) -> float:
    """Average RMSE across FX series using inner walk-forward on training panel."""
    params = sanitize_params(params)

    start_p = pd.Period(CFG.inner_start_period, freq=CFG.m_freq)
    end_p   = pd.Period(CFG.inner_end_period,   freq=CFG.m_freq)

    def _score_one_series(col: str):
        s = train_panel_d[col].dropna()
        if len(s) < CFG.min_hist_days * 2:
            return np.nan
        df_eval = walk_forward_monthly_one_series(
            S_d=s,
            params=params,
            start_period=start_p,
            end_period=end_p,
            n_jobs_months=n_jobs_months,
            backend=backend
        )
        r, _ = rmse_mae(df_eval)
        return r

    if n_jobs_series == 1:
        rmses = [_score_one_series(c) for c in FINETUNE_FX_COLS]
    else:
        rmses = Parallel(n_jobs=n_jobs_series, backend=backend)(
            delayed(_score_one_series)(c) for c in FINETUNE_FX_COLS
        )

    rmses = [r for r in rmses if np.isfinite(r)]
    return float(np.mean(rmses)) if rmses else np.nan


# -----------------------------
# Outer evaluation on validation set
# -----------------------------
def evaluate_outer(
    best_params: Dict,
    val_panel_d: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate best params on outer validation (1996–1999), per series."""
    best_params = sanitize_params(best_params)

    rows = []
    for col in FINETUNE_FX_COLS:
        s = val_panel_d[col].dropna()
        if len(s) < CFG.min_hist_days * 2:
            continue
        df_eval = walk_forward_monthly_one_series(
            S_d=s,
            params=best_params,
            start_period=pd.Period("1996-01", freq="M"),
            end_period=pd.Period("1999-12", freq="M"),
            n_jobs_months=1
        )
        r, m = rmse_mae(df_eval)
        rows.append({"series": col, "val_rmse": r, "val_mae": m})

    return pd.DataFrame(rows).sort_values("val_rmse")


# -----------------------------
# Plot best outer-validation series
# -----------------------------
def plot_best_series(
    val_panel_d: pd.DataFrame,
    best_params: Dict,
    best_series: str,
    png_path: str
):
    """Plot monthly actual vs forecast for the best series on outer validation."""
    best_params = sanitize_params(best_params)

    s = val_panel_d[best_series].dropna()
    df_eval = walk_forward_monthly_one_series(
        S_d=s,
        params=best_params,
        start_period=pd.Period("1996-01", freq="M"),
        end_period=pd.Period("1999-12", freq="M"),
        n_jobs_months=1
    )

    if df_eval.empty:
        print("Nothing to plot.")
        return

    plt.figure(figsize=(10,6))
    x = df_eval.index.to_timestamp()
    plt.plot(x, df_eval["y_true"], color="black", label="Actual (monthly mean, B-days)")
    plt.plot(x, df_eval["y_pred"], color="tab:blue", linestyle="--", label="Forecast (XGB tuned)")
    plt.title(f"Outer validation: {best_series} (1996–1999), tuned XGB")
    plt.xlabel("Month")
    plt.ylabel("FX level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG: {png_path}")


# -----------------------------
# Hyperparameter candidates (random search)
# -----------------------------
def build_param_candidates(n_random: int = 40) -> List[Dict]:
    """Random search over a sensible hyperparameter space with 150-tree cap."""
    rng = random.Random(CFG.random_state)
    candidates = []
    for _ in range(n_random):
        candidates.append({
            "n_estimators": 150,  # int
            "learning_rate": rng.choice([0.02, 0.04, 0.06, 0.08, 0.12, 0.16]),
            "max_depth": rng.choice([2, 3, 4, 5, 6]),  # will be cast to int anyway
            "min_child_weight": rng.choice([1, 2, 3, 5, 7, 9]),
            "subsample": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
            "colsample_bytree": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
            "gamma": rng.choice([0.0, 0.03, 0.05, 0.1, 0.2]),
            "reg_alpha": rng.choice([0.0, 0.05, 0.1, 0.2, 0.5]),
            "reg_lambda": rng.choice([0.5, 1.0, 2.0, 3.0, 5.0]),
        })
    return candidates


# -----------------------------
# Main
# -----------------------------
def main():
    panel_d = load_nb_fx_panel(NB_PANEL_URL)

    val_start = pd.Timestamp("1996-01-01")
    train_panel_d = panel_d.loc[:val_start - pd.Timedelta(days=1)]
    val_panel_d   = panel_d.loc[val_start:]

    if CFG.verbose:
        print(f"NB panel full: {panel_d.index.min().date()} → {panel_d.index.max().date()} | n={len(panel_d)}")
        print(f"Train: {train_panel_d.index.min().date()} → {train_panel_d.index.max().date()} | n={len(train_panel_d)}")
        print(f"Val:   {val_panel_d.index.min().date()} → {val_panel_d.index.max().date()} | n={len(val_panel_d)}")

    candidates = build_param_candidates(n_random=40)
    if CFG.verbose:
        print(f"\nTuning candidates: {len(candidates)}")

    # Guard against nested parallelism:
    # If outer loop uses multiple workers, force inner parallelism off.
    n_jobs_series = CFG.n_jobs_series if (CFG.n_jobs_outer == 1) else 1
    n_jobs_months = CFG.n_jobs_months if (CFG.n_jobs_outer == 1) else 1

    def _score_one(cand):
        cand = sanitize_params(cand)
        r = score_params_inner(
            cand, train_panel_d,
            n_jobs_series=n_jobs_series,
            n_jobs_months=n_jobs_months,
            backend=CFG.backend_inner
        )
        return cand, r

    inner_results = Parallel(n_jobs=CFG.n_jobs_outer, backend=CFG.backend_outer)(
        delayed(_score_one)(c) for c in candidates
    )

    inner_df = pd.DataFrame([
        {**cand, "inner_avg_rmse": score}
        for cand, score in inner_results
        if np.isfinite(score)
    ]).sort_values("inner_avg_rmse")

    if inner_df.empty:
        raise RuntimeError("All inner scores are NaN. Check data / settings.")

    best_params = sanitize_params(inner_df.iloc[0].drop("inner_avg_rmse").to_dict())
    best_rmse   = float(inner_df.iloc[0]["inner_avg_rmse"])

    print("\n=== Best params from INNER walk-forward (1980–1995) ===")
    print(best_params)
    print(f"Inner avg RMSE: {best_rmse:.6f}")

    outer_df = evaluate_outer(best_params, val_panel_d)
    avg_outer_rmse = float(outer_df["val_rmse"].mean())
    avg_outer_mae  = float(outer_df["val_mae"].mean())

    print("\n=== OUTER validation (1996–1999) ===")
    print(outer_df)
    print(f"\nOuter avg RMSE across series: {avg_outer_rmse:.6f}")
    print(f"Outer avg MAE  across series: {avg_outer_mae:.6f}")

    best_series = outer_df.iloc[0]["series"]
    plot_best_series(val_panel_d, best_params, best_series, CFG.fig_png)


if __name__ == "__main__":
    main()

