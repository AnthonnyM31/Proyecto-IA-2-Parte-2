"""Modelos de línea base (Persistencia, Perfil Condicional y SARIMA)."""

import time
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import DEFAULT_CONFIG, ProjectConfig
from src.data_pipeline import SplitTensors, inverse_scale_targets


def predict_persistence_24h(
    split_tensors: SplitTensors,
    target_scaler: MinMaxScaler,
    horizon: int = 24,
) -> np.ndarray:
    """
    Línea base de persistencia estacional diaria (24h):
    usa las últimas 24 horas conocidas [t-23 .. t] como pronóstico de [t+1 .. t+24].
    """
    last_24h_scaled = split_tensors.X[:, -horizon:, 0]
    return inverse_scale_targets(last_24h_scaled, target_scaler)


def predict_persistence_168h(
    split_df: pd.DataFrame,
    target_datetimes: np.ndarray,
    cfg: ProjectConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Línea base de persistencia semanal (168h):
    y_pred(t + h) = y_real(t + h - 168 horas) dentro del mismo split.
    """
    lookup = split_df.set_index(cfg.datetime_col)[cfg.target_col]
    dt_flat = pd.to_datetime(np.asarray(target_datetimes).ravel())
    lagged_dt = dt_flat - pd.Timedelta(hours=cfg.weekly_lag_hours)
    preds = lookup.reindex(lagged_dt).to_numpy(dtype=np.float64)
    if np.isnan(preds).any():
        raise ValueError("Existen timestamps sin historia previa de 168h dentro del split.")
    return preds.reshape(target_datetimes.shape)


def fit_conditional_hourly_profile(
    train_df: pd.DataFrame,
    cfg: ProjectConfig = DEFAULT_CONFIG,
) -> Tuple[Dict[Tuple[int, int], float], float]:
    """Calcula perfil horario promedio condicional (fin de semana vs. laboral) solo en train."""
    dt = train_df[cfg.datetime_col]
    is_weekend = (dt.dt.day_of_week >= 5).astype(int)
    hours = dt.dt.hour
    grouped = train_df.groupby([is_weekend, hours])[cfg.target_col].mean()
    profile = {(int(k[0]), int(k[1])): float(v) for k, v in grouped.items()}
    fallback = float(train_df[cfg.target_col].mean())
    return profile, fallback


def predict_conditional_hourly_profile(
    profile: Dict[Tuple[int, int], float],
    fallback: float,
    target_datetimes: np.ndarray,
) -> np.ndarray:
    """Predicción vectorizada por tipo de día y hora sobre ventanas [N, H]."""
    dt_flat = pd.to_datetime(np.asarray(target_datetimes).ravel())
    is_wknd = (dt_flat.day_of_week >= 5).astype(int)
    hours = dt_flat.hour
    preds = np.array(
        [profile.get((int(w), int(h)), fallback) for w, h in zip(is_wknd, hours)],
        dtype=np.float64,
    )
    return preds.reshape(target_datetimes.shape)


def run_stationarity_diagnostics(
    train_series: pd.Series,
    max_samples: int = 24 * 365,
) -> Dict[str, float]:
    """Ejecuta pruebas ADF y KPSS sobre la cola reciente de entrenamiento para justificar SARIMA."""
    sub = train_series.iloc[-max_samples:].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, _, _, _, _ = adfuller(sub, maxlag=48, autolag="AIC")
        kpss_stat, kpss_p, _, _ = kpss(sub, regression="c", nlags="auto")
        # Diferenciación estacional real de período s=24: (1 - B^24) y_t
        diff24 = sub[24:] - sub[:-24]
        adf_d24_stat, adf_d24_p, _, _, _, _ = adfuller(diff24, maxlag=48, autolag="AIC")

    return {
        "n_samples": float(len(sub)),
        "adf_stat_raw": float(adf_stat),
        "adf_pvalue_raw": float(adf_p),
        "kpss_stat_raw": float(kpss_stat),
        "kpss_pvalue_raw": float(kpss_p),
        "adf_stat_diff24": float(adf_d24_stat),
        "adf_pvalue_diff24": float(adf_d24_p),
    }


def select_sarima_eval_indices(
    val_tensors: SplitTensors,
    n_origins: int = 28,
) -> np.ndarray:
    """
    Selecciona índices de evaluación estratificados cronológicamente a lo largo de `val`
    en horas 00:00 (pronóstico Day-Ahead completo de 24h) para comparar SARIMA de forma justa.
    """
    origins = pd.to_datetime(val_tensors.origin_datetimes)
    # Origen en hora 23:00 para que el horizonte [t+1..t+24] cubra exactamente un día 00:00..23:00
    day_end_mask = np.where(origins.hour == 23)[0]
    if len(day_end_mask) < n_origins:
        return day_end_mask
    pick_pos = np.linspace(0, len(day_end_mask) - 1, num=n_origins, dtype=int)
    return day_end_mask[pick_pos]


def evaluate_sarima_on_indices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_tensors: SplitTensors,
    eval_indices: Sequence[int],
    cfg: ProjectConfig = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evalúa SARIMA(p,d,q)x(P,D,Q)_24 de forma estrictamente causal sobre orígenes seleccionados.
    Para cada origen t_i, ajusta el modelo sobre las últimas `sarima_fit_days` de historia previa
    (combinando el tramo final de train si es necesario para los primeros días de val) y pronostica 24h.
    """
    history_df = (
        pd.concat([train_df, val_df], axis=0, ignore_index=True)
        .set_index(cfg.datetime_col)[cfg.target_col]
        .asfreq("h")
    )

    fit_hours = cfg.sarima_fit_days * 24
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    fit_times: List[float] = []
    infer_times_ms: List[float] = []
    aics: List[float] = []
    failures = 0

    for idx in eval_indices:
        origin_ts = pd.Timestamp(val_tensors.origin_datetimes[idx])
        hist_end = origin_ts
        hist_start = origin_ts - pd.Timedelta(hours=fit_hours - 1)
        endog = history_df.loc[hist_start:hist_end]

        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    endog,
                    order=cfg.sarima_order,
                    seasonal_order=cfg.sarima_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=100)
                t_fit = time.perf_counter() - t0

                t_inf0 = time.perf_counter()
                fc = fitted.forecast(steps=cfg.horizon).to_numpy(dtype=np.float64)
                t_inf_ms = (time.perf_counter() - t_inf0) * 1000.0

                aics.append(float(fitted.aic))
        except Exception:
            failures += 1
            t_fit = time.perf_counter() - t0
            t_inf_ms = 0.0
            fc = endog.iloc[-cfg.horizon :].to_numpy(dtype=np.float64)

        fit_times.append(t_fit)
        infer_times_ms.append(t_inf_ms)
        preds.append(fc)
        trues.append(val_tensors.y_raw[idx])

    meta = {
        "n_origins": float(len(eval_indices)),
        "fit_days": float(cfg.sarima_fit_days),
        "mean_fit_time_s": float(np.mean(fit_times)),
        "total_eval_time_s": float(np.sum(fit_times)),
        "mean_infer_latency_ms": float(np.mean(infer_times_ms)),
        "mean_aic": float(np.mean(aics)) if aics else float("nan"),
        "convergence_fallbacks": float(failures),
    }
    return np.vstack(trues), np.vstack(preds), meta
