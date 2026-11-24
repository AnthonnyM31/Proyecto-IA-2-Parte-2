"""Cálculo estandarizado de métricas de error multihito, análisis de picos y latencia."""

import time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Calcula RMSE, MAE, MAPE (%) y R2 sobre arreglos de demanda en MW reales."""
    y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Dimensiones incompatibles: y_true={y_true_arr.shape}, y_pred={y_pred_arr.shape}"
        )
    if y_true_arr.size == 0:
        raise ValueError("Los arreglos de evaluación no pueden estar vacíos.")

    error = y_true_arr - y_pred_arr
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    denom = np.where(np.abs(y_true_arr) < eps, eps, np.abs(y_true_arr))
    mape = float(np.mean(np.abs(error) / denom) * 100.0)

    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    ss_res = float(np.sum(error ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > eps else 0.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
    }


def compute_horizon_metrics(
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
) -> pd.DataFrame:
    """Calcula RMSE, MAE, MAPE y R2 para cada paso del horizonte h = 1..H."""
    y_t = np.asarray(y_true_2d, dtype=np.float64)
    y_p = np.asarray(y_pred_2d, dtype=np.float64)
    if y_t.ndim != 2 or y_p.ndim != 2 or y_t.shape != y_p.shape:
        raise ValueError(f"Se esperan arreglos 2D iguales [N, H]; recibido {y_t.shape} vs {y_p.shape}")

    rows = []
    for h_idx in range(y_t.shape[1]):
        m = compute_regression_metrics(y_t[:, h_idx], y_p[:, h_idx])
        m["horizon_h"] = h_idx + 1
        rows.append(m)

    cols = ["horizon_h", "RMSE", "MAE", "MAPE", "R2"]
    return pd.DataFrame(rows)[cols]


def compute_regime_metrics(
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
    target_datetimes: np.ndarray,
    peak_threshold_mw: float,
    peak_hours: Sequence[int] = tuple(range(14, 21)),
) -> Dict[str, Dict[str, float]]:
    """
    Desagrega el desempeño según regímenes operativos críticos (PI-2):
    - peak_demand / off_peak_demand (según umbral percentil de entrenamiento en MW)
    - peak_hours_14_20 / off_peak_hours (según franja horaria operativa)
    - weekday / weekend
    """
    y_t = np.asarray(y_true_2d, dtype=np.float64)
    y_p = np.asarray(y_pred_2d, dtype=np.float64)
    dt_flat = pd.to_datetime(np.asarray(target_datetimes).ravel())

    y_t_flat = y_t.ravel()
    y_p_flat = y_p.ravel()

    masks = {
        "Global": np.ones_like(y_t_flat, dtype=bool),
        "Pico_Demanda_P90": y_t_flat >= peak_threshold_mw,
        "Valle_Demanda": y_t_flat < peak_threshold_mw,
        "Franja_Pico_14_20h": dt_flat.hour.isin(peak_hours),
        "Franja_No_Pico": ~dt_flat.hour.isin(peak_hours),
        "Dias_Laborales": dt_flat.day_of_week < 5,
        "Fines_de_Semana": dt_flat.day_of_week >= 5,
    }

    report: Dict[str, Dict[str, float]] = {}
    for name, mask in masks.items():
        if int(np.sum(mask)) == 0:
            continue
        metrics = compute_regression_metrics(y_t_flat[mask], y_p_flat[mask])
        metrics["n_points"] = int(np.sum(mask))
        report[name] = metrics
    return report


def measure_inference_latency_ms(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    n_warmup: int = 15,
    n_runs: int = 100,
) -> Dict[str, float]:
    """Mide latencia promedio de inferencia por lote y por ventana (ms)."""
    model.eval()
    x = sample_input.to(device)
    batch_size = x.shape[0]
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t_start

    batch_ms = (elapsed_s / n_runs) * 1000.0
    per_sample_ms = batch_ms / max(batch_size, 1)
    return {
        "batch_size": float(batch_size),
        "batch_latency_ms": float(batch_ms),
        "sample_latency_ms": float(per_sample_ms),
    }


def summarize_multi_seed_runs(
    run_metrics: Iterable[Dict[str, float]],
    keys: Sequence[str] = ("RMSE", "MAE", "MAPE", "R2"),
) -> Dict[str, float]:
    """Resume media y desviación estándar entre múltiples semillas aleatorias."""
    rows = list(run_metrics)
    summary: Dict[str, float] = {"n_seeds": float(len(rows))}
    for key in keys:
        vals = [r[key] for r in rows if key in r]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary
