"""Evaluación estandarizada y homogénea sobre Validación (manteniendo Test sellado)."""

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import DEFAULT_CONFIG, ProjectConfig
from src.data_pipeline import PreparedDatasets, inverse_scale_targets, prepare_all_datasets
from src.metrics import (
    compute_horizon_metrics,
    compute_regime_metrics,
    compute_regression_metrics,
    measure_inference_latency_ms,
    summarize_multi_seed_runs,
)
from src.models.baselines import (
    evaluate_sarima_on_indices,
    fit_conditional_hourly_profile,
    predict_conditional_hourly_profile,
    predict_persistence_24h,
    predict_persistence_168h,
    run_stationarity_diagnostics,
    select_sarima_eval_indices,
)
from src.train import build_model, count_trainable_parameters, resolve_device


def predict_neural_model(
    model_name: str,
    seed: int,
    datasets: PreparedDatasets,
    cfg: ProjectConfig = DEFAULT_CONFIG,
    batch_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, float], int]:
    """Carga el mejor checkpoint de un modelo y genera predicciones en MW reales sobre val."""
    device = resolve_device()
    ckpt_path = cfg.checkpoints_dir / f"{model_name}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}. Ejecute src.train primero.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    in_features = int(ckpt["in_features"])
    model = build_model(model_name, in_features=in_features, cfg=cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = count_trainable_parameters(model)

    X_val = datasets.val.X
    preds_scaled_list: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_val), batch_size):
            xb = torch.from_numpy(X_val[start : start + batch_size]).to(device)
            preds_scaled_list.append(model(xb).cpu().numpy())

    preds_scaled = np.vstack(preds_scaled_list)
    preds_mw = inverse_scale_targets(preds_scaled, datasets.target_scaler)

    sample_batch = torch.from_numpy(X_val[: min(64, len(X_val))])
    latency = measure_inference_latency_ms(model, sample_batch, device=device)
    return preds_mw, latency, n_params


def _load_train_time_s(model_name: str, seed: int, cfg: ProjectConfig) -> float:
    summary_file = cfg.logs_dir / "training_runs_summary.json"
    if not summary_file.exists():
        return float("nan")
    rows = json.loads(summary_file.read_text(encoding="utf-8"))
    for row in rows:
        if row["model_name"] == model_name and int(row["seed"]) == int(seed):
            return float(row["total_train_time_s"])
    return float("nan")


def main() -> None:
    cfg = DEFAULT_CONFIG
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Cargando y preparando dataset real PJME...")
    datasets = prepare_all_datasets(cfg)

    # Guardar auditoría de datos y particiones
    data_audit = {
        "cleaning": datasets.cleaning_stats,
        "splits": datasets.split_stats,
    }
    (cfg.processed_dir / "dataset_audit.json").write_text(
        json.dumps(data_audit, indent=2), encoding="utf-8"
    )
    (cfg.tables_dir / "dataset_audit.json").write_text(
        json.dumps(data_audit, indent=2), encoding="utf-8"
    )

    # Pruebas de estacionariedad (ADF / KPSS) sobre train para justificar SARIMA
    print("[INFO] Ejecutando pruebas de estacionariedad (ADF/KPSS) sobre entrenamiento...")
    stationarity = run_stationarity_diagnostics(datasets.train_df[cfg.target_col])
    (cfg.tables_dir / "stationarity_diagnostics.json").write_text(
        json.dumps(stationarity, indent=2), encoding="utf-8"
    )

    y_val_true = datasets.val.y_raw
    dt_val = datasets.val.target_datetimes

    # 1. Líneas base heurísticas sobre el conjunto completo de validación
    print("[INFO] Evaluando líneas base heurísticas sobre Validación completa...")
    pred_persist24 = predict_persistence_24h(datasets.val, datasets.target_scaler, cfg.horizon)
    pred_persist168 = predict_persistence_168h(datasets.val_df, dt_val, cfg)
    profile, fallback_mean = fit_conditional_hourly_profile(datasets.train_df, cfg)
    pred_profile = predict_conditional_hourly_profile(profile, fallback_mean, dt_val)

    full_models_preds: Dict[str, np.ndarray] = {
        "Persistencia_24h": pred_persist24,
        "Persistencia_168h": pred_persist168,
        "Perfil_Condicional": pred_profile,
    }

    summary_rows: List[Dict[str, object]] = []
    for name, pred in full_models_preds.items():
        m = compute_regression_metrics(y_val_true, pred)
        summary_rows.append(
            {
                "Modelo": name,
                "RMSE_mean": m["RMSE"],
                "RMSE_std": 0.0,
                "MAE_mean": m["MAE"],
                "MAE_std": 0.0,
                "MAPE_mean": m["MAPE"],
                "MAPE_std": 0.0,
                "R2_mean": m["R2"],
                "R2_std": 0.0,
                "Parametros": 0,
                "Train_Time_s": 0.0,
                "Infer_Sample_ms": 0.0,
            }
        )

    # 2. Modelos neuronales (multi-semilla sobre Validación completa)
    best_seed_preds: Dict[str, np.ndarray] = {}
    for model_key, display_name in (("pure_lstm", "LSTM_Pura"), ("cnn1d_lstm", "CNN1D_LSTM")):
        seed_metrics: List[Dict[str, float]] = []
        latencies: List[float] = []
        train_times: List[float] = []
        best_rmse = float("inf")
        best_pred = None
        n_params = 0

        for seed in cfg.seeds:
            pred_mw, lat, n_params = predict_neural_model(model_key, seed, datasets, cfg)
            m = compute_regression_metrics(y_val_true, pred_mw)
            seed_metrics.append(m)
            latencies.append(lat["sample_latency_ms"])
            train_times.append(_load_train_time_s(model_key, seed, cfg))
            if m["RMSE"] < best_rmse:
                best_rmse = m["RMSE"]
                best_pred = pred_mw

        agg = summarize_multi_seed_runs(seed_metrics)
        best_seed_preds[display_name] = best_pred
        full_models_preds[display_name] = best_pred

        summary_rows.append(
            {
                "Modelo": display_name,
                "RMSE_mean": agg["RMSE_mean"],
                "RMSE_std": agg["RMSE_std"],
                "MAE_mean": agg["MAE_mean"],
                "MAE_std": agg["MAE_std"],
                "MAPE_mean": agg["MAPE_mean"],
                "MAPE_std": agg["MAPE_std"],
                "R2_mean": agg["R2_mean"],
                "R2_std": agg["R2_std"],
                "Parametros": n_params,
                "Train_Time_s": float(np.nanmean(train_times)),
                "Infer_Sample_ms": float(np.mean(latencies)),
            }
        )

    df_full_summary = pd.DataFrame(summary_rows)
    df_full_summary.to_csv(cfg.tables_dir / "val_full_summary.csv", index=False)

    # 3. Evaluación desagregada por horizonte (h=1..24) y por régimen operativo (PI-2)
    horizon_frames = []
    regime_rows = []
    for model_name, pred in full_models_preds.items():
        df_h = compute_horizon_metrics(y_val_true, pred)
        df_h.insert(0, "Modelo", model_name)
        horizon_frames.append(df_h)

        reg = compute_regime_metrics(
            y_val_true, pred, dt_val, datasets.peak_threshold_mw, cfg.peak_hours
        )
        for reg_name, reg_m in reg.items():
            row = {"Modelo": model_name, "Regimen": reg_name}
            row.update(reg_m)
            regime_rows.append(row)

    pd.concat(horizon_frames, ignore_index=True).to_csv(
        cfg.tables_dir / "val_horizon_metrics.csv", index=False
    )
    pd.DataFrame(regime_rows).to_csv(cfg.tables_dir / "val_regime_metrics.csv", index=False)

    # 4. Comparación homogénea con SARIMA en orígenes estratificados de validación
    print(f"[INFO] Evaluando SARIMA sobre {cfg.sarima_eval_origins} días estratificados de Validación...")
    sarima_idx = select_sarima_eval_indices(datasets.val, n_origins=cfg.sarima_eval_origins)
    sarima_true, sarima_pred, sarima_meta = evaluate_sarima_on_indices(
        datasets.train_df, datasets.val_df, datasets.val, sarima_idx, cfg
    )
    (cfg.tables_dir / "sarima_metadata.json").write_text(
        json.dumps(sarima_meta, indent=2), encoding="utf-8"
    )

    matched_rows = []
    m_sar = compute_regression_metrics(sarima_true, sarima_pred)
    matched_rows.append(
        {
            "Modelo": f"SARIMA{cfg.sarima_order}x{cfg.sarima_seasonal_order}",
            **m_sar,
            "Infer_Day_ms": sarima_meta["mean_infer_latency_ms"],
        }
    )
    for model_name, pred in full_models_preds.items():
        m_sub = compute_regression_metrics(y_val_true[sarima_idx], pred[sarima_idx])
        matched_rows.append({"Modelo": model_name, **m_sub, "Infer_Day_ms": float("nan")})

    df_matched = pd.DataFrame(matched_rows)
    df_matched.to_csv(cfg.tables_dir / "val_sarima_matched_summary.csv", index=False)

    # Guardar predicciones en formato binario ligero para src/plots.py
    np.savez_compressed(
        cfg.logs_dir / "eval_predictions_val.npz",
        y_true_val=y_val_true,
        target_dt_val=dt_val.astype("datetime64[ns]"),
        origin_dt_val=datasets.val.origin_datetimes.astype("datetime64[ns]"),
        pred_persist24=pred_persist24,
        pred_persist168=pred_persist168,
        pred_profile=pred_profile,
        pred_pure_lstm=best_seed_preds["LSTM_Pura"],
        pred_cnn1d_lstm=best_seed_preds["CNN1D_LSTM"],
        sarima_idx=sarima_idx,
        sarima_y_true=sarima_true,
        sarima_y_pred=sarima_pred,
    )

    print("\n=== RESUMEN VALIDACIÓN COMPLETA (2014-2015) ===")
    print(df_full_summary.to_string(index=False))
    print("\n=== COMPARATIVA EMPAREJADA CON SARIMA (28 DÍAS ESTRATIFICADOS) ===")
    print(df_matched.to_string(index=False))
    print("[OK] Evaluación completada y tablas exportadas en results/tables/.")


if __name__ == "__main__":
    main()
