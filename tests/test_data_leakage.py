"""Tests unitarios de integridad temporal, limpieza DST y prevención de data leakage."""

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.data_pipeline import (
    add_causal_features,
    build_sliding_windows,
    clean_hourly_series,
    fit_target_scaler,
    scale_split_features,
    temporal_split,
)


def _make_synthetic_df(hours: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01 00:00:00", periods=hours, freq="h")
    vals = 30000.0 + 4000.0 * np.sin(2.0 * np.pi * np.arange(hours) / 24.0)
    return pd.DataFrame({"Datetime": dates, "PJME_MW": vals})


def test_clean_hourly_series_resolves_dst_duplicates_and_gaps():
    df = _make_synthetic_df(100)
    # Simular hueco DST (eliminar 2 horas) y duplicado DST (duplicar 1 fila con valor distinto)
    df_gap = df.drop(index=[20, 21]).reset_index(drop=True)
    dup_row = df_gap.iloc[[40]].copy()
    dup_row["PJME_MW"] = df_gap.loc[40, "PJME_MW"] + 200.0
    df_dirty = pd.concat([df_gap, dup_row], ignore_index=True).sample(frac=1.0, random_state=42)

    df_clean, stats = clean_hourly_series(df_dirty)
    assert stats["duplicate_timestamps_resolved"] == 1
    assert stats["missing_hours_inserted"] == 2
    assert stats["max_consecutive_missing_hours"] == 2
    assert len(df_clean) == 100
    assert df_clean["Datetime"].is_monotonic_increasing
    assert df_clean["PJME_MW"].isna().sum() == 0


def test_temporal_split_strict_chronological_order():
    df = _make_synthetic_df(1000)
    train_df, val_df, test_df, _ = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150
    assert train_df["Datetime"].max() < val_df["Datetime"].min()
    assert val_df["Datetime"].max() < test_df["Datetime"].min()


def test_scaler_fit_only_on_train_prevents_leakage():
    df = _make_synthetic_df(3000)
    train_df, val_df, test_df, _ = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    # Inyectar pico extremo fuera de rango después de la ventana warm-up (lag_168)
    val_df.loc[200, "PJME_MW"] = 999999.0
    test_df.loc[200, "PJME_MW"] = -50000.0

    scaler = fit_target_scaler(train_df)
    assert float(scaler.data_max_[0]) == float(train_df["PJME_MW"].max())
    assert float(scaler.data_min_[0]) == float(train_df["PJME_MW"].min())

    cfg = ProjectConfig()
    val_feat = add_causal_features(val_df, cfg)
    X_val, y_val_sc, _ = scale_split_features(val_feat, scaler, cfg)
    # Si no hubo leakage al ajustar scaler, el valor inyectado en val debe salir > 1.0
    assert np.max(y_val_sc) > 1.0
    assert X_val.shape[1] == len(cfg.feature_cols)


def test_sliding_windows_causality_and_split_isolation():
    df = _make_synthetic_df(3000)
    train_df, val_df, _, _ = temporal_split(df, train_ratio=0.70, val_ratio=0.15)
    cfg = ProjectConfig()
    scaler = fit_target_scaler(train_df)

    val_feat = add_causal_features(val_df, cfg)
    X_val, y_sc, y_rw = scale_split_features(val_feat, scaler, cfg)
    tensors = build_sliding_windows(
        X_val, y_sc, y_rw, val_feat["Datetime"], window_size=cfg.window_size, horizon=cfg.horizon
    )

    val_start = pd.Timestamp(val_df["Datetime"].min())
    val_end = pd.Timestamp(val_df["Datetime"].max())
    for i in range(len(tensors.X)):
        origin = pd.Timestamp(tensors.origin_datetimes[i])
        first_target = pd.Timestamp(tensors.target_datetimes[i, 0])
        last_target = pd.Timestamp(tensors.target_datetimes[i, -1])
        assert origin < first_target
        assert (first_target - origin) == pd.Timedelta(hours=1)
        assert origin >= val_start
        assert last_target <= val_end
