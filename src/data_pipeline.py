"""Pipeline causal de ingesta, limpieza DST, ingeniería de características y ventanas."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import DEFAULT_CONFIG, ProjectConfig


@dataclass
class SplitTensors:
    """Contenedor inmutable de tensores de entrada/salida y metadatos reales."""

    X: np.ndarray                # [N, W, F] normalizado
    y_scaled: np.ndarray         # [N, H] normalizado [0, 1]
    y_raw: np.ndarray            # [N, H] en MW reales
    origin_datetimes: np.ndarray # [N] último timestamp conocido de la ventana X
    target_datetimes: np.ndarray # [N, H] timestamps reales del horizonte H


@dataclass
class PreparedDatasets:
    """Conjunto completo preparado sin fuga de datos para entrenamiento y evaluación."""

    clean_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train: SplitTensors
    val: SplitTensors
    test: SplitTensors
    target_scaler: MinMaxScaler
    cleaning_stats: Dict[str, object]
    split_stats: Dict[str, object]
    peak_threshold_mw: float


def load_raw_pjme(csv_path: Path, datetime_col: str = "Datetime", target_col: str = "PJME_MW") -> pd.DataFrame:
    """Carga el CSV crudo de demanda horaria PJME y valida el esquema esperado."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en '{path}'. "
            "Consulta README.md para descargar PJME_hourly.csv."
        )
    df = pd.read_csv(path)
    if datetime_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Columnas esperadas {[datetime_col, target_col]}, encontradas {list(df.columns)}")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype(np.float64)
    return df.sort_values(datetime_col).reset_index(drop=True)


def clean_hourly_series(
    df: pd.DataFrame,
    datetime_col: str = "Datetime",
    target_col: str = "PJME_MW",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Resuelve duplicados de cambio horario (DST), reindexa a grilla horaria continua
    e imputa huecos aislados mediante interpolación lineal.
    """
    raw_rows = len(df)
    n_nan_raw = int(df[target_col].isna().sum())

    # Promediar marcas de tiempo duplicadas (cambio DST de otoño)
    n_dup_timestamps = int(df.duplicated(subset=[datetime_col]).sum())
    df_dedup = (
        df.groupby(datetime_col, as_index=False)[target_col]
        .mean()
        .sort_values(datetime_col)
        .reset_index(drop=True)
    )

    # Reindexar a frecuencia horaria estricta para detectar huecos faltantes (ej. DST primavera)
    full_index = pd.date_range(
        start=df_dedup[datetime_col].min(),
        end=df_dedup[datetime_col].max(),
        freq="h",
    )
    df_hourly = df_dedup.set_index(datetime_col).reindex(full_index)
    df_hourly.index.name = datetime_col

    missing_mask = df_hourly[target_col].isna()
    n_missing_hours = int(missing_mask.sum())

    # Longitud máxima de hueco consecutivo para auditar calidad de interpolación
    max_gap = 0
    if n_missing_hours > 0:
        groups = (~missing_mask).cumsum()
        max_gap = int(missing_mask.groupby(groups).sum().max())

    df_hourly[target_col] = df_hourly[target_col].interpolate(method="linear", limit_direction="both")
    df_clean = df_hourly.reset_index()

    stats: Dict[str, object] = {
        "raw_rows": raw_rows,
        "raw_nans": n_nan_raw,
        "duplicate_timestamps_resolved": n_dup_timestamps,
        "missing_hours_inserted": n_missing_hours - n_nan_raw,
        "total_hours_interpolated": n_missing_hours,
        "max_consecutive_missing_hours": max_gap,
        "clean_rows": len(df_clean),
        "start_datetime": str(df_clean[datetime_col].min()),
        "end_datetime": str(df_clean[datetime_col].max()),
        "mean_mw": float(df_clean[target_col].mean()),
        "std_mw": float(df_clean[target_col].std()),
        "min_mw": float(df_clean[target_col].min()),
        "max_mw": float(df_clean[target_col].max()),
    }
    return df_clean, stats


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    datetime_col: str = "Datetime",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Partición cronológica estricta 70/15/15 sin mezcla aleatoria."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    split_stats: Dict[str, object] = {
        "train_rows": len(train_df),
        "train_start": str(train_df[datetime_col].min()),
        "train_end": str(train_df[datetime_col].max()),
        "val_rows": len(val_df),
        "val_start": str(val_df[datetime_col].min()),
        "val_end": str(val_df[datetime_col].max()),
        "test_rows": len(test_df),
        "test_start": str(test_df[datetime_col].min()),
        "test_end": str(test_df[datetime_col].max()),
    }
    return train_df, val_df, test_df, split_stats


def add_causal_features(df_split: pd.DataFrame, cfg: ProjectConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Genera variables temporales cíclicas y retardo semanal causal dentro de un split aislado.
    Descarta las primeras `weekly_lag_hours` filas sin historia previa dentro del split.
    """
    data = df_split.copy()
    dt = data[cfg.datetime_col]
    hours = dt.dt.hour.to_numpy(dtype=np.float64)
    dow = dt.dt.day_of_week.to_numpy(dtype=np.float64)
    months = dt.dt.month.to_numpy(dtype=np.float64)

    data["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    data["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    data["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    data["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    data["month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
    data["month_cos"] = np.cos(2.0 * np.pi * months / 12.0)
    data["is_weekend"] = (dt.dt.day_of_week >= 5).astype(np.float64)
    data["lag_168"] = data[cfg.target_col].shift(cfg.weekly_lag_hours)

    return data.iloc[cfg.weekly_lag_hours :].reset_index(drop=True)


def fit_target_scaler(train_df: pd.DataFrame, target_col: str = "PJME_MW") -> MinMaxScaler:
    """Ajusta MinMaxScaler [0, 1] exclusivamente sobre la partición de entrenamiento."""
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(train_df[[target_col]].to_numpy(dtype=np.float64))
    return scaler


def scale_split_features(
    df_feat: pd.DataFrame,
    target_scaler: MinMaxScaler,
    cfg: ProjectConfig = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza `PJME_MW` y `lag_168` con el escalador ajustado únicamente en train.
    Las variables trigonométricas quedan acotadas en [-1, 1] y `is_weekend` en {0, 1}.
    """
    scaled_df = df_feat.copy()
    mw_vals = scaled_df[[cfg.target_col]].to_numpy(dtype=np.float64)
    lag_vals = scaled_df[["lag_168"]].to_numpy(dtype=np.float64)

    scaled_mw = target_scaler.transform(mw_vals).ravel()
    scaled_lag = target_scaler.transform(lag_vals).ravel()

    scaled_df[cfg.target_col] = scaled_mw
    scaled_df["lag_168"] = scaled_lag

    X_matrix = scaled_df[list(cfg.feature_cols)].to_numpy(dtype=np.float32)
    y_scaled_vec = scaled_mw.astype(np.float32)
    y_raw_vec = mw_vals.ravel().astype(np.float64)
    return X_matrix, y_scaled_vec, y_raw_vec


def build_sliding_windows(
    X_matrix: np.ndarray,
    y_scaled_vec: np.ndarray,
    y_raw_vec: np.ndarray,
    datetimes: pd.Series,
    window_size: int = 72,
    horizon: int = 24,
) -> SplitTensors:
    """
    Construye ventanas deslizantes estrictamente causales dentro de un único split:
    entrada [t-W+1 .. t] -> objetivo [t+1 .. t+H].
    """
    total_len = len(X_matrix)
    n_samples = total_len - window_size - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Serie demasiado corta ({total_len}) para W={window_size} y H={horizon}."
        )

    dt_arr = datetimes.to_numpy()
    n_features = X_matrix.shape[1]

    X = np.empty((n_samples, window_size, n_features), dtype=np.float32)
    y_scaled = np.empty((n_samples, horizon), dtype=np.float32)
    y_raw = np.empty((n_samples, horizon), dtype=np.float64)
    origin_dt = np.empty(n_samples, dtype=dt_arr.dtype)
    target_dt = np.empty((n_samples, horizon), dtype=dt_arr.dtype)

    for i in range(n_samples):
        end_w = i + window_size
        end_h = end_w + horizon
        X[i] = X_matrix[i:end_w]
        y_scaled[i] = y_scaled_vec[end_w:end_h]
        y_raw[i] = y_raw_vec[end_w:end_h]
        origin_dt[i] = dt_arr[end_w - 1]
        target_dt[i] = dt_arr[end_w:end_h]

    return SplitTensors(
        X=X,
        y_scaled=y_scaled,
        y_raw=y_raw,
        origin_datetimes=origin_dt,
        target_datetimes=target_dt,
    )


def inverse_scale_targets(y_scaled: np.ndarray, target_scaler: MinMaxScaler) -> np.ndarray:
    """Invierte la transformación MinMax [0, 1] a megavatios (MW) reales."""
    arr = np.asarray(y_scaled, dtype=np.float64)
    orig_shape = arr.shape
    inv = target_scaler.inverse_transform(arr.reshape(-1, 1))
    return inv.reshape(orig_shape)


def prepare_all_datasets(cfg: ProjectConfig = DEFAULT_CONFIG) -> PreparedDatasets:
    """Ejecuta el pipeline causal completo y devuelve splits y tensores listos."""
    raw_df = load_raw_pjme(cfg.raw_csv_path, cfg.datetime_col, cfg.target_col)
    clean_df, cleaning_stats = clean_hourly_series(raw_df, cfg.datetime_col, cfg.target_col)
    train_raw, val_raw, test_raw, split_stats = temporal_split(
        clean_df, cfg.train_ratio, cfg.val_ratio, cfg.datetime_col
    )

    # Umbral de demanda pico calculado únicamente sobre entrenamiento (cero leakage)
    peak_threshold_mw = float(np.quantile(train_raw[cfg.target_col].to_numpy(), cfg.peak_quantile))

    # Escalamiento ajustado exclusivamente sobre entrenamiento
    target_scaler = fit_target_scaler(train_raw, cfg.target_col)

    # Construcción independiente de características y ventanas por split
    splits = {}
    for name, split_df in (("train", train_raw), ("val", val_raw), ("test", test_raw)):
        feat_df = add_causal_features(split_df, cfg)
        X_mat, y_sc, y_rw = scale_split_features(feat_df, target_scaler, cfg)
        splits[name] = build_sliding_windows(
            X_mat,
            y_sc,
            y_rw,
            feat_df[cfg.datetime_col],
            window_size=cfg.window_size,
            horizon=cfg.horizon,
        )

    split_stats["train_windows"] = len(splits["train"].X)
    split_stats["val_windows"] = len(splits["val"].X)
    split_stats["test_windows"] = len(splits["test"].X)
    split_stats["peak_threshold_p90_mw"] = peak_threshold_mw

    return PreparedDatasets(
        clean_df=clean_df,
        train_df=train_raw,
        val_df=val_raw,
        test_df=test_raw,
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        target_scaler=target_scaler,
        cleaning_stats=cleaning_stats,
        split_stats=split_stats,
        peak_threshold_mw=peak_threshold_mw,
    )
