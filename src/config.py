"""Configuración centralizada y reproducible del proyecto (Parte 2 - Entregable 2)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProjectConfig:
    """Parámetros globales de datos, modelo, entrenamiento y evaluación."""

    # Rutas del proyecto
    root_dir: Path = ROOT_DIR
    raw_csv_path: Path = ROOT_DIR / "data" / "raw" / "PJME_hourly.csv"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    results_dir: Path = ROOT_DIR / "results"
    checkpoints_dir: Path = ROOT_DIR / "results" / "checkpoints"
    logs_dir: Path = ROOT_DIR / "results" / "logs"
    tables_dir: Path = ROOT_DIR / "results" / "tables"
    figures_dir: Path = ROOT_DIR / "results" / "figures"
    entregable_figures_dir: Path = ROOT_DIR / "entregable_2" / "figures"

    # Serie temporal y partición cronológica estricta
    datetime_col: str = "Datetime"
    target_col: str = "PJME_MW"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    window_size: int = 72
    horizon: int = 24
    weekly_lag_hours: int = 168

    # Lista ordenada de características de entrada (F = 9)
    feature_cols: Tuple[str, ...] = (
        "PJME_MW",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "lag_168",
    )

    # Arquitectura neuronal (comparativa controlada LSTM Pura vs. CNN1D-LSTM)
    conv_filters: int = 64
    conv_kernels: Tuple[int, int] = (3, 5)
    pool_size: int = 2
    lstm_hidden: int = 128
    lstm_layers: int = 1
    fc_hidden: int = 64
    dropout: float = 0.2

    # Hiperparámetros de entrenamiento
    batch_size: int = 128
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 8
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    grad_clip_norm: float = 1.0
    seeds: Tuple[int, ...] = (42, 123, 456)

    # Configuración del baseline estadístico SARIMA
    sarima_order: Tuple[int, int, int] = (1, 1, 1)
    sarima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)
    sarima_fit_days: int = 60
    sarima_eval_origins: int = 28  # 28 orígenes diarios distribuídos por estaciones en val

    # Parámetro de análisis en horas pico
    peak_quantile: float = 0.90
    peak_hours: Tuple[int, ...] = tuple(range(14, 21))  # 14:00 a 20:00


DEFAULT_CONFIG = ProjectConfig()
