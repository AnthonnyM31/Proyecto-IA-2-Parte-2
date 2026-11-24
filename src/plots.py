"""Generación reproducible de figuras científicas (300 DPI PNG y PDF vectorial)."""

from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import acf

from src.config import DEFAULT_CONFIG, ProjectConfig
from src.data_pipeline import clean_hourly_series, load_raw_pjme, temporal_split


def _configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _save_figure(fig: plt.Figure, filename: str, cfg: ProjectConfig) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.entregable_figures_dir.mkdir(parents=True, exist_ok=True)
    for out_dir in (cfg.figures_dir, cfg.entregable_figures_dir):
        fig.savefig(out_dir / f"{filename}.png", bbox_inches="tight", dpi=300)
        fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_eda_and_splits(cfg: ProjectConfig) -> None:
    """Figura 1: Serie PJME con particiones, perfiles diarios/mensuales y ACF."""
    raw_df = load_raw_pjme(cfg.raw_csv_path, cfg.datetime_col, cfg.target_col)
    clean_df, _ = clean_hourly_series(raw_df, cfg.datetime_col, cfg.target_col)
    train_df, val_df, test_df, _ = temporal_split(
        clean_df, cfg.train_ratio, cfg.val_ratio, cfg.datetime_col
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))

    # Panel (a): Serie completa diaria promedio con splits
    daily = clean_df.set_index(cfg.datetime_col)[cfg.target_col].resample("D").mean()
    ax = axes[0, 0]
    ax.plot(daily.index, daily.values / 1000.0, linewidth=0.7, color="#1f4e78", alpha=0.85)
    val_start = pd.Timestamp(val_df[cfg.datetime_col].min())
    test_start = pd.Timestamp(test_df[cfg.datetime_col].min())
    ax.axvline(val_start, color="#d95f02", linestyle="--", linewidth=1.3, label="Inicio Val (2014)")
    ax.axvline(test_start, color="#7570b3", linestyle="-.", linewidth=1.3, label="Inicio Test Sellado")
    ax.axvspan(test_start, daily.index.max(), color="#7570b3", alpha=0.08)
    ax.set_title("(a) Demanda media diaria PJME y partición cronológica")
    ax.set_ylabel("Carga promedio (GW)")
    ax.legend(loc="upper left", frameon=True)

    # Panel (b): Perfil horario laboral vs fin de semana
    ax = axes[0, 1]
    dt = train_df[cfg.datetime_col]
    tmp = train_df.assign(hour=dt.dt.hour, is_wknd=dt.dt.day_of_week >= 5)
    for is_wknd, label, color in (
        (False, "Días laborales", "#1b9e77"),
        (True, "Fines de semana", "#d95f02"),
    ):
        sub = tmp[tmp["is_wknd"] == is_wknd].groupby("hour")[cfg.target_col]
        mean = sub.mean() / 1000.0
        q25 = sub.quantile(0.25) / 1000.0
        q75 = sub.quantile(0.75) / 1000.0
        ax.plot(mean.index, mean.values, marker="o", markersize=3, color=color, label=label)
        ax.fill_between(mean.index, q25.values, q75.values, color=color, alpha=0.16)
    ax.set_title("(b) Ciclo diurno típico e IQR (Entrenamiento)")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Demanda (GW)")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=True)

    # Panel (c): Estacionalidad mensual (boxplot)
    ax = axes[1, 0]
    tmp_m = train_df.assign(month=dt.dt.month, gw=train_df[cfg.target_col] / 1000.0)
    sns.boxplot(data=tmp_m, x="month", y="gw", ax=ax, color="#6baed6", fliersize=0.8)
    ax.set_title("(c) Distribución estacional por mes (Entrenamiento)")
    ax.set_xlabel("Mes del año")
    ax.set_ylabel("Demanda (GW)")

    # Panel (d): Función de autocorrelación (0 a 168 horas)
    ax = axes[1, 1]
    acf_vals = acf(train_df[cfg.target_col].iloc[-24 * 365 :].to_numpy(), nlags=168, fft=True)
    lags = np.arange(len(acf_vals))
    ax.plot(lags, acf_vals, color="#2b5c8f", linewidth=1.2)
    for lag_mark, lbl in ((24, "24h"), (72, "72h (W)"), (168, "168h (7d)")):
        ax.axvline(lag_mark, color="#cb181d", linestyle=":", linewidth=1.1)
        ax.text(lag_mark + 2, 0.22, lbl, color="#cb181d", fontsize=8, rotation=90)
    ax.set_title("(d) Autocorrelación horaria (ACF hasta t-168)")
    ax.set_xlabel("Retardo (horas)")
    ax.set_ylabel("ACF")
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    _save_figure(fig, "fig1_eda_pjme", cfg)


def plot_architecture_diagram(cfg: ProjectConfig) -> None:
    """Figura 2: Diagrama arquitectónico comparativo controlado (PureLSTM vs CNN1D-LSTM)."""
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def draw_box(x, y, w, h, text, fc, ec="#2c3e50"):
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.0",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#34495e"))

    # Entrada común
    draw_box(2, 15, 16, 12, "Entrada Causal\n[B, W=72, F=9]\n(MW, sin/cos, lag 168)", "#e8f4f8")

    # Rama superior: Baseline LSTM Pura
    ax.text(50, 38.5, "Rama A: Baseline Homogéneo (LSTM Pura)", ha="center", fontsize=9.5, fontweight="bold", color="#d95f02")
    draw_box(28, 28, 22, 8, "LSTM (128 unidades)\n[B, 72, 9] -> [B, 128]", "#fdebd0", "#d95f02")
    draw_box(56, 28, 19, 8, "FC(64)+ReLU+Drop\n-> Salida [B, 24]", "#fdebd0", "#d95f02")

    # Rama inferior: Arquitectura Propuesta CNN1D-LSTM
    ax.text(50, 19.5, "Rama B: Arquitectura Híbrida Propuesta (CNN1D-LSTM)", ha="center", fontsize=9.5, fontweight="bold", color="#1b7837")
    draw_box(23, 5, 19, 11, "Conv1D(k=3, 64)\n+ Conv1D(k=5, 64)\n[B, 64, 72]", "#e5f5e0", "#1b7837")
    draw_box(45, 5, 15, 11, "MaxPool1D(2)\n+ Dropout1d(0.2)\n[B, 36, 64]", "#e5f5e0", "#1b7837")
    draw_box(63, 5, 15, 11, "LSTM (128 u.)\nÚltimo estado\n[B, 128]", "#e5f5e0", "#1b7837")
    draw_box(81, 5, 17, 11, "FC(64)+ReLU+Drop\n-> Salida [B, 24]\n(Horizonte 24h)", "#d9f0d3", "#1b7837")

    # Salida común etiqueta
    draw_box(81, 28, 17, 8, "Pronóstico 24h\nDesnormalizado\n(MW reales)", "#f0f0f0")

    draw_arrow(18.5, 24, 27.5, 32)
    draw_arrow(50.5, 32, 55.5, 32)
    draw_arrow(75.5, 32, 80.5, 32)

    draw_arrow(18.5, 18, 22.5, 10.5)
    draw_arrow(42.5, 10.5, 44.5, 10.5)
    draw_arrow(60.5, 10.5, 62.5, 10.5)
    draw_arrow(78.5, 10.5, 80.5, 10.5)
    draw_arrow(89.5, 16.5, 89.5, 27.5)

    _save_figure(fig, "fig2_architecture_diagram", cfg)


def plot_training_curves(cfg: ProjectConfig) -> None:
    """Figura 3: Curvas de pérdida MSE entrenamiento vs validación entre 3 semillas."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    specs = (("pure_lstm", "LSTM Pura (Baseline)"), ("cnn1d_lstm", "CNN1D-LSTM (Propuesta)"))

    for ax, (model_key, title) in zip(axes, specs):
        for seed, alpha in zip(cfg.seeds, (0.95, 0.55, 0.55)):
            hist_path = cfg.logs_dir / f"{model_key}_seed{seed}_history.csv"
            if not hist_path.exists():
                continue
            df = pd.read_csv(hist_path)
            lbl_tr = "Entrenamiento" if seed == cfg.seeds[0] else None
            lbl_va = "Validación" if seed == cfg.seeds[0] else None
            ax.plot(df["epoch"], df["train_loss"], color="#1f78b4", alpha=alpha, lw=1.3, label=lbl_tr)
            ax.plot(df["epoch"], df["val_loss"], color="#e31a1c", alpha=alpha, lw=1.3, ls="--", label=lbl_va)

        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(frameon=True)

    axes[0].set_ylabel("Pérdida MSE (escala log)")
    fig.tight_layout()
    _save_figure(fig, "fig3_training_curves", cfg)


def plot_forecast_examples(cfg: ProjectConfig) -> None:
    """Figura 4: Trayectoria semanal típica y episodio de demanda pico (24h)."""
    data = np.load(cfg.logs_dir / "eval_predictions_val.npz", allow_pickle=False)
    y_true = data["y_true_val"]
    dt_val = pd.to_datetime(data["target_dt_val"].ravel()).to_numpy().reshape(y_true.shape)
    pred_p24 = data["pred_persist24"]
    pred_lstm = data["pred_pure_lstm"]
    pred_cnn = data["pred_cnn1d_lstm"]
    sarima_idx = data["sarima_idx"]
    sarima_true = data["sarima_y_true"]
    sarima_pred = data["sarima_y_pred"]

    # Seleccionar 7 días consecutivos sin solapamiento (pasos de 24h) en primavera/verano de val
    origins = pd.to_datetime(data["origin_dt_val"])
    day_indices = np.where(origins.hour == 23)[0]
    mid_pos = len(day_indices) // 3
    week_idx = day_indices[mid_pos : mid_pos + 7]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # Panel (a): Semana típica (7 bloques consecutivos de 24h)
    ax = axes[0]
    t_week = dt_val[week_idx].ravel()
    ax.plot(t_week, y_true[week_idx].ravel() / 1000.0, color="black", lw=1.8, label="Real (PJME)")
    ax.plot(t_week, pred_p24[week_idx].ravel() / 1000.0, color="#999999", lw=1.0, ls=":", label="Persistencia 24h")
    ax.plot(t_week, pred_lstm[week_idx].ravel() / 1000.0, color="#d95f02", lw=1.3, ls="--", label="LSTM Pura")
    ax.plot(t_week, pred_cnn[week_idx].ravel() / 1000.0, color="#1b9e77", lw=1.5, label="CNN1D-LSTM")
    ax.set_title("(a) Pronóstico concatenado 7 días (horizonte 24h/día)")
    ax.set_ylabel("Demanda (GW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(loc="upper right", frameon=True)

    # Panel (b): Día de mayor demanda pico dentro de la muestra SARIMA
    ax = axes[1]
    peak_pos = int(np.argmax(sarima_true.max(axis=1)))
    idx_orig = sarima_idx[peak_pos]
    hours_h = np.arange(1, 25)
    ax.plot(hours_h, sarima_true[peak_pos] / 1000.0, color="black", lw=2.0, marker="o", ms=3, label="Real")
    ax.plot(hours_h, sarima_pred[peak_pos] / 1000.0, color="#7570b3", lw=1.4, ls="-.", label="SARIMA")
    ax.plot(hours_h, pred_lstm[idx_orig] / 1000.0, color="#d95f02", lw=1.4, ls="--", label="LSTM Pura")
    ax.plot(hours_h, pred_cnn[idx_orig] / 1000.0, color="#1b9e77", lw=1.7, label="CNN1D-LSTM")
    peak_date = str(pd.Timestamp(dt_val[idx_orig, 0]).date())
    ax.set_title(f"(b) Episodio pico en validación ({peak_date})")
    ax.set_xlabel("Horizonte de pronóstico h (horas)")
    ax.set_ylabel("Demanda (GW)")
    ax.set_xticks(range(1, 25, 3))
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    _save_figure(fig, "fig4_forecast_comparison", cfg)


def plot_horizon_and_regimes(cfg: ProjectConfig) -> None:
    """Figura 5: Degradación del error por horizonte h=1..24 y comparativa por régimen (PI-2)."""
    df_h = pd.read_csv(cfg.tables_dir / "val_horizon_metrics.csv")
    df_r = pd.read_csv(cfg.tables_dir / "val_regime_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))

    # Panel (a): RMSE por horizonte h=1..24
    ax = axes[0]
    styles = {
        "Persistencia_24h": ("#999999", ":"),
        "Perfil_Condicional": ("#7570b3", "-."),
        "LSTM_Pura": ("#d95f02", "--"),
        "CNN1D_LSTM": ("#1b9e77", "-"),
    }
    for model_name, (col, ls) in styles.items():
        sub = df_h[df_h["Modelo"] == model_name]
        ax.plot(sub["horizon_h"], sub["RMSE"], color=col, linestyle=ls, lw=1.8, label=model_name.replace("_", " "))
    ax.set_title("(a) Crecimiento del RMSE por horizonte (h = 1..24h)")
    ax.set_xlabel("Paso horario futuro (h)")
    ax.set_ylabel("RMSE (MW)")
    ax.set_xticks(range(1, 25, 3))
    ax.legend(frameon=True)

    # Panel (b): RMSE por régimen operativo (Global, Pico P90, Franja 14-20h, Laboral, Fin de semana)
    ax = axes[1]
    regimes = ["Global", "Pico_Demanda_P90", "Franja_Pico_14_20h", "Dias_Laborales", "Fines_de_Semana"]
    reg_labels = ["Global", "Pico P90", "14–20h", "Laboral", "Fin sem."]
    models_plot = ["Persistencia_24h", "LSTM_Pura", "CNN1D_LSTM"]
    sub_r = df_r[df_r["Regimen"].isin(regimes) & df_r["Modelo"].isin(models_plot)]

    x = np.arange(len(regimes))
    width = 0.25
    colors = ["#a6cee3", "#d95f02", "#1b9e77"]
    for idx, (m_name, col) in enumerate(zip(models_plot, colors)):
        vals = [
            float(sub_r[(sub_r["Modelo"] == m_name) & (sub_r["Regimen"] == r)]["RMSE"].iloc[0])
            for r in regimes
        ]
        ax.bar(x + (idx - 1) * width, vals, width=width, color=col, label=m_name.replace("_", " "))

    ax.set_title("(b) Desempeño en regímenes operativos críticos (PI-2)")
    ax.set_xticks(x)
    ax.set_xticklabels(reg_labels)
    ax.set_ylabel("RMSE (MW)")
    ax.legend(frameon=True)

    fig.tight_layout()
    _save_figure(fig, "fig5_horizon_and_regimes", cfg)


def main() -> None:
    cfg = DEFAULT_CONFIG
    _configure_style()
    print("[INFO] Generando figuras científicas en results/figures/ y entregable_2/figures/...")
    plot_eda_and_splits(cfg)
    plot_architecture_diagram(cfg)
    plot_training_curves(cfg)
    plot_forecast_examples(cfg)
    plot_horizon_and_regimes(cfg)
    print("[OK] Todas las figuras generadas exitosamente.")


if __name__ == "__main__":
    main()
