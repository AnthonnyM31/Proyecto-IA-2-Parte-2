"""Módulo unificado de entrenamiento en GPU/CPU para PureLSTM y HybridCNN1DLSTM."""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_CONFIG, ProjectConfig
from src.data_pipeline import PreparedDatasets, prepare_all_datasets
from src.models import HybridCNN1DLSTM, PureLSTMForecaster


def set_reproducible_seed(seed: int) -> None:
    """Fija semillas aleatorias en Python, NumPy y PyTorch (CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    """Devuelve GPU CUDA si está disponible; CPU en caso contrario."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name: str, in_features: int, cfg: ProjectConfig = DEFAULT_CONFIG) -> nn.Module:
    """Instancia la arquitectura solicitada garantizando idéntico bloque recurrente/denso."""
    if model_name == "pure_lstm":
        return PureLSTMForecaster(
            in_features=in_features,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            fc_hidden=cfg.fc_hidden,
            out_horizon=cfg.horizon,
            dropout=cfg.dropout,
        )
    if model_name == "cnn1d_lstm":
        return HybridCNN1DLSTM(
            in_features=in_features,
            conv_filters=cfg.conv_filters,
            conv_kernels=cfg.conv_kernels,
            pool_size=cfg.pool_size,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            fc_hidden=cfg.fc_hidden,
            out_horizon=cfg.horizon,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Modelo desconocido '{model_name}'. Use 'pure_lstm' o 'cnn1d_lstm'.")


def count_trainable_parameters(model: nn.Module) -> int:
    """Cuenta parámetros entrenables de una arquitectura PyTorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    grad_clip_norm: float = 1.0,
) -> float:
    """Ejecuta una época de entrenamiento o validación y retorna pérdida MSE promedio."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            preds = model(xb)
            loss = criterion(preds, yb)

            if is_train:
                loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            batch_n = xb.size(0)
            total_loss += float(loss.item()) * batch_n
            total_samples += batch_n

    return total_loss / max(total_samples, 1)


def train_single_run(
    model_name: str,
    seed: int,
    datasets: PreparedDatasets,
    cfg: ProjectConfig = DEFAULT_CONFIG,
) -> Dict[str, object]:
    """Entrena un modelo con una semilla dada, aplica EarlyStopping y guarda checkpoint/log."""
    set_reproducible_seed(seed)
    device = resolve_device()

    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TensorDataset(
        torch.from_numpy(datasets.train.X),
        torch.from_numpy(datasets.train.y_scaled),
    )
    val_ds = TensorDataset(
        torch.from_numpy(datasets.val.X),
        torch.from_numpy(datasets.val.y_scaled),
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin)

    in_features = datasets.train.X.shape[2]
    model = build_model(model_name, in_features=in_features, cfg=cfg).to(device)
    n_params = count_trainable_parameters(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    ckpt_path = cfg.checkpoints_dir / f"{model_name}_seed{seed}.pt"
    history_rows: List[Dict[str, float]] = []

    t_train_start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        t_ep = time.perf_counter()
        train_loss = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, grad_clip_norm=cfg.grad_clip_norm
        )
        val_loss = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        ep_time = time.perf_counter() - t_ep

        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "epoch_time_s": ep_time,
            }
        )

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "state_dict": model.state_dict(),
                    "in_features": in_features,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1

        print(
            f"[{model_name} | seed={seed}] Época {epoch:02d}/{cfg.max_epochs} "
            f"- train_mse={train_loss:.6f} - val_mse={val_loss:.6f} "
            f"- lr={current_lr:.2e} ({ep_time:.2f}s)"
        )

        if patience_counter >= cfg.early_stopping_patience:
            print(f"[{model_name} | seed={seed}] Early stopping activado en época {epoch}.")
            break

    total_train_time = time.perf_counter() - t_train_start
    hist_df = pd.DataFrame(history_rows)
    hist_path = cfg.logs_dir / f"{model_name}_seed{seed}_history.csv"
    hist_df.to_csv(hist_path, index=False)

    return {
        "model_name": model_name,
        "seed": seed,
        "device": str(device),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_ran": len(history_rows),
        "best_val_mse": best_val_loss,
        "total_train_time_s": total_train_time,
        "mean_epoch_time_s": float(hist_df["epoch_time_s"].mean()),
        "checkpoint_path": str(ckpt_path),
        "history_path": str(hist_path),
    }


def main() -> None:
    cfg = DEFAULT_CONFIG
    print(f"[INFO] Preparando datasets desde {cfg.raw_csv_path}...")
    datasets = prepare_all_datasets(cfg)
    print(f"[INFO] Dispositivo activo: {resolve_device()}")

    all_summaries: List[Dict[str, object]] = []
    for model_name in ("pure_lstm", "cnn1d_lstm"):
        for seed in cfg.seeds:
            summary = train_single_run(model_name, seed, datasets, cfg)
            all_summaries.append(summary)

    summary_path = cfg.logs_dir / "training_runs_summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"[OK] Entrenamiento completado. Resumen guardado en {summary_path}")


if __name__ == "__main__":
    main()
