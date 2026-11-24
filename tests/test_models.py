"""Tests unitarios de arquitecturas neuronales y cálculo exacto de métricas."""

import numpy as np
import torch

from src.metrics import compute_horizon_metrics, compute_regression_metrics
from src.models import HybridCNN1DLSTM, PureLSTMForecaster


def test_models_forward_shapes_and_gradients():
    batch_size, window_len, features, horizon = 8, 72, 9, 24
    x = torch.randn(batch_size, window_len, features)
    target = torch.randn(batch_size, horizon)

    for model in (
        PureLSTMForecaster(in_features=features, out_horizon=horizon),
        HybridCNN1DLSTM(in_features=features, out_horizon=horizon),
    ):
        out = model(x)
        assert out.shape == (batch_size, horizon)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_regression_and_horizon_metrics_accuracy():
    y_true = np.array([[100.0, 200.0], [100.0, 200.0]])
    y_pred = np.array([[110.0, 180.0], [90.0, 220.0]])

    m = compute_regression_metrics(y_true, y_pred)
    # Errores: 10, 20, 10, 20 -> MSE=(100+400+100+400)/4=250 -> RMSE=sqrt(250)
    assert np.isclose(m["RMSE"], np.sqrt(250.0))
    assert np.isclose(m["MAE"], 15.0)
    assert np.isclose(m["MAPE"], 10.0)

    df_h = compute_horizon_metrics(y_true, y_pred)
    assert list(df_h["horizon_h"]) == [1, 2]
    assert np.isclose(df_h.loc[0, "MAE"], 10.0)
    assert np.isclose(df_h.loc[1, "MAE"], 20.0)
