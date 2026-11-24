"""Línea base de Deep Learning: red LSTM pura sin extracción convolucional."""

import torch
import torch.nn as nn

from src.models.base import RecurrentForecastHead


class PureLSTMForecaster(nn.Module):
    """
    Línea base recurrente homogénea (LSTM pura).
    Comparte exactamente la misma etapa LSTM + proyección densa que HybridCNN1DLSTM
    para aislar el aporte neto de las convoluciones 1D.
    """

    def __init__(
        self,
        in_features: int = 9,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        fc_hidden: int = 64,
        out_horizon: int = 24,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.recurrent_head = RecurrentForecastHead(
            input_size=in_features,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            fc_hidden=fc_hidden,
            out_horizon=out_horizon,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Entrada: [Batch, Window_Len, Features] -> Salida: [Batch, out_horizon]."""
        return self.recurrent_head(x)
