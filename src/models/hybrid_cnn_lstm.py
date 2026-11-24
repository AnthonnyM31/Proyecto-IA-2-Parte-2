"""Arquitectura híbrida propuesta CNN1D-LSTM para pronóstico multihito de demanda."""

from typing import Tuple

import torch
import torch.nn as nn

from src.models.base import RecurrentForecastHead


class HybridCNN1DLSTM(nn.Module):
    """
    Arquitectura híbrida secuencial:
    1) Bloque Conv1D + MaxPool1D + Spatial Dropout (Dropout1d) para extraer patrones locales.
    2) Bloque LSTM + proyección densa compartida (RecurrentForecastHead) para pronóstico de H horas.
    """

    def __init__(
        self,
        in_features: int = 9,
        conv_filters: int = 64,
        conv_kernels: Tuple[int, int] = (3, 5),
        pool_size: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        fc_hidden: int = 64,
        out_horizon: int = 24,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        k1, k2 = conv_kernels
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=conv_filters, kernel_size=k1, padding=k1 // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_filters, out_channels=conv_filters, kernel_size=k2, padding=k2 // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout1d(p=dropout),
        )
        self.recurrent_head = RecurrentForecastHead(
            input_size=conv_filters,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            fc_hidden=fc_hidden,
            out_horizon=out_horizon,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Entrada: [Batch, Window_Len, Features] -> Salida: [Batch, out_horizon]."""
        # Conv1d y Dropout1d esperan [Batch, Channels, Seq_Len]
        x_conv = self.conv_block(x.transpose(1, 2))
        x_seq = x_conv.transpose(1, 2)
        return self.recurrent_head(x_seq)
