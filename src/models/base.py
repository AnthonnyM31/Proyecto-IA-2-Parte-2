"""Bloques neuronales compartidos para garantizar paridad experimental entre modelos."""

import torch
import torch.nn as nn


class RecurrentForecastHead(nn.Module):
    """Bloque común LSTM + proyección densa para horizonte multihito H."""

    def __init__(
        self,
        input_size: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        fc_hidden: int = 64,
        out_horizon: int = 24,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, out_horizon),
        )

    def forward(self, seq_features: torch.Tensor) -> torch.Tensor:
        """Recibe [Batch, Seq_Len, Features] y devuelve [Batch, out_horizon]."""
        lstm_out, _ = self.lstm(seq_features)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)
