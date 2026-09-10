r"""Neural network components for the Precily model.

Exact port of the Keras architecture from Chawla et al. (Nat Commun 2022),

    Input(input_dim)
      -> Dense(1429) -> act
      -> Dense(512)  -> act -> Dropout(p)
      -> Dense(140)  -> act -> Dropout(p)
      -> Dense(200)  -> act -> Dropout(p)
      -> Dense(1)

``act`` is ReLU by default (the published architecture). Setting the
hyperparameter ``activation="leaky_relu"`` swaps in LeakyReLU with
``negative_slope``; the topology and the state_dict layout are unchanged.
This is a remedy for dying ReLUs: with plain ReLU the trained network can
collapse a whole hidden layer to zero for part of the inputs, after which the
prediction is a constant that no longer depends on the sample.

input_dim = n_pathways (GSVA) + n_drug_features (Morgan/SMILESVec).
With Morgan fingerprints the drug dimension differs and input_dim
is set accordingly at build time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PrecilyNetwork(nn.Module):
    """Feed-forward regressor predicting LN(IC50) from pathway + drug features."""

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        negative_slope: float = 0.01,
    ):
        """
        Initialize the Precily network.

        :param input_dim: total feature dimension (pathways + drug features)
        :param dropout: dropout probability between hidden layers
        :param activation: "relu" (published architecture) or "leaky_relu"
        :param negative_slope: slope of LeakyReLU, ignored for "relu"
        :raises ValueError: if activation is neither "relu" nor "leaky_relu"
        """
        super().__init__()
        if activation not in ("relu", "leaky_relu"):
            raise ValueError(f"Unknown activation '{activation}', expected 'relu' or 'leaky_relu'.")

        def act() -> nn.Module:
            return nn.ReLU() if activation == "relu" else nn.LeakyReLU(negative_slope)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1429),
            act(),
            nn.Linear(1429, 512),
            act(),
            nn.Dropout(dropout),
            nn.Linear(512, 140),
            act(),
            nn.Dropout(dropout),
            nn.Linear(140, 200),
            act(),
            nn.Dropout(dropout),
            nn.Linear(200, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass.

        :param x: [batch, input_dim] feature tensor
        :return: [batch] predicted LN(IC50)
        """
        return self.net(x).squeeze(-1)
