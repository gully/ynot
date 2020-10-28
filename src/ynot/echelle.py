"""
The ``echelle`` module provides the core functionality of ynot via :class:`ynot.echelle.Echellogram`.
"""

import torch
from torch import nn


class Echellogram(nn.Module):
    r"""
    A PyTorch layer that provides a parameter set and transformations to model echellograms.

    The parameter set depends on the science goal.

    Args:
        task (str): which task to run
    """

    def __init__(self, task=None):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.b = nn.Parameter(
            torch.tensor(5.0, requires_grad=True, dtype=torch.float64)
        )
        self.a = nn.Parameter(
            torch.tensor([0.00, 0.50], requires_grad=True, dtype=torch.float64)
        )

    def forward(self, x):
        # Computes the outputs / predictions
        variable_part = torch.matmul(x, self.a)
        bias_term = self.b * torch.ones(
            (x.shape[0], 1024, 1024), dtype=torch.float64, device="cuda"
        )

        dark_term = variable_part.unsqueeze(1).unsqueeze(1) * torch.ones(
            (x.shape[0], 1024, 1024), dtype=torch.float64, device="cuda"
        )

        return bias_term + dark_term
