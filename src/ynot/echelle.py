"""
The ``echelle`` module provides the core functionality of ynot via :class:`ynot.echelle.Echellogram`.
"""

import torch
from torch import nn


class Echellogram(nn.Module):
    r"""
    A PyTorch layer that provides a parameter set and transformations to model echellograms.

    Args:
        ybounds (tuple): the y_0 and y_max of the raw echellogram to analyze. Default: (425, 510)
    """

    def __init__(self, device="cuda", ybounds=(425, 510)):
        super().__init__()

        self.device = device
        self.y0 = ybounds[0]
        self.ymax = ybounds[1]
        self.ny = self.ymax = self.y0

        self.nx = 1024
        self.xvec = torch.arange(0, self.nx, 1.0)
        self.yvec = torch.arange(0, self.ny, 1.0)
        self.xx, self.yy = torch.meshgrid(self.xvec, self.yvec)

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

    def s_of_xy(self, params):
        """
        Return the along-slit coordinate :math`s` as a function of :math`(x,y)`

        Args:
            params (torch.tensor or tuple): the coefficents relating s to (x,y).
        Returns:
            (torch.tensor): the 2D map of math:`s(x,y)`
        """
        y0, kk, dy0_dx = params
        s_out = kk * ((self.yy - y0) - dy0_dx * self.xx)
        return s_out
