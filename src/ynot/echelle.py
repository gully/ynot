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
        self.fiducial = torch.tensor([21779.0, 0.310])

        self.nx = 1024
        self.xvec = torch.arange(0, self.nx, 1.0)
        self.yvec = torch.arange(0, self.ny, 1.0)
        self.xx, self.yy = torch.meshgrid(self.xvec, self.yvec)

        self.bkg_const = nn.Parameter(
            torch.tensor(200.0, requires_grad=True, dtype=torch.float64, device=device)
        )
        # self.sky_const = nn.Parameter(torch.tensor(50.0, requires_grad=True, dtype=torch.float64, device='cuda'))
        self.s_coeffs = nn.Parameter(
            torch.tensor(
                [14.635, 0.20352, -0.004426],
                requires_grad=True,
                dtype=torch.float64,
                device="cuda",
            )
        )
        self.n_amps = 2000
        self.amps = nn.Parameter(
            torch.ones(
                self.n_amps, requires_grad=True, dtype=torch.float64, device=device
            )
        )

        self.smoothness = nn.Parameter(
            torch.tensor(-2.27, requires_grad=True, dtype=torch.float64, device=device)
        )

        self.lam_coeffs = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 1.0, -0.7923],
                requires_grad=True,
                dtype=torch.float64,
                device=device,
            )
        )

        self.p_coeffs = nn.Parameter(
            torch.tensor(
                [[3.0, -1.0], [9.0, -1.0]],
                requires_grad=True,
                dtype=torch.float64,
                device=device,
            )
        )
        self.src_amps = nn.Parameter(
            self.amps.clone().detach().requires_grad_(True).double().to(device)
        )

        self._ss = None
        self._emask = None
        self._λλ = None
        self.λλ = None

    def forward(self, x):

        return 1

    def s_of_xy(self, params):
        """
        Return the along-slit coordinate :math:`s` as a function of :math:`(x,y)`

        Args:
            params (torch.tensor or tuple): the coefficents relating s to (x,y).
        Returns:
            (torch.tensor): the 2D map of :math:`s(x,y)`
        """
        y0, kk, dy0_dx = params
        s_out = kk * ((self.yy - y0) - dy0_dx * self.xx)
        return s_out


    def edge_mask(self, smoothness):
        """Apply the product of two sigmoid functions to make a smooth tophat

        Currently hard-coded with a 12 arcsecond slit.
        """
        arg1 = self.ss - 0.0
        arg2 = 12.0 - s_in
        return (
            1.0
            / (1.0 + torch.exp(-arg1 / torch.exp(smoothness)))
            * 1.0
            / (1.0 + torch.exp(-arg2 / torch.exp(smoothness)))
        )

    def lam_xy(self, c):
        """A 2D Surface mapping :math:`(x,y)` pixels to :math:`\lambda`

        Each (x,y) pixel coordinate maps to a single central wavelength. This
        function performs that transformation, given the coefficents of polynomials,
        the `x` and `y` values, and a fiducial central wavelength and dispersion.
        The coefficients in this function are intended to be fit through iterative
        stochastic gradient descent.

        Args:
            arg1 (int): Description of arg1
            arg2 (str): Description of arg2

        Returns:
            bool: Description of return value

        """
        x = (self.xx-self.nx/2)/(self.nx/2)
        y = (self.yy-self.ny/2)/(self.ny/2)
        const = self.fiducial[0]
        c0 = c[0] # Shift: Angstroms ~[-3, 3]
        cx1 = self.fiducial[1]*(1+c[1]*0.01)*self.nx/2 # Dispersion adjustment: Dimensionless ~[-1, 1]
        cx2 = 1.0+c[2] # Pixel-dependent dispersion: Angstroms [-1.5, 1.5]
        #cx3 = c[4] # Higher-order dispersion [-1,1]
        cy1 = c[3] # Vertically-Tilted straight arclines [Angstroms/pixel] [ -0.3, 0.3]

        term0 = c0
        xterm1 = cx1 * x
        xterm2 = cx2 * (2*x**2 - 1)
        #xterm3 = cx3 * (4*x**3 - 3*x)
        yterm1 = cy1 * y

        output = const + (term0 + xterm1 + xterm2 ) + yterm1
        return output

    @property
    def ss(self):
        return self._ss

    @ss.setter
    def ss(self, params):
        self._ss = self.s_of_xy(params)

    @property
    def emask(self):
        return self._emask

    @emask.setter
    def emask(self, param):
        self._emask = self.edge_mask(param)

    @property
    def λλ(self):
        return self._λλ

    @λλ.setter
    def λλ(self, params):
        self._λλ = self.lam_xy(params)
