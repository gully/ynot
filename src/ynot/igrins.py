"""
echelle
-------

Spectral dispersion and slit length axes are generally not perfectly aligned with the rectilinear pixel grid of a spectrograph detector, complicating the extraction of echelle spectroscopy.  There exists some mapping of each 2D :math:`(x,y)` pixel to a new coordinate system of wavelength and slit position :math:`(\lambda,s)`, with :math:`x` and :math:`y` in units of pixels, :math:`\lambda` in units of Ångstroms, and :math:`s` in units of arcseconds.  These surfaces can therefore be represented as scalar functions over :math:`x` and :math:`y`.  The `ynot` project infers this mapping for all pixels in an echelle order.  For example, this mapping could be parameterized as separable polynomials:

.. math::

   \lambda(x,y) &= \lambda_0 + c_1 x + c_2 x^2 + c_3 y

   s(x,y)      &= s_0 + b_1 y + b_2 x



Echellogram
############
"""

import torch
from torch import nn
from torch.distributions import Normal
import pandas as pd


class Echellogram(nn.Module):
    r"""
    A PyTorch layer that provides a parameter set and transformations to model echellograms.

    Args:
        device (str): Either "cuda" for GPU acceleration, or "cpu" otherwise
        ybounds (tuple): the :math:`y_0` and :math:`y_{max}` of the raw echellogram to analyze.
            Default: (425, 510)
        dense_sky (bool): whether or not to treat the sky background as dense (~1400) parameters or fit a few (~3-10) lines from a pre-determined line list.  A pre-determined line list is needed for wavelength calibration, while a dense sky gives the best sky subtraction for weak lines that are not in the line list.  The best approach---a *hybrid* of the two---is not yet implemented.
            Default: False
    """

    def __init__(self, device="cuda", ybounds=(425, 510), dense_sky=False):
        super().__init__()

        self.device = device
        self.y0 = ybounds[0]
        self.ymax = ybounds[1]
        self.ny = self.ymax - self.y0
        self.fiducial = torch.tensor([21779.0, 0.310], device=device).double()

        self.nx = 1024
        self.xvec = torch.arange(0, self.nx, device=device).double()
        self.xn = (
            2 * (self.xvec - self.xvec.mean()) / (self.xvec.max() - self.xvec.min())
        )
        self.yvec = torch.arange(0, self.ny, device=device).double()
        self.xx, self.yy = torch.meshgrid(self.xvec, self.yvec)
        self.cheb_x = torch.stack(
            [self.xn ** 0, self.xn ** 1, 2 * self.xn ** 2 - 1]
        ).to(self.device)

        # This is sampled in log
        self.bkg_const = nn.Parameter(
            torch.tensor(6.9, requires_grad=True, dtype=torch.float64, device=device)
        )

        self.s_coeffs = nn.Parameter(
            torch.tensor(
                [14.635, 0.20352, -0.004426, 0.0],
                requires_grad=True,
                dtype=torch.float64,
                device=device,
            )
        )
        self.n_amps = 1500

        # These represent inputs to log.
        self.src_amps = nn.Parameter(
            4.1
            * torch.ones(
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
                [[3, -1, 0, 0], [9, -1, 0, 0]],
                requires_grad=True,
                dtype=torch.float64,
                device=device,
            )
        )

        # Set the s(x,y), and λ(x,y) coordinates
        self.ss = self.s_of_xy(self.s_coeffs)
        self.λλ = self.lam_xy(self.lam_coeffs)
        self.emask = self.edge_mask(self.smoothness)
        self.λλ_min = self.λλ.min().detach().item()
        self.λλ_max = self.λλ.max().detach().item()

        # The wavelength vector should not require grad.
        self.λ_src_vector = torch.linspace(
            self.λλ_min,
            self.λλ_max,
            self.n_amps,
            device=self.device,
            requires_grad=False,
            dtype=torch.float64,
        )

        # For dense sky sampling (no wavelength calibration)
        if dense_sky:
            self.n_sky = self.n_amps
            self.λ_sky_vector = self.λ_src_vector
            self.sky_amps = nn.Parameter(
                5.2
                * torch.ones(
                    self.n_amps, requires_grad=True, dtype=torch.float64, device=device
                )
            )
            self.sky_model_function = self.dense_sky_model
        else:
            self.λ_sky_vector, peaks = self.get_skyline_wavelengths()
            self.n_sky = len(self.λ_sky_vector)

            # These will be treated as natural logs
            self.sky_amps = nn.Parameter(
                torch.log(peaks)
                + 2.3
                * torch.ones(
                    self.n_sky, requires_grad=True, dtype=torch.float64, device=device,
                )
            )
            # Sampled in log
            self.sky_continuum_coeffs = nn.Parameter(
                torch.tensor(
                    [5.7, 0.0, 0.0, 0.0],
                    requires_grad=True,
                    dtype=torch.float64,
                    device=device,
                )
            )

            # The grad needlessly adds memory to the backpropagation for this smooth function
            with torch.no_grad():
                self.λn = (
                    2 * (self.λλ - self.λλ.mean()) / (self.λλ.max() - self.λλ.min())
                )
            self.cheb_array = torch.stack(
                [
                    torch.ones_like(self.xx, device=device),
                    self.λn,
                    2 * self.λn ** 2 - 1,
                    4 * self.λn ** 3 - 3 * self.λn,
                ]
            ).to(device)
            self.sky_model_function = self.sparse_sky_model

    def forward(self, index):
        """The forward pass of the neural network model

        Args:
            index (int): the index of the ABB'A' nod frames: *e.g.* A=0, B=1, B'=2, A'=3
        Returns:
            (torch.tensor): the 2D generative scene model destined for backpropagation parameter tuning
        """
        return self.generative_model(index)

    def s_of_xy(self, params):
        """
        The along-slit coordinate :math:`s` as a function of :math:`(x,y)`, given coefficients

        Args:
            params (torch.tensor or tuple): the polynomial weights, first order in :math:`x` and :math:`y`
        Returns:
            (torch.tensor): the 2D surface map :math:`s(x,y)`
        """
        y0, kk, dy0_dx, kk_x = params
        s_out = (kk + kk_x / 100 * self.xx) * ((self.yy - y0) - dy0_dx * self.xx)
        return s_out

    def edge_mask(self, log_smoothness):
        r"""The soft-edge pixel mask defined by the extent of the spectrograph slit length

        Constructed by the product of two sigmoid functions to make a smooth tophat:

        .. math::

           m_e = \mathscr{S}(0) \cdot (1 - \mathscr{S}(12) )

        Currently hard-coded to a 12 arcsecond slit.

        Args:
            log_smoothness (torch.tensor or tuple): the :math:`\beta` smoothness parameter related to image quality
        Returns:
            (torch.tensor): the 2D surface map :math:`m_e(x,y)`
        """
        arg1 = self.ss - 0.0
        arg2 = 12.0 - self.ss
        bottom_edge = torch.sigmoid(arg1 / torch.exp(log_smoothness))
        top_edge = torch.sigmoid(arg2 / torch.exp(log_smoothness))
        return bottom_edge * top_edge

    def lam_xy(self, c):
        r"""A 2D Surface mapping :math:`(x,y)` pixels to :math:`\lambda`


        Each (x,y) pixel coordinate maps to a single central wavelength. This
        function performs that transformation, given the coefficents of polynomials,
        the `x` and `y` values, and a fiducial central wavelength and dispersion.
        The coefficients in this function are intended to be fit through iterative
        stochastic gradient descent.

        Args:
            c (torch.tensor): polynomial weights and bias fitted through backpropagation

        Returns:
            (torch.tensor): the 2D surface map :math:`\lambda(x,y)`

        """
        x = (self.xx - self.nx / 2) / (self.nx / 2)
        y = (self.yy - self.ny / 2) / (self.ny / 2)
        const = self.fiducial[0]
        c0 = c[0]  # Shift: Angstroms ~[-3, 3]
        cx1 = (
            self.fiducial[1] * (1 + c[1] * 0.01) * self.nx / 2
        )  # Dispersion adjustment: Dimensionless ~[-1, 1]
        cx2 = 1.0 + c[2]  # Pixel-dependent dispersion: Angstroms [-1.5, 1.5]
        # cx3 = c[4] # Higher-order dispersion [-1,1]
        cy1 = c[3]  # Vertically-Tilted straight arclines [Angstroms/pixel] [ -0.3, 0.3]

        term0 = c0
        xterm1 = cx1 * x
        xterm2 = cx2 * (2 * x ** 2 - 1)
        # xterm3 = cx3 * (4*x**3 - 3*x)
        yterm1 = cy1 * y

        output = const + (term0 + xterm1 + xterm2) + yterm1
        return output

    def single_arcline(self, amp, lam_0, lam_sigma):
        """Evaluate a normalized arcline given a 2D wavelength map"""
        ln_prob = Normal(loc=lam_0, scale=lam_sigma).log_prob(self.λλ)
        return amp * torch.exp(ln_prob)

    def native_pixel_model(self, amp_of_lambda, lam_vec):
        """A Native-pixel model of the scene"""
        log_scene_cube = Normal(
            loc=lam_vec.unsqueeze(0).unsqueeze(0), scale=0.42
        ).log_prob(self.λλ.unsqueeze(2))
        return (
            amp_of_lambda.unsqueeze(0).unsqueeze(0) * torch.exp(log_scene_cube)
        ).sum(axis=2)

    def dense_sky_model(self):
        """A sky model with dense (~1400) spectral lines"""
        return self.native_pixel_model(torch.exp(self.sky_amps), self.λ_sky_vector)

    def sparse_sky_model(self):
        """A sky model with a few (~3-10) spectral lines"""
        sky_lines = self.native_pixel_model(torch.exp(self.sky_amps), self.λ_sky_vector)
        sky_continuum = self.sky_continuum_model()
        return sky_lines + sky_continuum

    def sky_continuum_model(self):
        """A smooth model for the background sky emission in sparse-sky models

        Returns:
            (torch.tensor): the 2D sky emission continuum
        """
        log_sky_cont = (
            self.cheb_array * self.sky_continuum_coeffs.unsqueeze(1).unsqueeze(2)
        ).sum(0)
        return torch.exp(log_sky_cont)

    def source_profile_simple(self, p_coeffs):
        """The profile of the sky source, given position and width coefficients and s

        p_coeffs[0]: Position in arcseconds (0,12)
        p_coeffs[1]: Width in arcseconds ~1.0
        """
        sigma = torch.exp(p_coeffs[1])
        ln_prob = Normal(loc=p_coeffs[0], scale=sigma).log_prob(self.ss)
        return torch.exp(ln_prob)

    def source_profile_medium(self, p_coeffs):
        """The profile of the sky source, given position, width, trend coefficients

        p_coeffs[0]: Position in arcseconds (0,12)
        p_coeffs[1]: Width in arcseconds ~1.0
        p_coeffs[2]: Position drift as a function of x (arcseconds per pixel)
        p_coeffs[3]: Position drift as a function of x^2 (arcseconds per pixel^2)
        """
        sigma = torch.exp(p_coeffs[1])
        coeffs = p_coeffs[[0, 2, 3]]
        loc_vector = (coeffs.unsqueeze(1) * self.cheb_x).sum(0)
        ln_prob = Normal(loc=loc_vector.unsqueeze(1), scale=sigma).log_prob(self.ss)
        return torch.exp(ln_prob)

    def generative_model(self, index):
        """The generative model resembles echelle spectra traces in astronomy data"""
        self.ss = self.s_of_xy(self.s_coeffs)
        self.λλ = self.lam_xy(self.lam_coeffs)
        self.emask = self.edge_mask(self.smoothness)
        sky_model = self.sky_model_function()
        src_model = self.native_pixel_model(torch.exp(self.src_amps), self.λ_src_vector)
        src_prof = self.source_profile_medium(self.p_coeffs[index].squeeze())
        net_sky = self.emask * sky_model
        net_src = src_prof * src_model
        return net_sky + net_src + torch.exp(self.bkg_const)

    def get_skyline_wavelengths(self):
        """Get the wavelengths of bright sky lines (e.g. OH)"""
        df = pd.read_csv(
            "/home/gully/GitHub/ynot/data/ir_ohlines.dat",
            names=["wl", "rel_flux"],
            delim_whitespace=True,
        )
        df = df[(df.wl > self.λλ_min) & (df.wl < self.λλ_max)]
        wls = torch.tensor(df.wl.values, device=self.device, dtype=torch.float64)
        peaks = torch.tensor(
            df.rel_flux.values, device=self.device, dtype=torch.float64
        )
        return (wls, peaks)

