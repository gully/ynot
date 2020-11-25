import pytest
import torch
from torch.distributions import Normal
import time
from ynot.echelle import Echellogram


def test_cuda():
    """Do you have NVIDIA GPUs (CUDA) available?"""

    assert torch.cuda.is_available()

    vec1 = torch.arange(10).unsqueeze(0).cuda()
    assert vec1.sum() == 45


def test_echelle_device():
    """Can we cast the model to GPU?"""

    device = "cuda"
    echellogram = Echellogram(device="cuda")
    echellogram = echellogram.to(device)  # no-op

    assert echellogram.xx.device.type == device
    output = echellogram.forward(1)
    assert echellogram.xx.device.type == device


@pytest.mark.parametrize(
    "attribute",
    ["xx", "yy", "ss", "emask", "λλ", "device", "y0", "ymax", "\u03bb\u03bb"],
)
def test_valid_module_attributes(attribute):
    echellogram = Echellogram()
    assert hasattr(echellogram, attribute)
    assert getattr(echellogram, attribute) is not None


@pytest.mark.parametrize(
    "attribute", ["xx", "yy", "ss", "emask", "λλ"],
)
def test_attributes_properties(attribute):
    """Do the 2D arrays all have the same device and dtype"""
    echellogram = Echellogram()
    assert getattr(echellogram, attribute).device.type == echellogram.device
    assert getattr(echellogram, attribute).device.type in ["cuda", "cpu"]
    assert getattr(echellogram, attribute).shape == (echellogram.nx, echellogram.ny)
    assert getattr(echellogram, attribute).dtype == torch.float64


@pytest.mark.parametrize(
    "device", ["cuda", "cpu"],
)
def test_scene_model(device):
    """Do the scene models have the right shape"""
    echellogram = Echellogram(device=device)
    scene_model = echellogram.single_arcline(3.1, 21770.0, 0.76)
    assert scene_model.shape == echellogram.xx.shape
    assert scene_model.dtype == echellogram.xx.dtype

    t0 = time.time()
    scene_model = echellogram.native_pixel_model(
        echellogram.src_amps, echellogram.λ_src_vector
    )
    t1 = time.time()
    net_time = t1 - t0
    print(f"\n\t{echellogram.device}: {net_time:0.5f} seconds", end="\t")
    assert echellogram.src_amps.shape == (echellogram.n_amps,)
    assert echellogram.λ_src_vector.shape == (echellogram.n_amps,)
    assert echellogram.λ_src_vector.dtype == echellogram.xx.dtype
    assert scene_model.shape == echellogram.xx.shape
    assert scene_model.dtype == echellogram.xx.dtype


@pytest.mark.parametrize(
    "device", ["cuda", "cpu"],
)
def test_sky_model(device):
    """Do the scene models have the right shape"""
    echellogram = Echellogram(device=device, dense_sky=True)
    scene_model = echellogram.sky_model_function()
    assert scene_model.shape == echellogram.xx.shape
    assert scene_model.dtype == echellogram.xx.dtype
    assert len(echellogram.sky_amps) > 1000
    assert not hasattr(echellogram, "sky_continuum_coeffs")
    assert not hasattr(echellogram, "λn")

    echellogram = Echellogram(device=device, dense_sky=False)
    scene_model = echellogram.sky_model_function()
    assert scene_model.shape == echellogram.xx.shape
    assert scene_model.dtype == echellogram.xx.dtype
    assert len(echellogram.sky_amps) > 1

    assert hasattr(echellogram, "sky_continuum_coeffs")
    assert hasattr(echellogram, "λn")
    assert hasattr(echellogram, "cheb_array")
    assert echellogram.λn.shape == echellogram.xx.shape
    assert echellogram.cheb_array.shape[1:3] == echellogram.xx.shape
    assert echellogram.cheb_array.shape[0] == 4
    assert echellogram.λn.max() < 1.2
    assert echellogram.λn.min() > -1.2

    sky_cont = (
        echellogram.cheb_array
        * echellogram.sky_continuum_coeffs.unsqueeze(1).unsqueeze(2)
    ).sum(0)

    assert sky_cont is not None
    assert sky_cont.shape == echellogram.xx.shape


@pytest.mark.parametrize(
    "device", ["cuda", "cpu"],
)
def test_generative_model(device):
    """Do the scene models have the right shape"""
    echellogram = Echellogram(device=device)
    t0 = time.time()
    scene_model = echellogram.generative_model(0)
    t1 = time.time()
    net_time = t1 - t0
    print(f"\n\t{echellogram.device}: {net_time:0.5f} seconds", end="\t")
    assert scene_model.shape == echellogram.xx.shape
    assert scene_model.dtype == echellogram.xx.dtype


@pytest.mark.parametrize(
    "device", ["cuda", "cpu"],
)
def test_forward(device):
    """Do the scene models have the right shape"""
    echellogram = Echellogram(device=device)
    t0 = time.time()
    scene_model = echellogram.forward(1)
    t1 = time.time()
    net_time = t1 - t0
    print(f"\n\t{echellogram.device}: {net_time:0.5f} seconds", end="\t")
    assert scene_model.shape == echellogram.xx.shape
    assert scene_model.dtype == echellogram.xx.dtype


@pytest.mark.parametrize(
    "device", ["cuda", "cpu"],
)
def test_parameters(device):
    """Do the scene models have the right shape"""
    echellogram = Echellogram(device=device)
    for parameter in echellogram.parameters():
        assert parameter.isfinite().all()


def test_trace_profile():
    """Does the trace profile have the right shape?"""

    # Original, trace center resides at fixed slit coordinate
    echellogram = Echellogram()
    profile_coeffs = torch.tensor([3.2, -1.5]).double()
    profile = echellogram.source_profile_simple(profile_coeffs)
    assert profile.shape == echellogram.xx.shape
    assert profile.dtype == echellogram.xx.dtype

    # Still works, p_coeffs 2 and 3 are simply ignored in the simple version
    profile = echellogram.source_profile_simple(echellogram.p_coeffs[0].squeeze())
    assert profile.shape == echellogram.xx.shape
    assert profile.dtype == echellogram.xx.dtype

    # New, trace center drifts ever-so-slightly along slit coordinate
    echellogram = Echellogram()
    profile_coeffs = (
        torch.tensor([3.2, -1.5, -0.1, 0.03]).double().to(echellogram.device)
    )
    assert profile_coeffs.shape == torch.Size([4])

    profile = echellogram.source_profile_medium(profile_coeffs)
    assert profile.shape == echellogram.xx.shape
    assert profile.dtype == echellogram.xx.dtype

    # Original Behavior
    sigma = torch.exp(profile_coeffs[1])
    loc = profile_coeffs[0]
    ln_prob = Normal(loc=loc, scale=sigma).log_prob(echellogram.ss)
    assert ln_prob.shape == echellogram.ss.shape

    # New Behavior
    coeffs = profile_coeffs[[0, 2, 3]]

    loc_array = coeffs.unsqueeze(1) * echellogram.cheb_x
    assert loc_array.shape == torch.Size([3, 1024])
    loc_vector = loc_array.sum(0)
    assert loc_vector.shape == torch.Size([1024])

    ln_prob = Normal(loc=loc_vector.unsqueeze(1), scale=sigma).log_prob(echellogram.ss)
    assert ln_prob.shape == echellogram.ss.shape
