import pytest
import torch
import time
from ynot.echelle import Echellogram


def test_cuda():
    """Do you have NVIDIA GPUs (CUDA) available?"""

    assert torch.cuda.is_available()

    vec1 = torch.arange(10).unsqueeze(0).cuda()
    assert vec1.sum() == 45


def test_echelle_device():
    """Can we cast the model to GPU?"""

    device = 'cuda'
    echellogram = Echellogram(device='cuda')
    echellogram = echellogram.to(device) #no-op

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
    scene_model = echellogram.native_pixel_model(echellogram.amps, echellogram.λ_vector)
    t1 = time.time()
    net_time = t1 - t0
    print(f"\n\t{echellogram.device}: {net_time:0.5f} seconds", end="\t")
    assert echellogram.amps.shape == (echellogram.n_amps,)
    assert echellogram.λ_vector.shape == (echellogram.n_amps,)
    assert echellogram.λ_vector.dtype == echellogram.xx.dtype
    assert scene_model.shape == echellogram.xx.shape
    assert scene_model.dtype == echellogram.xx.dtype


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
    echellogram = Echellogram()
    profile_coeffs = torch.tensor([3.2, -1.5]).double()
    profile = echellogram.source_profile_simple(profile_coeffs)
    assert profile.shape == echellogram.xx.shape
    assert profile.dtype == echellogram.xx.dtype

    assert echellogram.p_coeffs.shape == (2,2)
    assert echellogram.p_coeffs[0].shape == (2,)
    profile = echellogram.source_profile_simple(echellogram.p_coeffs[0].squeeze())
    assert profile.shape == echellogram.xx.shape
    assert profile.dtype == echellogram.xx.dtype
