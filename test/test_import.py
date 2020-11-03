import pytest
import torch
import time
from ynot.echelle import Echellogram


def test_cuda():
    """Do you have NVIDIA GPUs (CUDA) available?"""
    # Test that cuda is available

    assert torch.cuda.is_available()

    vec1 = torch.arange(10).unsqueeze(0).cuda()
    assert vec1.sum() == 45


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_big_matrices_cuda(device):
    """Can you do invert a large matrix (13,000 x 13,000)"""
    dim = 13_000
    t0 = time.time()
    matrix = torch.randn(size=(dim, dim)).to(device)
    product = matrix.mm(matrix.T)
    inverse = torch.inverse(product)
    t1 = time.time()
    net_time = t1 - t0
    print(f"{device}: {net_time:0.1f} seconds")
    assert net_time < 60.0


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

    amplitudes = echellogram.amps.unsqueeze(0).unsqueeze(0)
    dense_λ = (
        torch.linspace(
            echellogram.λλ.min().item(),
            echellogram.λλ.max().item(),
            echellogram.n_amps,
            device=echellogram.device,
            dtype=torch.float64,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    t0 = time.time()
    scene_model = echellogram.native_pixel_model(amplitudes, dense_λ)
    t1 = time.time()
    net_time = t1 - t0
    print(f"\n\t{echellogram.device}: {net_time:0.5f} seconds", end="\t")
    assert amplitudes.shape == (1, 1, echellogram.n_amps)
    assert dense_λ.shape == (1, 1, echellogram.n_amps)
    assert dense_λ.dtype == echellogram.xx.dtype
    assert scene_model.shape == echellogram.xx.shape
    assert scene_model.dtype == echellogram.xx.dtype


def test_trace_profile():
    """Does the trace profile have the right shape?"""
    echellogram = Echellogram()
    profile_coeffs = torch.tensor([3.2, -1.5]).double()
    profile = echellogram.source_profile_simple(profile_coeffs)
    assert profile.shape == echellogram.xx.shape
    assert profile.dtype == echellogram.xx.dtype
