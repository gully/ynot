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
    dim = 13000
    t0 = time.time()
    matrix = torch.randn(size=(dim, dim)).to(device)
    product = matrix.mm(matrix.T)
    inverse = torch.inverse(product)
    t1 = time.time()
    net_time = t1 - t0
    print("{}: {:0.1f} seconds".format(device, net_time))
    assert net_time < 60.0


def test_devices():
    echellogram = Echellogram()
    print(echellogram.device)
    assert echellogram.device in ["cuda", "cpu"]


@pytest.mark.parametrize(
    "attribute",
    ["xx", "yy", "ss", "emask", "λλ", "device", "y0", "ymax", "\u03bb\u03bb"],
)
def test_valid_module_attributes(attribute):
    echellogram = Echellogram()
    assert hasattr(echellogram, attribute)


@pytest.mark.parametrize("attribute", ["\u03bb\u03bb\u03bb", "junk", "λ"])
def test_invalid_module_attributes(attribute):
    echellogram = Echellogram()
    assert not hasattr(echellogram, attribute)


# def test_property_setting():
