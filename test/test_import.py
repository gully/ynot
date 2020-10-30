import pytest
import torch
import time

def test_cuda():

    # Test that cuda is available

    assert torch.cuda.is_available()

    vec1 = torch.arange(10).unsqueeze(0).cuda()
    assert torch.arange(10).unsqueeze(0).cuda().sum() == 45


def test_big_matrices_cuda():

    devices = ["cuda"]
    dim = 10000
    for device in devices:
        t0 = time.time()
        matrix = torch.randn(size=(dim, dim)).to(device)
        product = matrix.mm(matrix.T)
        inverse = torch.inverse(product)
        t1 = time.time()
        net_time = t1 - t0
        print("{}: {:0.1f} seconds".format(device, net_time))
        assert net_time < 60.0

def test_module_import():

    from ynot.echelle import Echellogram

    echellogram = Echellogram()


#def test_property_setting():
