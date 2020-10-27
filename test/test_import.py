import pytest


def test_cuda():

    # Test that torch impport and that cuda is available
    import torch

    assert torch.cuda.is_available()

    vec1 = torch.arange(10).unsqueeze(0).cuda()
    assert torch.arange(10).unsqueeze(0).cuda().sum() == 45
