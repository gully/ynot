import pytest
import torch
import time
from ynot.datasets import FPADataset

def test_import():
    """Can we import the module?"""
    data = FPADataset()
    assert isinstance(data, torch.utils.data.Dataset)

def test_pixels():
    """Can we import the module?"""
    data = FPADataset()
    assert hasattr(data, 'pixels')
    assert hasattr(data, 'index')
