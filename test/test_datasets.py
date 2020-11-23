import pytest
import torch
import time
from ynot.datasets import FPADataset
import ccdproc
import warnings
import logging

logging.getLogger("ccdproc").setLevel(logging.ERROR)


def test_import():
    """Can we import the module?"""
    data = FPADataset()
    assert isinstance(data, torch.utils.data.Dataset)


def test_pixels():
    """Can we import the module?"""
    data = FPADataset()
    assert hasattr(data, "pixels")
    assert hasattr(data, "index")


def test_ccdproc():
    """Does CCDProc work and can we silence its warnings"""
    keywords = [
        "imagetyp",
        "filname",
        "slitname",
        "itime",
        "frameno",
        "object",
        "ut",
        "airmass",
        "dispers",
        "slitwidt",
        "slitlen",
        "ra",
        "dec",
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ims = (
            ccdproc.ImageFileCollection(
                "data/2012-11-27/", keywords=keywords, glob_include="NS*.fits",
            )
            .filter(dispers="high")
            .filter(regex_match=True, slitlen="12|24")
        )

    assert ims is not None
