import pytest
import torch
import time
from ynot.datasets import FPADataset
import ccdproc
import warnings
import logging
import astropy

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


def test_dataset_image_collection():
    """Does dataet initialization work and can we silence ccdproc warnings"""
    data = FPADataset(root_dir="data/2012-11-27/")

    assert data.nirspec_collection is not None

    attributes = [
        "ccds",
        "data",
        "ext",
        "files",
        "hdus",
        "headers",
        "location",
        "files_filtered",
        "filter",
        "keywords",
        "summary",
        "values",
    ]
    for attribute in attributes:
        assert hasattr(data.nirspec_collection, attribute)

    assert type(data.nirspec_collection) == ccdproc.image_collection.ImageFileCollection
    assert type(data.nirspec_collection.summary) == astropy.table.table.Table
    assert len(data.nirspec_collection.files) > 1
    assert len(list(data.nirspec_collection.headers())) == len(
        data.nirspec_collection.files
    )
