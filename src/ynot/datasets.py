"""
datasets
--------

This utility loads data, inheriting from the PyTorch Dataset model.
This approach allows very large datasets---possibly larger than RAM---to inform models.

FPADataset
############
"""

import torch
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import kornia
import ccdproc
import astropy.units as u
from astropy.nddata import CCDData
from sklearn.cluster import KMeans
import warnings
import logging

logging.getLogger("ccdproc").setLevel(logging.ERROR)


# custom dataset loader
class FPADataset(Dataset):
    r"""Read in two AB nods of an Echellogram (experimental)

    Args:
        ybounds (tuple of ints): the :math:`(y_0, y_{max})` bounds for isolated echelle trace
        root_dir (str): path to the directory containing 2D echellograms as fits files
        inpaint_bad_pixels (bool): flag for whether or not to inpaint bad pixels

    """

    def __init__(self, ybounds=(425, 510), root_dir=None, inpaint_bad_pixels=False):
        super().__init__()

        if root_dir is None:
            root_dir = "/home/gully/GitHub/ynot/test/data/2012-11-27/"
        self.root_dir = root_dir
        # self.nirspec_collection = self.create_nirspec_collection()
        # self.unique_objects = self.get_unique_objects()
        # self.label_nirspec_nods()
        nodA_path = self.root_dir + "/NS.20121127.49332.fits"
        nodA_data = fits.open(nodA_path)[0].data.astype(np.float64)
        nodA = torch.tensor(nodA_data)

        nodB_path = self.root_dir + "/NS.20121127.50726.fits"
        nodB_data = fits.open(nodB_path)[0].data.astype(np.float64)
        nodB = torch.tensor(nodB_data)

        # Read in the Bad Pixel mask
        self.bpm = self.load_bad_pixel_mask()

        data_full = torch.stack([nodA, nodB])  # Creats NxHxW tensor

        # Inpaint bad pixels.  In the future we will simply neglect these pixels
        if inpaint_bad_pixels:
            data_full = self.inpaint_bad_pixels(data_full)
        data = data_full[:, ybounds[0] : ybounds[1], :]
        data = data.permute(0, 2, 1)

        self.pixels = data
        self.index = torch.tensor([0, 1])

    def __getitem__(self, index):
        return (self.index[index], self.pixels[index])

    def __len__(self):
        return len(self.pixels[:, 0, 0])

    def load_bad_pixel_mask(self):
        """Load the global bad pixel mask"""
        bad_pixel_mask_path = "../../zoja/ccdproc/reduced/static/bad_pixel_mask.fits"
        bpm_data = fits.open(bad_pixel_mask_path)[0].data.astype(np.bool)
        return torch.tensor(bpm_data)

    def inpaint_bad_pixels(self, data_tensor):
        """Inpaint the bad pixels
        
        Args:
            data_tensor (tensor of shape NHW): Tensor of data destined for masking 
                in on the HW axis, and reapplied over all elements in batch axis.
        
        """
        smoothed_data = kornia.filters.median_blur(
            data_tensor.unsqueeze(1), (5, 5)
        ).squeeze()
        data_tensor[:, self.bpm] = smoothed_data[:, self.bpm]
        return data_tensor

    def create_nirspec_collection(self):
        """Create a collection of Keck NIRSPEC echelle spectra"""
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
            ims = (
                ccdproc.ImageFileCollection(
                    self.root_dir, keywords=keywords,  # glob_include="NS*.fits",
                )
                .filter(dispers="high")
                .filter(regex_match=True, slitlen="12|24")
            )
        return ims

    def get_unique_objects(self):
        """Return the unique object names from a NIRSPEC collection"""

        objects = np.unique(
            self.nirspec_collection.filter(imagetyp="object")
            .summary["object"]
            .data.data
        )
        return objects

    def label_nods_from_coordinates(self, coords):
        """Label the ABBA nods given Telescope RA, Dec coordinates"""
        nod_dict = {0: "A", 1: "B"}
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(coords)
        y_pred = kmeans.predict(coords)
        # Enforce that the first nod is usually "A"
        if y_pred[0] != 0:
            y_pred = 1 - y_pred
        return [nod_dict[key] for key in y_pred]

    # def label_nirspec_nods(self):
    #    """Label the ABBA nods in-place given a nirspec collection"""
    #    for object in self.unique_objects:
    #        target_subset = self.nirspec_collection.filter(
    #            imagetyp="object", object=object
    #        )
    #        target_table = target_subset.summary[["file", "object", "ra", "dec"]]
    #        coords = np.vstack([target_table[col].data.data for col in ["ra", "dec"]]).T
    #        target_subset.summary["nod"] = self.label_nods_from_coordinates(coords)

