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

# custom dataset loader
class FPADataset(Dataset):
    r"""Read in two AB nods of an Echellogram (experimental)

    Args:
        ybounds (tuple of ints): the :math:`(y_0, y_{max})` bounds for isolated echelle trace

    """

    def __init__(self, ybounds=(425, 510)):
        super().__init__()

        nodA_path = "../../zoja/nsdrp/raw/2012-11-27/NS.20121127.49332.fits"
        nodA_data = fits.open(nodA_path)[0].data.astype(np.float64)
        nodA = torch.tensor(nodA_data)

        nodB_path = "../../zoja/nsdrp/raw/2012-11-27/NS.20121127.50726.fits"
        nodB_data = fits.open(nodB_path)[0].data.astype(np.float64)
        nodB = torch.tensor(nodB_data)

        # Read in the Bad Pixel mask
        self.bpm = self.load_bad_pixel_mask()

        data_full = torch.stack([nodA, nodB])  # Creats NHW tensor

        # Inpaint bad pixels.  In the future we will simply neglect these pixels
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
