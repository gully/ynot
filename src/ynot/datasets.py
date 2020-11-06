import torch
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import kornia

# custom dataset loader
class FPADataset(Dataset):
    """Read in two AB nods

    Args:
        ybounds (tupe of ints): the :math:`(y_0, y_{max})` bounds for isolated echelle trace

    """

    def __init__(self, ybounds=(425, 510)):
        super().__init__()

        nodA_path = '../../zoja/nsdrp/raw/2012-11-27/NS.20121127.49332.fits'
        nodA_data = fits.open(nodA_path)[0].data.astype(np.float64)
        nodA = torch.tensor(nodA_data)

        nodB_path = '../../zoja/nsdrp/raw/2012-11-27/NS.20121127.50726.fits'
        nodB_data = fits.open(nodB_path)[0].data.astype(np.float64)
        nodB = torch.tensor(nodB_data)

        # Read in the Bad Pixel mask
        bad_pixel_mask_path = '../../zoja/ccdproc/reduced/static/bad_pixel_mask.fits'
        bpm_data = fits.open(bad_pixel_mask_path)[0].data.astype(np.bool)
        bpm = torch.tensor(bpm_data)

        data_full = torch.stack([nodA, nodB])

        data = data_full[:, ybounds[0]:ybounds[1], :]
        data = data.permute(0,2,1)

        self.pixels = data
        self.ind = torch.tensor([0,1])

        # Inpaint bad pixels.  In the future we will simply neglect these pixels
        smoothed_data = kornia.filters.median_blur(data_full.unsqueeze(1), (5,5)).squeeze()
        data_full[:, bpm] = smoothed_data[:, bpm]


    def __getitem__(self, index):
        return (elf.ind[index], self.pixels[index])

    def __len__(self):
        return len(self.pixels[:,0,0])
