import torch
from torch.utils.data import Dataset
import ynot
from astropy.io import fits
import pandas as pd

# custom dataset loader
class FPADataset(Dataset):
    """
    Focal Plane Array dataset

    Args:
        dark_paths (bool): Analyze Darks
        source_metadata (DataFrame): A dataframe with header info for each file

    Only darks are supported right now!
    """

    def __init__(
        self,
        source_metadata = None,
        n_ims=None,
        **kwargs
        ):
        df_all = pd.read_feather('../../zoja/data/J_band_NIRSPEC/calibration_metadata.feather')
        df_all = df_all[df_all.IMAGETYP == 'flatlampoff']

        df_all.local_path = df_all.local_path.str.replace('../../', '../../zoja/')
        list_of_darks = df_all.local_path.values
        if n_ims is None:
            n_ims = len(list_of_darks)
        self.images = torch.empty(n_ims, 1024, 1024,dtype=torch.float64)
        for i, fn in enumerate(list_of_darks[0:n_ims]):
            self.images[i,:,:] = torch.from_numpy(fits.open(fn)[0].data).to(torch.float64)

        # Filter out mis-labeled images
        device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
        device_median = self.images.view((n_ims, 1024*1024)).to(device).median(-1).values.cpu()
        mask = device_median > 100 # These are all arc-frames with erroneous labels
        self.images = self.images[~mask, :, :]
        df_all = df_all.iloc[0:n_ims]
        df_all = df_all[~mask.numpy()].reset_index(drop=True)

        # Use the cold head temperature as proxy for detector temp.
        det_temps = df_all.CRYOTEMP.values

        # Exposure time
        exp_times = df_all.ITIME.values

        # Combined x'es
        self.xs = torch.stack((torch.from_numpy(det_temps),
                             torch.from_numpy(exp_times)),1).to(torch.float64)

        #del device_median
        #torch.cuda.empty_cache()


    def __getitem__(self, index):
        return (self.xs[index,:], self.images[index,:,:])

    def __len__(self):
        return len(self.images)
