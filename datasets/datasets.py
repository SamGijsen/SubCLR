import numpy as np
import torch
from typing import Tuple
import mne

from base.base_dataset import BaseDataset

class TUAB_H5(BaseDataset):
    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)
            
    def _normalize(self, features):
        return (features - self.dataset_mean) / self.dataset_std

    def __getitem__(self, index):
        x = torch.from_numpy(self._normalize(self.features[index])).float()
        y = torch.from_numpy(np.array([self.labels[index]])).float()
        return x, y
    
    def __len__(self):
        return len(self.indices)      

class TUAB_H5_features(BaseDataset):
    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)

    def _normalize(self, features):
        return (features - self.dataset_mean) / self.dataset_std
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.features[index]).float()
        y = torch.from_numpy(np.array([self.labels[index]])).float()
        return x, y

class TUAB_H5_SSL(BaseDataset):
    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)

    def _normalize(self, features):
        return (features - self.dataset_mean) / self.dataset_std

    def __getitem__(self, index):
        x = self._normalize(self.features[index])
        return x
    
    def collate_fn(self, batch):
        x = np.stack(batch)
        
        if x.shape[-1] > self.cfg["model"]["n_time_samples"]:
            multiple = int(x.shape[-1] / self.cfg["model"]["n_time_samples"])
            start = np.random.choice(multiple) * self.cfg["model"]["n_time_samples"]
            x = x[:,:, start : start + self.cfg["model"]["n_time_samples"]]
        
        x1 = self.augmenter.augment(x)
        x2 = self.augmenter.augment(x)
        x = np.stack((x1, x2), axis=1)
        if self.cfg["model"]["convert_to_TF"]:
            x = filter_data(x)
        
        x = torch.from_numpy(x).float()

        return x
    
class TUAB_H5_SubCLR(BaseDataset):
    def __init__(self, cfg, setting):
        super().__init__(cfg, setting)

    def _normalize(self, features):
        return (features - self.dataset_mean) / self.dataset_std

    def __getitem__(self, index):
        x = torch.from_numpy(self._normalize(self.features[index])).float()
        y = torch.from_numpy(np.array([self.subject_ids[index]])).float()
        return x, y
    
    def collate_fn(self, batch):
        x, y = zip(*batch)
        
        x = np.stack(x)
        if self.cfg["training"]["n_augmentations"]:
            x = self.augmenter.augment(x)
        if self.cfg["model"]["convert_to_TF"]:
            x = filter_data(x)
        x = torch.from_numpy(x).float()
        
        y = torch.stack(y).float()

        return x, y
    
    def __len__(self):
        return len(self.features)

def filter_data(x: torch.Tensor, freq: int=200, bands: list=[(1,7), (8, 30), (31, 49)], axis: int=1,
                norm_stats: list=[0.2, 0.085, 0.045]):
    # Unoptimized band-pass filtering. Don't use this for more than an experiment.
    f = [mne.filter.filter_data(x.astype(np.float64), freq, band[0], band[1], verbose="critical", n_jobs=1)/norm_stats[i] for i, band in enumerate(bands)]
    return np.stack(f, axis=axis)