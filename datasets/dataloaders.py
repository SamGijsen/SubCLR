import numpy as np
from scipy.special import softmax
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
from torch.utils.data.distributed import DistributedSampler

class SubCLR_DistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 subset_indices: Optional[np.ndarray] = None, spb: int = 8, batch_size: int = 256,) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle,
                         seed, drop_last)
        self.subset_indices = subset_indices
        self.spb = spb
        self.bs_per_gpu = batch_size
        self.bs_total = batch_size * self.num_replicas
        self.numpy_rng = np.random.default_rng(seed=seed)
        self.subject_ids = dataset.subject_ids
        assert self.shuffle == True, "This is a train-only sampler, class assumes shuffle==True."
        
        # If no subject-subset is supplied we take the whole dataset
        self.subset_indices = subset_indices
        if subset_indices is not None:
            self.subject_ids = dataset.subject_ids[np.isin(dataset.subject_ids, subset_indices)]
        else:
            self.subject_ids = dataset.subject_ids

        uni_S, uni_S_ne = np.unique(self.subject_ids, return_counts=True)
        
        assert (spb*2) < self.bs_total
        assert self.bs_total/self.spb % 1 == 0

        self.E = int(self.bs_total/self.spb) # Amount of EEG-epochs to sample
        self.V = self.subset_indices[uni_S_ne>=self.E] # Viable subjects
        self.Ev = uni_S_ne[uni_S_ne>=self.E] # Epochs per viable subject
        self.N = len(self.V)

        # map subjects to epoch indices
        self.SE = {subject: list(np.where(self.subject_ids == subject)[0]) for subject in self.V}

        # Check number of batches to run
        self.total_batches = int(np.floor(np.sum(self.Ev) / self.bs_total))

    def __iter__(self):
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
        self.numpy_rng = np.random.default_rng(seed=self.seed+self.epoch)
        indices = []
        for _ in range(self.total_batches):
            indices.extend(self.create_batch_())
        
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        # Following assert will NOT hold in this implementation as we are non-uniquely sampling data.
        # Important if this is ever adapted for use-cases other than SubCLR.
        #assert len(indices) == self.num_samples 

        return iter(indices)
    
    #def __len__(self):
        #return self.num_samples

    def create_batch_(self):
        
        anchor_sub = np.random.choice(np.arange(self.N))
        available_subjects = np.delete(np.arange(self.N), anchor_sub)
        batch_subjects = np.random.choice(available_subjects, size=self.spb-1, replace=False)
        batch_subjects = self.V[np.sort(np.append(batch_subjects, anchor_sub))]
        
        batch = []
        for subject in batch_subjects:
            epoch_ids = self.numpy_rng.choice(self.SE[subject], size=self.E, replace=False)
            batch.extend(epoch_ids)
        return batch
    
class SubCLR_Sampler(Sampler):
    def __init__(self, dataset, subset_indices, spb, batch_size, seed=0):
        self.subset_indices = subset_indices
        self.spb = spb
        self.batch_size = batch_size
        self.base_seed = seed
        self.primary_rng = np.random.default_rng(seed=self.base_seed)
        self.subject_ids = dataset.subject_ids
        
        # If no subject-subset is supplied we take the whole dataset
        self.subset_indices = subset_indices
        if subset_indices is not None:
            self.subject_ids = dataset.subject_ids[np.isin(dataset.subject_ids, subset_indices)]
        else:
            self.subject_ids = dataset.subject_ids

        uni_S, uni_S_ne = np.unique(self.subject_ids, return_counts=True)
        
        assert (spb*2) < self.batch_size
        assert self.batch_size/self.spb % 1 == 0

        self.E = int(self.batch_size/self.spb) # Amount of EEG-epochs to sample
        self.V = self.subset_indices[uni_S_ne>=self.E] # Viable subjects
        self.Ev = uni_S_ne[uni_S_ne>=self.E] # Epochs per viable subject
        self.N = len(self.V)

        # map subjects to epoch indices
        self.SE = {subject: list(np.where(self.subject_ids == subject)[0]) for subject in self.V}

    def __iter__(self):
        new_epoch_seed = self.primary_rng.integers(low=0, high=1e9)
        self.numpy_rng = np.random.default_rng(seed=new_epoch_seed)
        # Check number of batches to run
        self.total_batches = int(np.floor(np.sum(self.Ev) / self.batch_size))
        
        indices = []
        for _ in range(self.total_batches):
            indices.extend(self.create_batch_())
        
        return iter(indices)

    def create_batch_(self):
        
        anchor_sub = np.random.choice(np.arange(self.N))
        available_subjects = np.delete(np.arange(self.N), anchor_sub)
        batch_subjects = np.random.choice(available_subjects, size=self.spb-1, replace=False)
        batch_subjects = self.V[np.sort(np.append(batch_subjects, anchor_sub))]
        
        batch = []
        for subject in batch_subjects:
            epoch_ids = self.numpy_rng.choice(self.SE[subject], size=self.E, replace=False)
            batch.extend(epoch_ids)
        return batch