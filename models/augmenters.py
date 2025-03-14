import numpy as np
import time
import torch
import torch.nn.functional as F 

from scipy.signal import sosfilt, butter

class BatchChannelAugmenter():

    def __init__(self, sfreq: int = 200, n: int = 1, TF_mode: bool=False):
        self.sfreq = sfreq # sampling frequency
        self.n = n # amount of augmentations to apply
        self.scale_amplitude_range = (0.5, 1.5)
        self.noise_gaussian_range = (0., 0.4) 
        self.dc_shift_range = (-4., 4.) # uV, reduced 10->4 due to normalization of input data
        self.zero_masking_range = (0, 151) # samples
        self.time_shift_range = (-50, 50) # samples
        self.bandstop_range = (2.8, 49) # Hz - top: 82.5
        self.bandstop_width = 5. # Hz
        self.TF_mode = TF_mode # Use only time-frequency-relevant augmentations
        self.TF_augmentations = ["aug_add_noise_gaussian", "aug_time_shift", "aug_bandstop_filter"]

    def augment(self, x: np.array) -> np.array:
        # augment a channel n times
        # x shape: (batch_size, num_samples)
        augmented_x = x.copy()
        if self.TF_mode:
            augmentations = [method for method in dir(self) if callable(getattr(self, method)) and method in self.TF_augmentations]
        else:
            augmentations = [method for method in dir(self) if callable(getattr(self, method)) and method.startswith('aug_')]
        sample = np.random.choice(len(augmentations), x.shape[0]) 
        
        if self.n == 2:
            sample2 = np.random.choice(len(augmentations), x.shape[0]) 

        for i in np.unique(sample): # apply augmentations
            if self.n == 1:
                mask = sample == i 
            elif self.n == 2:
                mask = (sample == i) | (sample2 == i)
            augmented_x[mask] = getattr(self, augmentations[i])(augmented_x[mask])

        return augmented_x

    # Augmentation functions. Naming starts with 'aug_'.
    def aug_scale_amplitude(self, x: np.array) -> np.array:
        # sample uniformly from range
        scale = np.random.uniform(*self.scale_amplitude_range, size=(x.shape[0], 1))
        return x * scale

    def aug_add_noise_gaussian(self, x: np.array) -> np.array:
        # add gaussian noise
        sig = np.random.uniform(*self.noise_gaussian_range, size=(x.shape[0], 1))
        noise = np.random.normal(0., sig, size=x.shape)
        return x + noise
    
    def aug_dc_shift(self, x: np.array) -> np.array:
        # add dc shift
        shift = np.random.uniform(*self.dc_shift_range, size=(x.shape[0],1))
        return x + shift
    
    def aug_zero_masking(self, x:np.array) -> np.array:
        # zero out random segment
        length = np.random.randint(*self.zero_masking_range, size=x.shape[0])
        start = np.random.randint(0, x.shape[1] - length + 1)
        end = start + length
        # Create a mask using broadcasting and reshaping
        mask = (np.arange(x.shape[1]) >= start[:, None]) & (np.arange(x.shape[1]) < end[:, None])
        return x * ~mask
    
    def aug_time_shift(self, x: np.array) -> np.array:
        # shift in time
        shift = np.random.randint(*self.time_shift_range, size=x.shape[0])
        for i in range(x.shape[0]):
            x[i] = np.roll(x[i], shift[i])
        return x
    
    def aug_bandstop_filter(self, x):
        start = np.random.uniform(self.bandstop_range[0], self.bandstop_range[1]-self.bandstop_width, size=x.shape[0])
        for i in range(x.shape[0]):
            x[i] = self._bandstop_filter(x[i], self.sfreq, start[i], start[i] + self.bandstop_width)
        return x
        
    def _bandstop_filter(self, signal, fs, lowcut, highcut, order=3):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high], btype='bandstop', output='sos')
        filtered_signal = sosfilt(sos, signal)
        return filtered_signal
