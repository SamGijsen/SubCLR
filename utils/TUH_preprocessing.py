import os
import sys
import argparse
import glob
from joblib import Parallel, delayed
from mne_bids import BIDSPath
import numpy as np
import mne

from TUH_utils import load_bv, preprocess_tuh, prep_metadata

def rereference_wrapper(cfg_name: str, pre_param: dict) -> None:

    cfg, subjects = prep_metadata(cfg_name) 
  
    if pre_param["debug"]:
        subjects = subjects[:2]

    Parallel(n_jobs=pre_param["n_jobs"])(
        delayed(rereference_subject)(subject, cfg) for subject in subjects
    )

    return None    

def rereference_subject(subject: str, cfg: dict):

    bp_args = dict(root=cfg.deriv_root,
                    subject=subject,
                    datatype=cfg.data_type,
                    session=cfg.session.lstrip("ses-"),
                    check=False,
                    task=cfg.task,
                    processing="clean",
                    suffix="epo")
    
    bp = BIDSPath(**bp_args)
    bp_out = bp.copy().update(extension=".fif")

    fname = bp.fpath

    epochs = mne.read_epochs(fname, proj=False)
    montage = mne.channels.make_standard_montage("standard_1005")
    epochs.set_montage(montage)
    epochs.pick_channels(cfg.analyze_channels)

    epochs.set_eeg_reference('average', projection=True).apply_proj()
    
    bp_out = bp.copy().update(
        processing="cleanar",
        extension=".fif"
    )

    epochs.save(bp_out, overwrite=True)

    return None


def preproc_wrapper(pre_param: dict) -> None:
    """
    Wrapper function for preprocessing TUH EEG data.

    Input:
        pre_param: Dictionary with preprocessing parameters
            - data_root: path to TUH EEG data (BIDS FORMAT)
            - cropping: [tmin, tmax] in seconds
            - filtering: [l_freq, h_freq] in Hz
            - resampling: in Hz
            - epoch_length: in seconds
            - artefact_method: "autoreject" or None
            - n_jobs: number of parallel jobs
            - verbose: print autoreject output to console
            - debug: only preprocess 2 subjects
    """

    # across subjects
    subject_paths = glob.glob(pre_param["data_root"] + "/sub-*")
    subject_paths.sort()

    if pre_param["debug"]:
        subject_paths = subject_paths[:12]    

    Parallel(n_jobs=pre_param["n_jobs"])(
        delayed(preproc_subject)(pre_param, s_path) for s_path in subject_paths
    )

    return None


def preproc_subject(pre_param: dict, s_path: str):
    """
    Input:
        pre_param: Dictionary with preprocessing parameters

    Output:
        Saves preprocessed data under derivatives path:
            - sub-0XXXX_ses-000_task-rest_eeg-epo.fif.gz
            - sub-0XXXX_ses-000_epo-000_task-rest_eeg.npy
            - sub-0XXXX_ses-000_epo-001_task-rest_eeg.npy
            - ...
    """
    # check if "preprocessing_parameters.txt" already exists
    if os.path.exists(os.path.join(
        pre_param["data_root"], "derivatives", "preprocessing", s_path.split("/")[-1], "eeg", "preprocessing_parameters.txt")):
        print("Skipping subject {} due to existing preprocessing parameters".format(s_path.split("/")[-1]))
        return None

    sub = s_path.split("/")[-1]
    data_input = os.path.join(s_path, "ses-000", "eeg", sub + "_ses-000_task-rest_eeg.vhdr")

    # load data & preprocess
    eeg_f = load_bv(data_input, pre_param["verbose"])

    # Skip participants with too little data
    duration_seconds = eeg_f.n_times / eeg_f.info['sfreq']
    if duration_seconds < pre_param["min_length"]: 
        print("Skipping subject {} due to too little data".format(sub))
        return None

    epochs_clean, bad_epoch_indices, n_bad_epochs = preprocess_tuh(eeg_f, pre_param)

    # save data under derivatives path
    deriv_path = os.path.join(pre_param["data_root"], "derivatives", "preprocessing", sub, "eeg")

    if not os.path.exists(deriv_path):
        os.makedirs(deriv_path)

    # FULL FILE
    # sub-00858_ses-000_task-rest_eeg-epo.fif.gz
    fname = os.path.join(deriv_path, sub + "_ses-000_task-rest_eeg-epo.fif.gz")
    epochs_clean.save(fname, fmt="single", overwrite=True, split_naming="neuromag", verbose=None)
    
    # EPOCHS
    # sub-0XXXX_ses-000_epo-000_task-rest_eeg.npy
    #                       ... 
    # sub-0XXXX_ses-000_epo-059_task-rest_eeg.npy
    epoch_np = epochs_clean.get_data() # shape: (n_epochs, n_channels, n_times)
    for i in range(epoch_np.shape[0]):
        fname = os.path.join(deriv_path, sub + "_ses-000_epo-" + str(i).zfill(3) + "_task-rest_eeg.npy")
        np.save(fname, epoch_np[i, :, :])            

    # also save preprocessing parameters and bad epoch indices
    pre_param["n_bad_epochs"] = n_bad_epochs
    pre_param["bad_epoch_indices"] = bad_epoch_indices

    with open(os.path.join(deriv_path, "preprocessing_parameters.txt"), "w") as f:
        f.write(str(pre_param))

    print("Preprocessed subject {} [SUCCESS]".format(sub))

    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess TUH EEG data")
    parser.add_argument("-root", "--data_root", type=str, help="Path to TUH EEG data (BIDS FORMAT)")
    parser.add_argument("-n", "--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("-v", "--verbose", default=False, action='store_true', help="Print autoreject output to console")
    parser.add_argument("-d", "--debug", default=False, action='store_true', help="Only preprocess 2 subjects")

    args = parser.parse_args()

    # PREPROCESS PARAMETERS
    cfg_name = 'TUAB.config_tuab_eeg_abnormal_60s'
    pre_param = {
        "cropping": [0, 600], # seconds
        "filtering": [0.1, 45], # Hz
        "resampling": 200, # Hz
        "epoch_length": 10, # seconds
        "min_length": 121, # minimum length of recording in seconds
        "artefact_method": "autoreject", # "autoreject" or None
        "rereference": "average" # "average" or None
    }

    pre_param["data_root"] = args.data_root
    pre_param["n_jobs"] = args.n_jobs
    pre_param["verbose"] = args.verbose
    pre_param["debug"] = args.debug

    # preproc_wrapper(pre_param)
    rereference_wrapper(cfg_name, pre_param)
