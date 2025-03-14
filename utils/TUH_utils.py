import numpy as np 
import os
import h5py
import importlib
import mne
import yaml
import glob
import datetime
import os
import ast
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from mne_bids import write_raw_bids, BIDSPath
from autoreject import AutoReject
from edfrd import read_header
from joblib import Parallel, delayed
from types import SimpleNamespace


def best_hp(path: str, n: int) -> dict:
    """Returns the best hyperparameters for a given fold."""

    # file path to best model
    best_model_path = f"{path}/model_best.pt"

    # file path to score file
    path = os.path.dirname(path.rstrip("/"))
    file_name = f"{path}/hp_fold-{n}.csv"

    df = pd.read_csv(file_name)

    # get the best hyperparameters and turn into dict
    best = df.min()["hp_key"]
    best_dict = ast.literal_eval(best)

    return best_dict, best_model_path


def update_score_file(track_val_scores: dict, hp_key: dict, n: int, path: str) -> None:
    """Updates the score file with the new tested hyperparameters and associated validation loss."""

    path = os.path.dirname(path.rstrip("/"))
    file_name = f"{path}/hp_fold-{n}.csv"

    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame(columns=["hp_key", "val_loss"])

    df = pd.concat([df, pd.DataFrame([[hp_key, track_val_scores[tuple(hp_key.items())]]], columns=["hp_key", "val_loss"])], ignore_index=True)
    df.to_csv(file_name, index=False)

    return 

def set_hp(cfg: dict, hp_key: dict, n: int) -> dict:
    """Adds the hyperparameters to the config file."""

    for k in hp_key:

        if k in cfg["model"]:
            cfg["model"][k] = hp_key[k]

        elif k in cfg["training"]:
            cfg["training"][k] = hp_key[k]

        else:
            raise

    cfg["training"]["fold"] = n
    cfg["training"]["hp_key"] = hp_key

    return cfg

def prepare_labels(participants_tsv: str, setting: str) -> np.ndarray:

    subjects_df = pd.read_csv(participants_tsv, sep="\t")
    good_subs = retrieve_good_subjects(participants_tsv)

    if setting == "regression":
        y = subjects_df[subjects_df["participant_id"].isin(good_subs)].age.values
    elif setting == "classification":
        y = subjects_df[subjects_df["participant_id"].isin(good_subs)].sex.values
        y = (y=="F").astype(int)

    return y

def split(subject_indices: np.ndarray, test_size: float = 0.1, val_size: float = 0.1, random_state: int = 0) -> np.ndarray:
    """Splits the data into train, val, test sets.
    Args: 
        subject_indices: array of subject indices
        test_size: size of the test set as portion of subject_indices
        val_size: size of the validation set as portion of training set
        random_state: seed
    """

    train_ids, test_ids = train_test_split(subject_indices, test_size=test_size, random_state=random_state)
    
    # Split train ids into train and val: val size is set to 10% of training size
    val_size = 0.1
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size, random_state=random_state)
    
    return train_ids, val_ids, test_ids

def retrieve_good_subjects(participants_tsv: str) -> np.ndarray:
    """"Specifies routine to retrieve subjects with valid labels."""
    subjects_df = pd.read_csv(participants_tsv, sep="\t")
    good_subs = subjects_df[subjects_df["age"] != 150].participant_id.values
    return good_subs

def dict_from_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

def fifs_to_dataset() -> None:
    """
    Convert epochs from fif files to a dataset of shape (subjects*n_epochs, n_channels, n_times).
    subjects_df should already only contain the subjects that are to be included in the dataset.
    """
    # INPUTS:
    set_type = "eval" # train or eval
    input_data = "EEG" # ["EEG", "RFB", "HC"]
    # ------

    if set_type == "train":
        cfgs = ["config.TUAB.config_tuab_eeg_normal_10s", "config.TUAB.config_tuab_eeg_abnormal_10s"]
    else: 
        cfgs = ["config.TUAB.config_tuab_eeg_normal_10s_eval", "config.TUAB.config_tuab_eeg_abnormal_10s_eval"]
    name_type = "" if set_type=="train" else "_test"
    epoch_wise = True # if True saves [subjects*epochs, channels, time], otherwise [subjects*epochs*channels, time]
    n_epochs = 66
    epoch_length = 10 # seconds
    sets = ["normal", "abnormal"]
    for ds in sets:
        assert ds == "normal" or ds == "abnormal"
        
    if input_data == "EEG":
        suffix = "EPOCHS" if epoch_wise else "CHANNELS"
    elif input_data == "HC":
        suffix = "HC_EPOCHS"
        assert epoch_wise == True
    elif input_data == "RFB":
        suffix = "RFB_EPOCHS"
        assert epoch_wise == True

    if len(sets) == 2:
        path_base = f"/data/TUH/TUAB_BIDS/train/both/data/"
        dataset_path = path_base + "/TUAB_BIDS_TCP_100Hz_clean_both_{}s_{}ne_{}{}.h5".format(epoch_length, n_epochs, suffix, name_type)
    else:
        cfg = importlib.import_module(cfgs[0])
        path_base = cfg.deriv_root + "/data/"
        dataset_path = path_base + "/TUAB_BIDS_TCP_100Hz_clean_{}_{}s_{}ne_{}{}.h5".format(sets[0], epoch_length, n_epochs, suffix, name_type)

    c = dict() # collection dictionary for the datasets
    running_n_subjects = 0
    for i_ds, ds in enumerate(sets):
        c[ds] = {}
    
        # remove subjects with age==150
        cfg = importlib.import_module(cfgs[i_ds])
        subjects_df = pd.read_csv(cfg.bids_root + "/participants.tsv", sep="\t")
        current_N = len(subjects_df)
        subjects_df = subjects_df[subjects_df["age"] != 150]

        c[ds]["epochs_per_subject"] = []

        subjects = sorted(subjects_df.participant_id.values)
        age = subjects_df.age.values
        sex = subjects_df.sex.values
        sex = (sex=="F").astype(int)

        for i, subject in enumerate(subjects):
            bp_args = dict(root=cfg.deriv_root,
                        subject=subject.lstrip("sub-"),
                        datatype=cfg.data_type,
                        session=cfg.sessions[0],
                        check=False,
                        task=cfg.task,
                        processing="cleanar",
                        suffix="epo")
            
            bp = BIDSPath(**bp_args)
            if input_data == "EEG" or i == 0:
                epochs = mne.read_epochs(bp.fpath, proj=False, preload=True)
                copy = epochs.copy()
                copy = mne.set_bipolar_reference(copy,
                                                 anode=['Fp1', 'F7', 'T3', 'T5', 'Fp2', 'F8', 'T4', 'T6', 'T3', 'C3', 'Cz', 'C4', 'Fp1', 'F3', 'C3', 'P3', 'Fp2', 'F4', 'C4', 'P4'],
                                                 cathode=['F7', 'T3', 'T5', 'O1', 'F8', 'T4', 'T6', 'O2', 'C3', 'Cz', 'C4', 'T4', 'F3', 'C3', 'P3', 'O1', 'F4', 'C4', 'P4', 'O2'])
                copy.drop_channels(["A1", "A2", "Fz", "Pz"])
                copy = copy.copy().resample(sfreq=100, npad="auto")
                epochs = copy.copy()
                
            if i == 0: # Construct empty arrays for features and labels
                c[ds]["eeg"] = np.empty((len(subjects) * n_epochs, epochs.get_data().shape[1], epochs.get_data().shape[2]))
                c[ds]["age_ep"] = np.empty(len(subjects) * n_epochs)
                c[ds]["sex_ep"] = np.empty(len(subjects) * n_epochs)
                c[ds]["epoch_id"] = []
                c[ds]["sub_id_ep"] = np.empty(len(subjects) * n_epochs)
            
            if input_data == "EEG":
                epoch_np = epochs.get_data()
                ne = epoch_np.shape[0] if epoch_np.shape[0] < n_epochs else n_epochs
                # compile eeg
                epoch_count = np.sum(c[ds]["epochs_per_subject"]).astype(int)
                c[ds]["eeg"][epoch_count:epoch_count + ne, :, :] = epoch_np[:ne, :, :]
                
            else:
                f_name = "fb-riemann" if (input_data=="RFB") else "HC"
                if input_data == "RFB":
                    f_path = os.path.join(cfg.deriv_root, "features", f"{f_name}_features_per_sub_epoch_wise")
                else:
                    f_path = os.path.join(cfg.deriv_root, "features", f"{f_name}_features_per_sub")
                features = np.load(os.path.join(f_path, f"{subject}_{f_name}_features.npy"))
                
                ne = min(features.shape[0], n_epochs)
                if i == 0: # init eeg array; dependent on selected features
                    c[ds]["eeg"] = np.empty((len(subjects) * n_epochs, *features.shape[1:]))
                epoch_count = np.sum(c[ds]["epochs_per_subject"]).astype(int)
                c[ds]["eeg"][epoch_count: epoch_count+ne] = features[:ne]
                
            # compile labels
            c[ds]["age_ep"][epoch_count:epoch_count + ne] = age[i]
            c[ds]["sex_ep"][epoch_count:epoch_count + ne] = sex[i]
            # compile subject ids
            c[ds]["sub_id_ep"][epoch_count:epoch_count + ne] = int(subject.lstrip("sub-")) + running_n_subjects

            c[ds]["epochs_per_subject"].append(ne)
            c[ds]["epoch_id"].extend(list(range(ne)))

        epoch_count = np.sum(c[ds]["epochs_per_subject"]).astype(int) # include final subject
        # shrink to correct size
        c[ds]["eeg"] = c[ds]["eeg"][:epoch_count]
        c[ds]["age_ep"] = c[ds]["age_ep"][:epoch_count]
        c[ds]["sex_ep"] = c[ds]["sex_ep"][:epoch_count]
        c[ds]["sub_id_ep"] = c[ds]["sub_id_ep"][:epoch_count]
        c[ds]["epoch_id"] = np.array(c[ds]["epoch_id"])
        if ds == "abnormal":
            c[ds]["pathology_ep"] = np.ones(len(c[ds]["age_ep"]))
        elif ds == "normal":
            c[ds]["pathology_ep"] = np.zeros(len(c[ds]["age_ep"]))

        running_n_subjects += current_N

    # Initialize lists to hold the data
    eeg_list = []
    age_ep_list = []
    sex_ep_list = []
    pathology_ep_list = []
    epoch_id_list = []
    sub_id_ep_list = []
    epochs_per_subject_list = []

    for ds in sets:
        # Append the data to the respective lists
        eeg_list.append(c[ds]["eeg"])
        age_ep_list.append(c[ds]["age_ep"])
        sex_ep_list.append(c[ds]["sex_ep"])
        pathology_ep_list.append(c[ds]["pathology_ep"])  # Make sure to adjust the key based on your data
        epoch_id_list.append(c[ds]["epoch_id"])
        sub_id_ep_list.append(c[ds]["sub_id_ep"])
        epochs_per_subject_list.append(c[ds]["epochs_per_subject"])
    del c

    # Concatenate the lists of arrays along the 0th dimension
    eeg = np.concatenate(eeg_list, axis=0)
    age_ep = np.concatenate(age_ep_list, axis=0)
    sex_ep = np.concatenate(sex_ep_list, axis=0)
    pathology_ep = np.concatenate(pathology_ep_list, axis=0)
    epoch_id = np.concatenate(epoch_id_list, axis=0)
    sub_id_ep = np.concatenate(sub_id_ep_list, axis=0)
    epochs_per_subject = np.concatenate(epochs_per_subject_list, axis=0)
    del eeg_list, age_ep_list, sex_ep_list, pathology_ep_list, epoch_id_list, sub_id_ep_list, epochs_per_subject_list

    # delete additional bad subjects
    if set_type == "train":
        to_del = [1102, 1606, 1742]

        # Find indices to delete
        indices_to_delete = np.where(np.isin(sub_id_ep, to_del))[0]

        # Delete corresponding indices from all arrays
        eeg = np.delete(eeg, indices_to_delete, axis=0)
        age_ep = np.delete(age_ep, indices_to_delete, axis=0)
        sex_ep = np.delete(sex_ep, indices_to_delete, axis=0)
        pathology_ep = np.delete(pathology_ep, indices_to_delete, axis=0)
        epoch_id = np.delete(epoch_id, indices_to_delete, axis=0)
        epochs_per_subject = np.delete(epochs_per_subject, np.where(np.isin(np.unique(sub_id_ep), to_del))[0], axis=0)
        sub_id_ep = np.delete(sub_id_ep, indices_to_delete, axis=0)

    # compute mean and std per channel across epochs
    dataset_std = np.std(eeg)
    dataset_mean = np.mean(eeg)

    if epoch_wise == False: # do it channel-wise instead
        N, C, L = eeg.shape
        sub_id_ep = np.repeat(sub_id_ep, C)
        age_ep = np.repeat(age_ep, C)
        sex_ep = np.repeat(sex_ep, C)
        pathology_ep = np.repeat(pathology_ep, C)
        epoch_id = np.repeat(epoch_id, C)
        eeg = eeg.reshape((N*C, L))

    # save to file
    file = h5py.File(dataset_path, 'w')

    file.create_dataset('features', data=eeg)
    file.create_dataset('ages', data=age_ep)
    file.create_dataset('sex', data=sex_ep)
    file.create_dataset("pathology", data=pathology_ep)
    file.create_dataset('subject_ids', data=sub_id_ep)
    file.create_dataset('epochs_per_subject', data=epochs_per_subject)
    file.create_dataset("epoch_ids", data=epoch_id)
    file.create_dataset('dataset_std', data=dataset_std)
    file.create_dataset('dataset_mean', data=dataset_mean)

    file.close()

    return None # python -c 'from TUH_utils import fifs_to_dataset; fifs_to_dataset()'

def prep_metadata(cfg_name: str='config_tuab_eeg'):

    cfg_name = "config." + cfg_name

    cfg_in = importlib.import_module(cfg_name)
    cfg_out = SimpleNamespace(
        bids_root=cfg_in.bids_root,
        deriv_root=cfg_in.deriv_root,
        task=cfg_in.task,
        # analyze_channels=cfg_in.analyze_channels,
        data_type=cfg_in.data_type,
        subjects_dir=cfg_in.subjects_dir,
    )

    cfg_out.conditions = ("rest",)
    cfg_out.feature_conditions = ("rest",)
    cfg_out.session = "ses-" + cfg_in.sessions[0]

    subjects_df = pd.read_csv(cfg_out.bids_root + "/participants.tsv", sep="\t")
    subjects = sorted(sub.lstrip("sub-") for sub in subjects_df.participant_id if
                    os.path.exists(os.path.join(cfg_out.deriv_root, sub, cfg_out.session, cfg_out.data_type)))
    
    return cfg_out, subjects


def fix_tuh_channel_names(name: str) -> str:
    
    # Remove "EEG " and "-REF" from channel names, and fit to standard naming
    if "-REF" in name:
        name = name.replace("EEG ", "").replace("-REF", "").replace("FP", "Fp").replace("Z", "z")
    else: # or, "-LE" in name:
       name = name.replace("EEG ", "").replace("-LE", "").replace("FP", "Fp").replace("Z", "z")
 
    return name


def load_edf(file: str, verbose: bool) -> mne.io.Raw:
    """
    Loads TUH EEG data (.edf files).

    Input:
        file: path to .edf file
    
    Output:
        f: MNE Raw object
    """
    
    # load data
    v = "info" if verbose else "warning"
    f = mne.io.read_raw_edf(file, verbose=v)

    # add channel template-locations for relevant electrodes
    f.rename_channels(fix_tuh_channel_names)
    m = mne.channels.make_standard_montage("standard_1005")
    ch_names = np.intersect1d(f.ch_names, m.ch_names)
    f.pick_channels(ch_names)
    f.set_montage(m)

    return f

def load_bv(file: str, verbose: bool) -> mne.io.Raw:
    """
    Loads TUH EEG data (.edf files).

    Input:
        file: path to .edf file
    
    Output:
        f: MNE Raw object
    """
    
    # load data
    v = "info" if verbose else "warning"
    f = mne.io.read_raw_brainvision(file, 
                                    misc="auto",
                                    scale=1.0,
                                    preload=False,
                                    verbose=v)
    
    channel_names = ['Fp1', # excludes A1 and A2
                    'Fp2',
                    'F3',
                    'F4',
                    'C3',
                    'C4',
                    'P3',
                    'P4',
                    'O1',
                    'O2',
                    'F7',
                    'F8',
                    'T3',
                    'T4',
                    'T5',
                    'T6',
                    'Fz',
                    'Cz',
                    'Pz',
                    'Oz']

    m = mne.channels.make_standard_montage("standard_1005")
    #ch_names = np.intersect1d(f.ch_names, m.ch_names)
    ch_names = np.intersect1d(f.ch_names, channel_names)
    f.pick_channels(ch_names)
    f.set_montage(m)

    f.pick_channels(channel_names)

    return f

def preprocess_tuh(f: mne.io.Raw, p: dict) -> mne.Epochs:
    """
    Preprocesses TUH EEG data.

    Input: 
        f: MNE Raw object
        p: Dictionary with preprocessing parameters
            - cropping: [tmin, tmax] in seconds
            - filtering: [l_freq, h_freq] in Hz
            - resampling: in Hz
            - epoch_length: in seconds
            - artefact_method: "autoreject" or None
            - rereference: "average" or None
            - rereference_only: boolean
            - verbose: boolean

    Output: 
        epochs_clean: MNE Epochs object with preprocessed data
        bad_epoch_indices: list of indices of bad epochs
        n_bad_epochs: number of bad epochs
    """

    # Cropping
    duration_seconds = f.n_times / f.info['sfreq']
    if p["cropping"]:
        if duration_seconds < p["cropping"][1]:
            f.crop(tmin=p["cropping"][0], tmax=duration_seconds-10)
        else:
            f.crop(tmin=p["cropping"][0], tmax=p["cropping"][1])

    # Filtering
    if p["filtering"]:
        f.load_data().filter(l_freq = p["filtering"][0], h_freq = p["filtering"][1], n_jobs=1, verbose="CRITICAL")

    # Resampling
    if p["resampling"]:
        f.resample(sfreq=p["resampling"], n_jobs=1)

    # Epoching
    if p["epoch_length"]:
        v = "info" if p["verbose"] else "warning"
        epochs = mne.make_fixed_length_epochs(f, duration=p["epoch_length"], verbose=v)
        del(f)
    
    # auto-reject
    if p["artefact_method"] == "autoreject":
        ar = AutoReject(random_state=1, cv=5, verbose=p["verbose"])
        epochs_clean = ar.fit_transform(epochs.load_data())
        reject_log = ar.get_reject_log(epochs)
        bad_epoch_indices = [i for i, x in enumerate(reject_log.bad_epochs) if x == True]
        n_bad_epochs = len(bad_epoch_indices)

    else:
        epochs_clean, bad_epoch_indices, n_bad_epochs = epochs, None, None

    if p["rereference"] == "average":
        if p["rereference_only"]:
            epochs_clean = f
            epochs_clean.set_eeg_reference("average", projection=True).apply_proj()
        else:
            epochs_clean.set_eeg_reference("average", projection=True).apply_proj()

    return epochs_clean, bad_epoch_indices, n_bad_epochs


def tuh_to_bids_SESSION(data_file: str, bids_dir: str, subject: int, session: int):
    """
    Converts TUH EEG data (.edf files) to BIDS format.

    Input:
        data_file: path to .edf file
        bids_dir: path to BIDS directory
        subject: subject ID
        session: session ID

    Output:
        None
    """

    eeg_f = load_edf(data_file, verbose=False)

    # grab demographics data from the .edf header
    header = read_header(data_file)
    sex = header[1].split(" ", 1)[1][0]

    try:
        age = int(header[1].split("Age",1)[-1][1:])
        if age < 0: # sometimes is coded as a negative number
            age = 999

    except ValueError:
        age = 999

    # create birthday from age for mne-bids
    #rec_date = datetime.datetime.strptime(header[3], "%d.%m.%y")
    rec_date = eeg_f.info["meas_date"]
    birth_date = rec_date.replace(year=rec_date.year - age)
    birth_date -= datetime.timedelta(weeks=4)

    sex_to_mne = {'n/a': 0, 'm': 1, 'f': 2}

    subject_info = {
        "participant_id": str(subject).zfill(5),
        "age": age,
        "sex": sex_to_mne[sex.lower()],
        "birthday": (birth_date.year, birth_date.month, birth_date.day),
        "handedness": None,
    }

    eeg_f.info["line_freq"] = 60
    eeg_f.info["subject_info"] = subject_info

    # for the subject, loop through "aaaaaaa" etc and turn them into integers?
    bids_path = BIDSPath(
        subject=str(subject).zfill(5), 
        session=str(session).zfill(3), 
        task="rest", 
        run=None, 
        root=bids_dir,
        datatype="eeg",
        check=True)

    write_raw_bids(eeg_f, bids_path, overwrite=True, allow_preload=True,
        format='BrainVision', verbose="ERROR")

    valid = (age!=999)
    return valid


def tuh_to_bids_DATASET(data_dir, bids_root, n_subjects = 1):
    """
    Convert the TUH EEG data to BIDS format using MNE.
    """

    # Original file structure:
    # .../edf/000...150/aaaaaaaa...aaaaaadv/s00*_20**_**_**/01_tcp_ar/aaaaaaaa_s001_t000.edf

    split_folders = os.listdir(data_dir)
    split_folders.sort()

    s_count = {"ar": 0, "le": 0} # subject counts for AR and LE montages
    
    for i, split_folder in enumerate(split_folders):

        subjects = os.listdir(os.path.join(data_dir, split_folder))
        subjects.sort()

        for j, subject in enumerate(subjects):

            sessions = os.listdir(os.path.join(data_dir, split_folder, subject))
            sessions.sort()

            for k, session in enumerate([sessions[0]]): # Only first session

                print(s_count, n_subjects)
            
                montage_name = os.listdir(os.path.join(data_dir, split_folder, subject, session))[0]
                montage = "ar" if "ar" in montage_name else ("le" if "le" in montage_name else None)

                if montage == None:
                    break # we are only collecting ar and le montages
                else:

                    bids_dir = bids_root+ "/TUEG_BIDS_" + montage + "_test"
                    edf_files = os.listdir(os.path.join(data_dir, split_folder, subject, session, montage_name))

                    if not edf_files:
                        break# if there are no .edf files, skip this subject

                    edf_f = os.path.join(data_dir, split_folder, subject, session) + "/" + montage_name + "/" + edf_files[0]
                    # construct path to raw .edf file

                    # convert to BIDS
                    valid = tuh_to_bids_SESSION(edf_f, bids_dir, subject=s_count[montage], session=k)

                    if valid:
                        s_count[montage] += 1 # increment subject count

                    if s_count["ar"] >= n_subjects or s_count["le"] >= n_subjects:
                        return
            

    return None


# TEMPLE UNIVERSITY ABNORMAL EEG CORPUS (TUAB) ------------------------------------
def tuab_to_bids_SESSION(data_file: str, bids_dir: str, subject: int, session: int):
    """
    Converts TUH EEG data (.edf files) to BIDS format.

    Input:
        data_file: path to .edf file
        bids_dir: path to BIDS directory
        subject: subject ID
        session: session ID

    Output:
        None
    """
    eeg_f = load_edf(data_file, verbose=False)

    # grab demographics data from the .edf header
    header = read_header(data_file)
    sex = header[1].split(" ", 1)[1][0]

    try:
        age = int(header[1].split("Age",1)[-1][1:])
        if age < 0: # sometimes is coded as a negative number
            age = 150 # EDFWriter will throw an error if birthdate is <1800

    except ValueError:
        age = 150

    # create birthday from age for mne-bids
    rec_date = eeg_f.info["meas_date"]

    # if rec_date is before 1903, then it is probably wrong
    if rec_date.year < 1903:
        rec_date = rec_date.replace(year=2000)
        
    eeg_f.set_meas_date(rec_date) # attempted fix for mne_bids_pipeline
    birth_date = rec_date.replace(year=rec_date.year - age, day=1)
    birth_date -= datetime.timedelta(weeks=4)
    print("birth_date=",birth_date)

    sex_to_mne = {'n/a': 0, 'm': 1, 'f': 2}

    subject_info = {
        "participant_id": str(subject).zfill(5),
        "age": age,
        "sex": sex_to_mne[sex.lower()],
        "birthday": (birth_date.year, birth_date.month, birth_date.day),
        "handedness": None,
    }

    eeg_f.info["line_freq"] = 60
    eeg_f.info["subject_info"] = subject_info

    bids_path = BIDSPath(
        subject=str(subject).zfill(5), 
        session=str(session).zfill(3), 
        task="rest", 
        run=None, 
        root=bids_dir,
        datatype="eeg",
        check=True)

    write_raw_bids(eeg_f, bids_path, overwrite=True, allow_preload=True,
        format='EDF', verbose="ERROR")

    valid = (age!=150)
    return valid


def tuab_to_bids_DATASET(data_dir, bids_root, n_jobs=1, normal_only=True):
    """
    Convert the TUH EEG data to BIDS format using MNE.
    """

    # Original file structure:
    # TUAB/edf/eval/normal/01_tcp_ar/

    datasets = ["normal"] if normal_only else ["normal", "abnormal"]
    datatypes = ["eval"] #"eval"
    for dataset in datasets:
        for datatype in datatypes:
            
            bids_dir = os.path.join(bids_root, "TUAB_BIDS", datatype, dataset)
            edf_files = glob.glob(os.path.join(data_dir, "edf", datatype, dataset, "01_tcp_ar/*.edf"))

            # convert to BIDS
            print("TOTAL FILES:", len(edf_files))
            for i, file in enumerate(edf_files):
                _ = tuab_to_bids_SESSION(file, bids_dir, subject=i, session=0)
    print("DONE")
            # Parallel(n_jobs=n_jobs)(
            #     delayed(tuab_to_bids_SESSION)(file, bids_dir, subject=i, session=0) for i, file in enumerate(edf_files)
            # )

    return None


def features_to_dataset() -> None:
    """
    Convert epochs from subject-specific feature .npy to a dataset of shape (subjects, n_features).
    subjects_df should already only contain the subjects that are to be included in the dataset.
    """

    feature = "fb-riemann" # "HC", "RFB"
    n_epochs = 66

    set_type = "eval" # train or eval
    name_type = "" if set_type=="train" else "_test"
    sets = ["normal", "abnormal"]
    cfgs = ["config.config_tuab_eeg_normal_10s_eval", "config.config_tuab_eeg_abnormal_10s_eval"]
    for ds in sets:
        assert ds == "normal" or ds == "abnormal"

    if len(sets) == 2:
        path_base = f"/data/TUH/TUAB_BIDS/{set_type}/both/data/"
        dataset_path = path_base + "/TUAB_BIDS_200Hz_clean_both_10s_66ne_{}{}.h5".format(feature, name_type)
    else:
        cfg = importlib.import_module(cfgs[0])
        path_base = cfg.deriv_root + "/data/"
        dataset_path = path_base + "/TUAB_BIDS_200Hz_clean_{}_{}{}.h5".format(sets[0], feature, name_type)

    c = dict() # collection dictionary for the datasets
    running_n_subjects = 0
    for i_ds, ds in enumerate(sets):
        c[ds] = {}
    
        # remove subjects with age==150
        cfg = importlib.import_module(cfgs[i_ds])
        subjects_df = pd.read_csv(cfg.bids_root + "/participants.tsv", sep="\t")
        current_N = len(subjects_df)
        subjects_df = subjects_df[subjects_df["age"] != 150]

        subjects = sorted(subjects_df.participant_id.values)
        age = subjects_df.age.values
        sex = subjects_df.sex.values
        sex = (sex=="F").astype(int)

        for i, subject in enumerate(subjects):
            # Load features
            sub_path = os.path.join(cfg.deriv_root, "features", f"{feature}_features_per_sub", 
                                    f"sub-{subject.lstrip('sub-').zfill(5)}_{feature}_features.npy")
            
            f = np.load(sub_path)

            # For HC features, we average across epochs
            if feature == "HC":
                f = np.nanmean(f[:n_epochs],axis=0)

            if i == 0: # Construct empty arrays for features and labels
                c[ds]["features"] = np.empty((len(subjects), *f.shape))
                c[ds]["age"] = np.empty(len(subjects))
                c[ds]["sex"] = np.empty(len(subjects))
                c[ds]["sub_id"] = np.empty(len(subjects))

            c[ds]["features"][i] = f
            c[ds]["age"][i] = age[i]
            c[ds]["sex"][i] = sex[i]
            c[ds]["sub_id"][i] = int(subject.lstrip("sub-")) + running_n_subjects

        if ds == "abnormal":
            c[ds]["pathology"] = np.ones(len(c[ds]["age"]))
        elif ds == "normal":
            c[ds]["pathology"] = np.zeros(len(c[ds]["age"]))

        running_n_subjects += current_N

    # Initialize lists to hold the data
    features_list, age_list, sex_list, pathology_list, sub_id_list = [], [], [], [], []

    for ds in sets:
        # Append the data to the respective lists
        features_list.append(c[ds]["features"])
        age_list.append(c[ds]["age"])
        sex_list.append(c[ds]["sex"])
        pathology_list.append(c[ds]["pathology"]) 
        sub_id_list.append(c[ds]["sub_id"])
    del c

    # Concatenate the lists of arrays along the 0th dimension
    features = np.concatenate(features_list, axis=0)
    age = np.concatenate(age_list, axis=0)
    sex = np.concatenate(sex_list, axis=0)
    pathology = np.concatenate(pathology_list, axis=0)
    sub_id = np.concatenate(sub_id_list, axis=0)

    # delete additional bad subjects
    if set_type == "train":
        to_del = [1102, 1606, 1742]

        # Find indices to delete
        indices_to_delete = np.where(np.isin(sub_id, to_del))[0]

        print(to_del, indices_to_delete)

        # Delete corresponding indices from all arrays
        features = np.delete(features, indices_to_delete, axis=0)
        age = np.delete(age, indices_to_delete, axis=0)
        sex = np.delete(sex, indices_to_delete, axis=0)
        pathology = np.delete(pathology, indices_to_delete, axis=0)
        sub_id = np.delete(sub_id, indices_to_delete, axis=0)


    # compute mean and std per channel across epochs
    dataset_std = 1.
    dataset_mean = 1.

    # save to file
    print("Creating as ", dataset_path)
    file = h5py.File(dataset_path, 'w')

    file.create_dataset('features', data=features)
    file.create_dataset('ages', data=age)
    file.create_dataset('sex', data=sex)
    file.create_dataset("pathology", data=pathology)
    file.create_dataset('subject_ids', data=sub_id)
    file.create_dataset('dataset_std', data=dataset_std)
    file.create_dataset('dataset_mean', data=dataset_mean)

    file.close()

    return None # python -c 'from TUH_utils import features_to_dataset; features_to_dataset()'
