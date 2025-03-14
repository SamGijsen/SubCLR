import mne
import glob
import os
import h5py
import numpy as np
import pandas as pd
import importlib

from scipy.io import loadmat
from copy import deepcopy
from mne_bids import write_raw_bids, BIDSPath


def HBN_fifs_to_dataset() -> None:

    train = False # Train or Test set
    input_data = "RFB" # ["EEG", "RFB", "HC"]
    releases = [i for i in range(1,11)]
    epoch_wise = True
    epoch_length = 10 # seconds
    n_epochs = 30
    n_channels = 104
    n_timepoints = 2000
    required_file_suffix  = "_ses-000_task-rs_proc-cleanfar_epo.fif"
    meta_df = pd.read_csv("phenotype_metadata_filepath/demo_and_diag.csv")

    base_path = "/hbn_bids_path/"
    if input_data == "EEG":
        suffix = "EPOCHS" if epoch_wise else "CHANNELS"
    elif input_data == "HC":
        suffix = "HC_EPOCHS"
        assert epoch_wise == True
    elif input_data == "RFB":
        suffix = "RFB_EPOCHS"
        assert epoch_wise == True

    train_path = base_path + "/data/HBN_200Hz_clean_{}s_{}.h5".format(epoch_length, suffix)
    test_path = base_path + "/data/HBN_200Hz_clean_{}s_{}_test.h5".format(epoch_length, suffix)
    train_IDs = np.load(os.path.join(base_path, "indices/id", "ALL_train_IDs.npy"), allow_pickle=True) 
    test_IDs = np.load(os.path.join(base_path, "indices/id", "ALL_test_IDs.npy"), allow_pickle=True)

    if train: 
        subset_IDs = train_IDs
        save_path = train_path
    else:
        subset_IDs = test_IDs
        save_path = test_path

    df = pd.DataFrame(columns=
        ["Age", "Sex", "NO", "ADHD", "ANX", "SLD", "ASD", "DIS", "COM", "DEP", "OTH",
        "CGAS", "epochs_per_subject"])
    meta_array = np.empty((n_epochs*n_channels*3500, 13))
    epoch_ids = np.empty((n_epochs*n_channels*3500))
    sub_ids = np.empty((n_epochs*n_channels*3500),  dtype='S20')
    eeg = np.empty((3500 * n_epochs, n_channels, n_timepoints))

    total_E = 0 # running amount of epochs loaded
    sub_count = 0

    for i_r, r in enumerate(releases):
        print(f"Release {r}")
        r_path = os.path.join(base_path, f"release_{r}")
        subject_mapping = np.load(
            os.path.join(r_path, "subject_mapping.npy"), allow_pickle=True).item()
        subject_dirs = glob.glob(r_path + "/deriv/sub-*")
        subject_dirs = np.sort(subject_dirs)
        subjects = [sj.split("/")[-1] for sj in subject_dirs]

        for s in range(len(subjects)):
            s_str = subjects[s] # e.g. sub-00000
            s_ID = subject_mapping[s_str.lstrip("sub-")] # e.g. NDARVV473XTY
            s_file = os.path.join(
                    subject_dirs[s], "ses-000", "eeg", f"{s_str}{required_file_suffix}")
            
            if not os.path.exists(s_file): 
                print("Skipping due to no EEG data: ", s_str, s_ID)
                continue # No EEG data file
            if not np.isin(s_ID, meta_df["subject_id_long"]):
                print("Skipping due to no phenotypic data: ", s_str, s_ID)
                continue # No phenotypic data

            # is this necessarily true or false?
            in_subset = np.isin(s_ID, subset_IDs)
            if not in_subset:
                continue

            if input_data == "EEG":
                epochs = mne.read_epochs(s_file, proj=False, preload=True, verbose='error')
                epochs_ne = epochs.get_data(copy=True)
                E = len(epochs)
                # Collect EEG data
                eeg[total_E : total_E + E, :, :] = epochs_ne
            else:
                f_name = "fb-riemann" if (input_data=="RFB") else "hc"
                suffix = "_epoch_wise" if (input_data=="RFB") else ""
                f_path = os.path.join(r_path, "deriv/features", f"{f_name}_features_per_sub{suffix}")
                features = np.load(os.path.join(f_path, f"{s_str}_{f_name}_features.npy"))
                if sub_count == 0: # init eeg array; dependent on selected features
                    eeg = np.empty((3500*n_epochs, *features.shape[1:])) 
                E = features.shape[0]
                eeg[total_E : total_E+E] = features
                
            # Collect labels
            s_meta = meta_df[meta_df["subject_id_long"] == s_ID]
            s_meta = list(s_meta[df.columns[:12].values].values[0])
            s_meta.append(E)

            meta_array[total_E:total_E+E, :] = np.tile(s_meta, (E,1))
            epoch_ids[total_E:total_E+E] = np.arange(E)   
            sub_ids[total_E:total_E+E] = np.repeat(s_ID, E)

            total_E += E
            sub_count += 1

    # trim to correct size
    eeg = eeg[:total_E]
    meta_array = meta_array[:total_E, :]
    epoch_ids = epoch_ids[:total_E]
    sub_ids = sub_ids[:total_E]
    _, sub_idxs = np.unique(sub_ids, return_inverse=True)
    
    dataset_std = np.std(eeg)
    dataset_mean = np.mean(eeg)

    if epoch_wise == False:
        N, C, L = eeg.shape
        meta_array =  np.repeat(meta_array, C, axis=0)
        epoch_ids = np.repeat(epoch_ids, C)
        sub_ids = np.repeat(sub_ids, C)
        sub_idxs = np.repeat(sub_idxs, C)
        eeg = eeg.reshape((N*C, L))
        
    print(meta_array.shape, epoch_ids.shape, sub_ids.shape, sub_idxs.shape, eeg.shape)

    # save to file
    file = h5py.File(save_path, 'w')

    file.create_dataset('features', data=eeg.astype(np.float32))
    file.create_dataset('epoch_ids', data=epoch_ids)
    file.create_dataset('subject_ids', data=sub_idxs)
    file.create_dataset('long_subject_id', data=list(sub_ids))
    file.create_dataset('dataset_std', data=dataset_std)
    file.create_dataset('dataset_mean', data=dataset_mean)
    
    for i, name in enumerate(df.columns.values):
        file.create_dataset(name, data=meta_array[:,i])

    # Low vs High CGAS score
    lf = lambda x: 0 if 0 <= x < 51 else (1 if x > 72 else -999) 
    file.create_dataset('PAT', data=[lf(i) for i in meta_array[:,-2]])
        
    dataset_names = list(file.keys())
    print("Dataset names:", dataset_names)

    file.close()
            

def HBN_features_to_dataset(train: bool=True, feature: str="HC") -> None:

    # train = True # Train or Test set
    # feature = "RFB" # HC, RFB
    
    n_epochs = 30 # max epochs
    releases = [i for i in range(1,11)]
    required_file_suffix  = "_ses-000_task-rs_proc-cleanfar_epo.fif"
    meta_df = pd.read_csv("/hbn_demofile_path/demo_and_diag_061223.csv")

    base_path = "/hbn_bids_path/"

    suffix = "" if train else "_test"
    save_path = base_path + "/data/HBN_200Hz_clean_10s_{}{}.h5".format(feature, suffix)
    train_IDs = np.load(os.path.join(base_path, "indices/id/", "ALL_train_IDs.npy"), allow_pickle=True) 
    test_IDs = np.load(os.path.join(base_path, "indices/id/", "ALL_test_IDs.npy"), allow_pickle=True)

    if train: 
        subset_IDs = train_IDs
    else:
        subset_IDs = test_IDs

    df = pd.DataFrame(columns=
        ["Age", "Sex", "NO", "ADHD", "ANX", "SLD", "ASD", "DIS", "COM", "DEP", "OTH",
        "CGAS"])

    sub_count = 0

    for i_r, r in enumerate(releases):
        print(f"Release {r}")
        r_path = os.path.join(base_path, f"release_{r}")
        subject_mapping = np.load(
            os.path.join(r_path, "subject_mapping.npy"), allow_pickle=True).item()
        subject_dirs = glob.glob(r_path + "/deriv/sub-*")
        subject_dirs = np.sort(subject_dirs)
        subjects = [sj.split("/")[-1] for sj in subject_dirs]
        
        suffix = "hc" if (feature=="HC") else "fb-riemann"
        f_path = os.path.join(base_path, f"release_{r}", f"deriv/features/{suffix}_features_per_sub/")

        for s in range(len(subjects)):
            s_str = subjects[s] # e.g. sub-00000
            s_ID = subject_mapping[s_str.lstrip("sub-")] # e.g. NDARVV473XTY
            s_file = os.path.join(
                    subject_dirs[s], "ses-000", "eeg", f"{s_str}{required_file_suffix}")
            
            if not os.path.exists(s_file): 
                #print("Skipping due to no EEG data: ", s_str, s_ID)
                continue # No EEG data file
            if not np.isin(s_ID, meta_df["subject_id_long"]):
                #print("Skipping due to no phenotypic data: ", s_str, s_ID)
                continue # No phenotypic data

            # is this necessarily true or false?
            in_subset = np.isin(s_ID, subset_IDs)
            if not in_subset:
                continue
            # there is a duplicate in the dataset (NDARNZ792HBN in 4 and 9)
            if sub_count>0:
                if np.isin(s_ID, sub_ids[:sub_count]):
                    print("WARNING! DUPLICATE!", s_ID)
                    continue
            
            # Instead of EEG epochs, load the features
            f = np.load(os.path.join(f_path, f"{s_str}_{suffix}_features.npy"))
            # For HC features, we average across epochs
            if feature == "HC":
                f = np.nanmean(f[:n_epochs],axis=0)
                
            if sub_count==0:
                meta_array = np.empty((3500, 12))
                sub_ids = np.empty((3500),  dtype='S20')
                if feature == "HC":
                    features = np.empty((3500, f.shape[0]))
                else:
                    features = np.empty((3500, *f.shape))
                
            # Collect labels
            s_meta = meta_df[meta_df["subject_id_long"] == s_ID]
            print(list(s_meta[df.columns[:12].values].values[0]))
            s_meta = list(s_meta[df.columns[:12].values].values[0])
            
            features[sub_count] = f
            sub_ids[sub_count] = s_ID
            meta_array[sub_count] = s_meta

            sub_count += 1

    # trim to correct size
    features = features[:sub_count]
    meta_array = meta_array[:sub_count, :]
    sub_ids = sub_ids[:sub_count]
    _, sub_idxs = np.unique(sub_ids, return_inverse=True)

    print(meta_array.shape, sub_ids.shape, sub_idxs.shape, features.shape)

    # save to file
    file = h5py.File(save_path, 'w')

    file.create_dataset('features', data=features)
    file.create_dataset('subject_ids', data=sub_idxs)
    file.create_dataset('long_subject_id', data=list(sub_ids))
    
    for i, name in enumerate(df.columns.values):
        file.create_dataset(name, data=meta_array[:,i])

    # Low vs High CGAS score
    lf = lambda x: 0 if 0 <= x < 51 else (1 if x > 72 else -999) 
    file.create_dataset('PAT', data=[lf(i) for i in meta_array[:,-1]])
        
    dataset_names = list(file.keys())
    print("Dataset names:", dataset_names)

    file.close()

    # updated version march 2025
    # more flexible approach using continuous EEG data (no rejected epochs)
    # However, we now clip amplitudes and detect when to remove start or final epochs
    # Furthermore, we do not subset channels

    channel_subset = [0,1,2,4]
    amplitude_threshold = 600 # uV
    scaling_factor = 1e4 # important for fp16 (or, save bfloat16 for now?)
    # scaling factor is important for cross-dataset comparisons

    train = False # Train or Test set
    input_data = "EEG" # ["EEG", "RFB", "HC"]
    releases = [i for i in range(1,12)]
    epoch_wise = True
    epoch_length = 10 # seconds
    n_epochs = 80
    n_channels = 128
    n_timepoints = 1001
    required_file_suffix  = "_ses-000_task-rs_proc-cleanfar_epo.fif"
    meta_df = pd.read_csv("phenotype_metadata_filepath/demo_and_diag.csv")

    base_path = "/hbn_bids_path/"
    if input_data == "EEG":
        suffix = "EPOCHS" if epoch_wise else "CHANNELS"
    elif input_data == "HC":
        suffix = "HC_EPOCHS"
        assert epoch_wise == True
    elif input_data == "RFB":
        suffix = "RFB_EPOCHS"
        assert epoch_wise == True

    save_path = base_path + "/data/HBN_100Hz_clean_{}s_{}.h5".format(epoch_length, suffix)
    # train_path = base_path + "/data/HBN_100Hz_clean_{}s_{}.h5".format(epoch_length, suffix)
    # test_path = base_path + "/data/HBN_100Hz_clean_{}s_{}_test.h5".format(epoch_length, suffix)
    # train_IDs = np.load(os.path.join(base_path, "indices/id", "ALL_train_IDs.npy"), allow_pickle=True) 
    # test_IDs = np.load(os.path.join(base_path, "indices/id", "ALL_test_IDs.npy"), allow_pickle=True)

    # if train: 
    #     subset_IDs = train_IDs
    #     save_path = train_path
    # else:
    #     subset_IDs = test_IDs
    #     save_path = test_path

    df = pd.DataFrame(columns=
        ["Age", "Sex", "NO", "ADHD", "ANX", "SLD", "ASD", "DIS", "COM", "DEP", "OTH",
        "CGAS", "epochs_per_subject"])
    meta_array = np.empty((n_epochs*n_channels*3500, 13))
    epoch_ids = np.empty((n_epochs*n_channels*3500))
    sub_ids = np.empty((n_epochs*n_channels*3500),  dtype='S20')
    eeg = np.empty((3500 * n_epochs, n_channels, n_timepoints))

    total_E = 0 # running amount of epochs loaded
    sub_count = 0

    for i_r, r in enumerate(releases):
        print(f"Release {r}")
        r_path = os.path.join(base_path, f"release_{r}")
        subject_mapping = np.load(
            os.path.join(r_path, "subject_mapping.npy"), allow_pickle=True).item()
        subject_dirs = glob.glob(r_path + "/deriv/sub-*")
        subject_dirs = np.sort(subject_dirs)
        subjects = [sj.split("/")[-1] for sj in subject_dirs]

        for s in range(len(subjects)):
            s_str = subjects[s] # e.g. sub-00000
            s_ID = subject_mapping[s_str.lstrip("sub-")] # e.g. NDARVV473XTY
            s_file = os.path.join(
                    subject_dirs[s], "ses-000", "eeg", f"{s_str}{required_file_suffix}")
            
            if not os.path.exists(s_file): 
                print("Skipping due to no EEG data: ", s_str, s_ID)
                continue # No EEG data file
            if not np.isin(s_ID, meta_df["subject_id_long"]):
                print("Skipping due to no phenotypic data: ", s_str, s_ID)
                continue # No phenotypic data

            # is this necessarily true or false?
            # in_subset = np.isin(s_ID, subset_IDs)
            # if not in_subset:
            #     continue

            if input_data == "EEG":
                epochs = mne.read_epochs(s_file, proj=False, preload=True, verbose='error')
                # copy and average reference
                epochs_ar = epochs.copy().set_eeg_reference("average")
                epochs_ne = epochs_ar.get_data(copy=True)
                E = len(epochs)

                # STD check; do we need to delete final epochs?
                if E > 3:
                    mean_std = np.mean(np.std(epochs_ne, axis=2))
                    last_std = np.std(epochs_ne[-1])
                    std_flag = last_std > 3 * mean_std
                else:
                    std_flag = False
                
                # Line length check; do we need to delete initial epochs?
                ll_flags = np.zeros(E)
                for epoch_index in range(E):
                    val = np.mean(np.sum(np.abs(np.diff(epochs_ne[epoch_index], axis=1)), axis=1)) > 0.1
                    ll_flags[epoch_index] = int(val)
                    if not val:
                        break

                valid_E = E - np.sum(ll_flags) - (3*int(std_flag))
                end_index = E - (3*int(std_flag))
                start_index = np.sum(ll_flags)

                valid_epochs = epochs_ne[start_index:end_index]
    
                # Collect EEG data
                eeg[total_E : total_E + valid_E, :, :] = valid_epochs
            else:
                raise ValueError("not implemented for v2")
                f_name = "fb-riemann" if (input_data=="RFB") else "hc"
                suffix = "_epoch_wise" if (input_data=="RFB") else ""
                f_path = os.path.join(r_path, "deriv/features", f"{f_name}_features_per_sub{suffix}")
                features = np.load(os.path.join(f_path, f"{s_str}_{f_name}_features.npy"))
                if sub_count == 0: # init eeg array; dependent on selected features
                    eeg = np.empty((3500*n_epochs, *features.shape[1:])) 
                E = features.shape[0]
                eeg[total_E : total_E+E] = features
                
            # Collect labels
            s_meta = meta_df[meta_df["subject_id_long"] == s_ID]
            s_meta = list(s_meta[df.columns[:12].values].values[0])
            s_meta.append(valid_E)

            meta_array[total_E:total_E+valid_E, :] = np.tile(s_meta, (valid_E,1))
            epoch_ids[total_E:total_E+valid_E] = np.arange(valid_E)   
            sub_ids[total_E:total_E+valid_E] = np.repeat(s_ID, valid_E)

            total_E += valid_E
            sub_count += 1

    # trim to correct size
    eeg = eeg[:total_E]
    meta_array = meta_array[:total_E, :]
    epoch_ids = epoch_ids[:total_E]
    sub_ids = sub_ids[:total_E]
    _, sub_idxs = np.unique(sub_ids, return_inverse=True)
    
    dataset_std = np.std(eeg)
    dataset_mean = np.mean(eeg)

    if epoch_wise == False:
        N, C, L = eeg.shape
        meta_array =  np.repeat(meta_array, C, axis=0)
        epoch_ids = np.repeat(epoch_ids, C)
        sub_ids = np.repeat(sub_ids, C)
        sub_idxs = np.repeat(sub_idxs, C)
        eeg = eeg.reshape((N*C, L))
        
    print(meta_array.shape, epoch_ids.shape, sub_ids.shape, sub_idxs.shape, eeg.shape)

    # save to file
    file = h5py.File(save_path, 'w')

    file.create_dataset('features', data=eeg.astype(np.float16))
    file.create_dataset('epoch_ids', data=epoch_ids)
    file.create_dataset('subject_ids', data=sub_idxs)
    file.create_dataset('long_subject_id', data=list(sub_ids))
    file.create_dataset('dataset_std', data=dataset_std)
    file.create_dataset('dataset_mean', data=dataset_mean)
    
    for i, name in enumerate(df.columns.values):
        file.create_dataset(name, data=meta_array[:,i])

    # Low vs High CGAS score
    lf = lambda x: 0 if 0 <= x < 51 else (1 if x > 72 else -999) 
    file.create_dataset('PAT', data=[lf(i) for i in meta_array[:,-2]])
        
    dataset_names = list(file.keys())
    print("Dataset names:", dataset_names)

    file.close()