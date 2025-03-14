import mne
import glob
import os
import argparse
import datetime
import importlib
#import subprocess
import numpy as np
import pandas as pd

from scipy.io import loadmat
from mne_bids import write_raw_bids, BIDSPath


def HBN_to_MNE_events(EO_samples, EC_samples, n_times, delta=5000):

    EO_samples = EO_samples[:5]
    EC_samples = EC_samples[:5]

    codes, extended_samples = [], []

    samples = np.sort(np.concatenate((EO_samples, EC_samples)))
    for sample in samples:
        if sample in EO_samples:
            extra_codes = ["20", "21"]
            extra_samples = [sample, sample+delta]
        else:
            extra_codes = ["30", "31", "32", "33"]
            extra_samples = [sample, sample+delta, sample+delta*2, sample+delta*3]

        for ec, es in zip(extra_codes, extra_samples):
            codes.append(ec)
            extended_samples.append(es)

    mne_events = np.vstack((
        np.array(extended_samples).astype(int),
        np.zeros(len(extended_samples)).astype(int),
        np.array(codes)
    )).T.astype(int)

    event_mapping = {
        "EyesOpen0": 20,
        "EyesOpen1": 21, 
        "EyesClosed0": 30,
        "EyesClosed1": 31,
        "EyesClosed2": 32,
        "EyesClosed3": 33
    }

    try:
        five_blocks = (len(EO_samples)>=5 and len(EC_samples)>=5)
        full_length = (extended_samples[-1] + delta + 1001) < n_times
        complete_run = (five_blocks and full_length)
    except:
        print("Failed to compute run stats.", len(extended_samples))
        complete_run = False

    print(five_blocks, full_length, complete_run, len(EO_samples), len(EC_samples), (extended_samples[-1] + delta + 1001) )
    return mne_events, event_mapping, complete_run   

def HBN_rereference(release):
    # First we interpolate 'flat' channels and then apply an average reference.
    flat_threshold = 1e-15
    min_flat_epochs = 0.1 
    min_n_epochs = 6
    verbose = 'error'

    cfg = importlib.import_module(f"config.HBN.config_HBN_10s_r{release}")

    subjects_df = pd.read_csv(cfg.bids_root + "/participants.tsv", sep="\t")
    subjects = sorted(sub.lstrip("sub-") for sub in subjects_df.participant_id if
                    os.path.exists(os.path.join(cfg.deriv_root, sub, "ses-000", cfg.data_type)))

    for sub_i, sub_str in enumerate(subjects):

        bp_args = dict(root=cfg.deriv_root,
                        subject=sub_str,
                        datatype=cfg.data_type,
                        session="000",
                        check=False,
                        task=cfg.task,
                        processing="clean",
                        suffix="epo")

        bp = BIDSPath(**bp_args)
        bp_out = bp.copy().update(extension=".fif")
        fname = bp.fpath

        # Interpolate flat channels
        # How to determine if they're flat? var(channel) < flat_treshold
        # How to determine if we interpolate? number of flat epochs > ceil(E*min_flat_epochs)
        try:
            eps_init = mne.read_epochs(fname, proj=False, verbose=verbose)
            eps = eps_init.copy()

            d = eps.get_data(copy=True)
            E, C, T = d.shape # number of Epochs, Channels, Timepoints
        except:
            E = 0

        if E >= min_n_epochs:
        
            flat_count = np.zeros(C)
            for epoch in range(E):
                flat_count += np.var(d[epoch,:,:], axis=-1) < flat_threshold
            eps.info["bads"] = list(np.array(eps.ch_names))

            flat_bool = flat_count > int(np.ceil(E * min_flat_epochs))
            
            eps.info["bads"] = list(np.array(eps.ch_names)[flat_bool])
            eps.interpolate_bads(verbose=verbose)

            # Average referencing
            eps.set_eeg_reference('average', projection=True, verbose=verbose).apply_proj(verbose=verbose)
            bp_out = bp.copy().update(
                processing="cleanfar",
                extension=".fif"
            )

            eps.save(bp_out, verbose=verbose, overwrite=True)
        else:
            print(f"Release {release}: Skipping {sub_str}. Number of epochs: {E} (Required: {min_n_epochs})")

def HBN_release_to_BIDS(release):
    # This code takes the HBN resting state data with events as
    # [EO 20 sec -> EC 40 sec] x 5, and creates
    # [EO0 10 sec, EO1 10 sec, EC0 10 sec, EC1 10 sec, EC2 10 sec, EC3 10 sec] x 5
    
    demo = pd.read_csv("/hbn_demofile_path/demo_and_diag_061223.csv")
    bids_dir = f"/hbn_bids_path/release_{release}/"
    source_dir = f"/hbn_extracted_path/release_{release}/"

    subject_dirs = glob.glob(source_dir + "*")
    subjects = [sj.split("/")[-1] for sj in subject_dirs]

    sfreq = 500
    epoch_in_sec = 10
    sub_count = 0

    EO_string = "20  " # HBN convention
    EC_string = "30  "

    subject_mapping = {}

    for s in range(len(subjects)):
        
        sub_id = subjects[s]
        s_dir = subject_dirs[s]
        print("starting ", s_dir)

        # Find data via two steps:
        # step 1. Try to find complete resting-state recording in the mff format.
        # step 2. If step 1 fails, resort to .mat and .csv formats.
        # If both fail, the subject is skipped.

        # step 1.
        if release <= 5:
            mff_paths = glob.glob(os.path.join(s_dir, "EEG", "raw", "mff_format/*"))
            mff_paths.reverse() # start at the 'latest' recording and work backwards
            print("mff paths", mff_paths)

            complete_run = False
            for path in mff_paths:
                try:
                    raw_init = mne.io.read_raw_egi(path, verbose='warning')
                    raw = raw_init.copy()
                    sfreq = raw.info['sfreq']
                except:
                    print("Failed to load raw egi data.")
                    continue

                #raw.drop_channels("Vertex Reference") # drop reference
                try:
                    found_events = mne.find_events(raw, shortest_event=0, verbose='warning')
                except:
                    print("Failed to find events for mff data.")
                    continue
                
                try:
                    EO_idx = found_events[:,-1] == raw.event_id[EO_string]
                    EO_samples = found_events[EO_idx, 0]
                    EC_idx = found_events[:,-1] == raw.event_id[EC_string]
                    EC_samples = found_events[EC_idx, 0]
                except:
                    print("Failed to index EO/EC strings.")
                    continue
                
                try:
                    mne_events, event_mapping, complete_run = HBN_to_MNE_events(
                        EO_samples, EC_samples, raw.n_times, sfreq*epoch_in_sec)
                except:
                    print("Failed to create mne_events.")
                    continue

                if complete_run: # If we have found a complete resting-state run for this subject, continue.
                    raw.load_data() 
                    #raw.apply_function(lambda x: x * 1e-6) 
                    break

        # step 2.
        if release > 5 or (release <= 5 and complete_run == False):
            complete_run = False
            mat_paths = glob.glob(os.path.join(s_dir, "EEG", "raw", "mat_format/*"))
            csv_paths = glob.glob(os.path.join(s_dir, "EEG", "raw", "csv_format/*"))

            print("mat path:", mat_paths, flush=True)
            try:
                mat = loadmat(mat_paths[0])
                eeg = mat["EEG"][0][0][15][:-1, :] # drop reference
                sfreq = mat["EEG"][0][0][11][0][0]
                assert sfreq.is_integer(), "sfreq is not an integer"
                eeg *= 1e-6 # HBN mat data seems to be uV 
                csv_events = pd.read_csv(csv_paths[0])
            except:
                print("Failed to load mat/csv data.")
                continue

            # Event handling based on provided .csv file
            try:
                EO_idx = csv_events["type"]==EO_string
                EO_samples = csv_events["sample"][EO_idx].values
                EC_idx = csv_events["type"]==EC_string
                EC_samples = csv_events["sample"][EC_idx].values
                n_times = eeg.shape[1]
            except:
                print("Failed to find EO and EC strings.")
                continue

            try:
                mne_events, event_mapping, complete_run = HBN_to_MNE_events(
                    EO_samples, EC_samples, n_times, sfreq*epoch_in_sec)
            except:
                print("Failed create mne_events.")
                continue
            
            # Add stimulus channel
            stim_channel = np.zeros((1, n_times))
            for t,c in zip(mne_events[:,0], mne_events[:,-1]):
                try:
                    stim_channel[0,t] = c
                except:
                    print(f"Failed to insert event {c} at time {t}.")

            # Create mne.Raw structure
            try:
                m = mne.channels.make_standard_montage("GSN-HydroCel-128") # 03.2025 128->129
                ch_types = ["eeg"]*len(m.ch_names) + ['stim']
                ch_names = m.ch_names + ['stim']
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                eeg_with_stim = np.vstack([eeg, stim_channel])
                raw = mne.io.RawArray(eeg_with_stim, info, verbose='warning')
            except:
                print("Failed to create RawArray")
                continue

        if complete_run: # Complete resting-state data has been found!

            try:
                raw.add_events(mne_events, replace=True)
                raw.event_id = event_mapping 

                # Overwrite resting-state specific events
                event_desc = {v: k for k, v in event_mapping.items()}
                anno = mne.annotations_from_events(mne_events, raw.info["sfreq"], event_desc)
                raw.set_annotations(anno)
            except:
                print("Failed to add events to Raw.")
                continue
            # eog = ['E126', 'E127', 'E17', 'E14', 'E21']
            # ch_map = {}
            # for ch_type, ch_name in zip(raw.get_channel_types(), raw.ch_names):
            #     if ch_name in eog:
            #         ch_map[ch_name] = 'eog'
            #         print(ch_name, 'eog')
            #     else:
            #         ch_map[ch_name] = ch_type
            #         print(ch_name, ch_type)
            # raw.set_channel_types(ch_map)
            # raw.get_channel_types()

            # Crop to resting state data
            delta = epoch_in_sec * sfreq
            buffer = 2*sfreq # 2 seconds
            #start = max(0, mne_events[0,0] - buffer)
            start = 0
            end = min(mne_events[-1,0] + delta + buffer, raw.n_times)
            try:
                raw.crop(start / sfreq, end / sfreq)
            except:
                print("Failed to crop Raw.")
                continue

            # if age is invalid, set to 150
            try:
                age = demo[demo["subject_id_long"]==sub_id]["Age"].values[0]
            except:
                age = 150.
            if age < 0 or age > 110:
                age = 150.
            
            # 0: m --> 1: m
            # 1: f --> 2: f
            try:
                sex = demo[demo["subject_id_long"]==sub_id]["Basic_Demos,Sex"].values[0] + 1
            except:
                sex = 0
            
            subject_info = {
                "participant_id": str(sub_count).zfill(5),
                "age": age,
                "sex": sex,
                "birthday": (1980, 5, 17), # placeholder
                "handedness": 1,
                "hand": 1
            }

            raw.info["subject_info"] = subject_info
            raw.set_meas_date(datetime.datetime(1990, 5, 7, 0, 0, tzinfo=datetime.timezone.utc))

            bids_path = BIDSPath(
                subject=str(sub_count).zfill(5), 
                session=str(0).zfill(3), 
                task="rs", 
                run=None, 
                root=bids_dir,
                datatype="eeg",
                check=True)
            
            try:
                write_raw_bids(
                    raw,
                    bids_path,
                    events=mne_events,
                    event_id=event_mapping,
                    overwrite=True,
                    allow_preload=True,
                    format="EDF",
                    verbose='error'
                )
            except:
                print("Failed to write_raw_bids.")
                continue

            subject_mapping[str(sub_count).zfill(5)] =  sub_id

            print(f"COMPLETED: # {sub_count} - sub-{str(sub_count).zfill(5)} - {sub_id}")
            sub_count += 1
        else:
            print(f"SKIP: Failed {sub_id} (Would've been # {sub_count} - sub-{str(sub_count).zfill(5)} - )")


    np.save(os.path.join(bids_dir, "subject_mapping.npy"), subject_mapping, allow_pickle=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess HBN EEG data")
    parser.add_argument("-s", "--step", type=int, help="Processing step to perform: 1 (BIDS) 2 (PL) 3 (AR)")
    parser.add_argument("-r", "--release", type=int, help="HBN data release to be used (1-10)")
    # 1. BIDS: using MNE-BIDS structure the dataset in a BIDS-compliant manner
    # 2. PL: uses the mne-bids-PipeLine to do preprocessing
    # 3. AR: interpolates flat channels and finally applies an average-reference

    args = parser.parse_args()

    if args.step == 1:
        HBN_release_to_BIDS(args.release)

    elif args.step == 2:
        print("Not implemented.")

    elif args.step == 3:
        HBN_rereference(args.release)

    else:
        print("Invalid step. Choose from {1, 3}.")
