import numpy as np 
import pandas as pd
import os
import mne
import h5io
import glob
import argparse
import psutil
import coffeine
from mne_features.feature_extraction import extract_features
from joblib import Parallel, delayed
from TUH_utils import prep_metadata
from mne_bids import BIDSPath


def features_wrapper(cfg_name:str, n_jobs: int, n_epochs: int, debug: bool=False, epoch_wise: bool=False, method: str='hc',) -> None:

    cfg, subjects = prep_metadata(cfg_name) 
  
    if debug:
        subjects = subjects[:2]

    features = Parallel(n_jobs=n_jobs)(
        delayed(features_subject)(subject, cfg, n_epochs, epoch_wise, method) for subject in subjects
    )

    out = {sub: ff for sub, ff in zip(subjects, features)
           if not isinstance(ff, str)}

    if not os.path.exists(cfg.deriv_root + "/features/"):
        os.makedirs(cfg.deriv_root + "/features/")
    
    out_fname = cfg.deriv_root + "/features/{}_features.h5".format(method)
    log_fname = cfg.deriv_root + "/features/{}_features.csv".format(method)

    # h5io.write_hdf5(
    #     out_fname,
    #     out,
    #     overwrite=True
    # )
    logging = ["OK" if not isinstance(ff, str) else ff for sub, ff in
                   zip(subjects, features)]
    out_log = pd.DataFrame({"ok": logging, "subject": subjects})
    #out_log.to_csv(log_fname)

    return None   

def features_subject(subject: str, cfg: dict, n_epochs: int, epoch_wise: bool, method: str) -> dict:

    print(f"Subject {subject}", flush=True)
    
    processing = "cleanfar" if "HBN" in cfg.deriv_root else "cleanar"

    bp_args = dict(root=cfg.deriv_root,
                subject=subject,
                datatype=cfg.data_type,
                session=cfg.session.lstrip("ses-"),
                check=False,
                task=cfg.task,
                processing=processing,
                suffix="epo")
    
    bp = BIDSPath(**bp_args)

    try:
        epochs = mne.read_epochs(bp.fpath, proj=False, preload=True, verbose='error')

        # Save subject-specific features as backup
        suffix = "_epoch_wise" if epoch_wise else ""
        fp = cfg.deriv_root + "/features/{0}_reduced_features_per_sub{1}/".format(method, suffix)
        if not os.path.exists(fp):
            os.makedirs(fp)
        out_fname = fp + "sub-{0}_{1}_features.npy".format(subject,method)

        if os.path.exists(out_fname):
            print(f"SKIP {subject}: {out_fname} already exists.")
            return None

        # Compute features
        if method == 'fb-riemann':
            out = extract_fb_covs(epochs[:n_epochs], epoch_wise)
            np.save(out_fname, out)
        elif method == 'hc':
            out = feature_extraction(epochs[:n_epochs])
            np.save(out_fname, out["feats"])
        else:
            raise NotImplementedError("Method '{}' not implemented, choose between [hc, fb-riemann]".format(method))

        return out
    except:
        print(f"SKIP {subject}: Error with {bp.fpath}")
        return None

def feature_extraction(epochs: mne.Epochs) -> dict:
    
    features = extract_features(
        epochs.get_data(),
        epochs.info["sfreq"], 
        hc_selected_funcs,
        funcs_params=hc_func_params,
        n_jobs=1,
        #ch_names=epochs.ch_names,
        return_as_df=False,
    )
    out = {"feats": features}

    return out

def extract_fb_covs(epochs, epoch_wise):
    if epoch_wise:
        fs = []
        for e in range(len(epochs)):
            features, _ = coffeine.compute_features(
                epochs[e], features=('covs',), n_fft=1024, n_overlap=512,
                fs=epochs.info['sfreq'], fmax=49, frequency_bands=frequency_bands)
            fs.append(features["covs"])
        features = np.array(fs)
    else:
        features, _ = coffeine.compute_features(
            epochs, features=('covs',), n_fft=1024, n_overlap=512,
            fs=epochs.info['sfreq'], fmax=49, frequency_bands=frequency_bands)
        features = features["covs"]

    return features

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("-prio", "--priority", action="store_true", help="Ups niceness from 19 to 20.")
    parser.add_argument("-e", "--epochs", type=int, default=999, help="Amount of epochs to consider")
    parser.add_argument("-d", "--debug", default=False, action='store_true', help="Only preprocess 2 subjects")
    parser.add_argument("-ew", "--epoch-wise", default=False, action='store_true', help="Produces RFB features for each epoch.")
    parser.add_argument("-m", "--fe_method", type=str, default='hc', choices=['hc','fb-riemann'], help="Feature extraction method")
    args = parser.parse_args()

    # Set the priority to 19
    current_process = psutil.Process(os.getpid())
    niceness = 20 if args.priority else 19
    current_process.nice(niceness)

    #cfg_name = 'HBN/config_HBN_10s_r1.py' #'config_tuab_eeg_normal_10s_eval'
    # mne features to extract
    # Use features from Engemann et al. (2022), https://doi.org/10.1016/j.neuroimage.2022.119521
    frequency_bands = {
        "low": (0.1, 1),
        "delta": (1, 4),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 15.0),
        "beta_low": (15.0, 26.0),
        "beta_mid": (26.0, 35.0),
        "beta_high": (35.0, 49)
    }

    hc_selected_funcs = [
        'std',
        'kurtosis',
        'skewness',
        'quantile',
        'ptp_amp',
        'mean',
        'pow_freq_bands',
        'spect_entropy',
        'app_entropy',
        'samp_entropy',
        'svd_entropy',
        'hurst_exp',
        'hjorth_complexity',
        'hjorth_mobility',
        'line_length',
        'wavelet_coef_energy',
        'higuchi_fd',
        'zero_crossings',
        'svd_fisher_info'
    ]

    hc_func_params = {
        'quantile__q': [0.1, 0.25, 0.75, 0.9],
        'pow_freq_bands__freq_bands': [0, 2, 4, 8, 13, 18, 24, 30, 49],
        'pow_freq_bands__ratios': 'all',
        'pow_freq_bands__ratios_triu': True,
        'pow_freq_bands__log': True,
        'pow_freq_bands__normalize': None,
    }
    
    hc_selected_funcs = [
        "pow_freq_bands"
    ]
    
    hc_func_params = {
        'pow_freq_bands__freq_bands': [0, 6, 14, 49],
        'pow_freq_bands__ratios': None,
        'pow_freq_bands__ratios_triu': False,
        'pow_freq_bands__log': False,
        'pow_freq_bands__normalize': None,
    }

    # Save feature parameters
    params = {"hc_selected_funcs": hc_selected_funcs, 
              "hc_func_params": hc_func_params, 
              "frequency_bands": frequency_bands}

    for release in range(1,11):
        cfg_name = f'HBN.config_HBN_10s_r{release}'

        print("Release:", release)
    
    # cfgs = ["config_tuab_eeg_normal_10s", "config_tuab_eeg_normal_10s_eval"]
    # for cfg_name in cfgs:
        
        features_wrapper(
            cfg_name=cfg_name, n_jobs=args.n_jobs, n_epochs=args.epochs, debug=args.debug, epoch_wise=args.epoch_wise, method = args.fe_method
            )

    # EXAMPLE USAGE:
    # python feature_extraction.py -n 1
    # For background:
    # nohup python feature_extraction.py -n 4 > ./logs/TUH_fe.log 2>&1 & echo $! > ./logs/TUH_fe.pid &