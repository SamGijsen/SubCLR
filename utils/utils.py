import torch
import os
import pandas as pd
import numpy as np
import yaml
import mne
import random

from sklearn.metrics import mean_absolute_error, r2_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, hamming_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from models.models import *
from datasets.datasets import TUAB_H5, TUAB_H5_SSL, TUAB_H5_features, TUAB_H5_SubCLR
from joblib import Parallel, delayed
from copy import deepcopy
from functools import reduce

def score(ys_true, ys_pred, test_ids, n_classes, convert_to_subject=True, logits=False):
    """ys_true and ys_pred should be 2-dimensional.
    Unilabel: (n_subjects, 1).
    Multilabel: (n_subjects, n_classes)"""

    if ys_true.ndim==1:
        ys_true = ys_true.reshape(-1, 1)
    if ys_pred.ndim==1:
        ys_pred = ys_pred.reshape(-1, 1)

    metrics = []
    metrics_temp = {}
    metrics_temp["1"] = []
    
    n_subjects = len(np.unique(test_ids))
    dims = 1 if n_classes < 3 else n_classes
    sub_ys_true = np.empty((n_subjects, dims))
    sub_ys_pred = np.empty((n_subjects, dims))
    
    if logits:
            ys_pred = torch.sigmoid(torch.tensor(ys_pred, dtype=torch.float)).numpy()
            
    for label in range(dims):
        if convert_to_subject:

            # average per subject (either regression target or, in case of classification, probabilities)
            df = pd.DataFrame({"y_true": ys_true[:,label], "y_pred": ys_pred[:,label], "subject_id": test_ids})
            df_grouped = df.groupby("subject_id")
            df_mean = df_grouped.mean()
            sub_ys_true[:,label] = df_mean["y_true"].values
            sub_ys_pred[:,label] = df_mean["y_pred"].values
        else:
            # sub_ys_true[:,label] = ys_true[:,label]
            # sub_ys_pred[:,label] = ys_pred[:,label]
            sub_ys_true = ys_true
            sub_ys_pred = ys_pred

        if n_classes == 1:
            if np.isnan(np.array(sub_ys_pred)).any():
                print("Scoring: NANs")
                metrics.append(0.)
                metrics.append(0.)
            else:
                metrics.append(mean_absolute_error(sub_ys_true[:, label], sub_ys_pred[:, label]))
                metrics.append(r2_score(sub_ys_true[:, label], sub_ys_pred[:, label]))
        elif n_classes == 2:
             metrics.append(balanced_accuracy_score(sub_ys_true[:, label], (sub_ys_pred[:, label] > 0.5).astype(float)))
             metrics.append(roc_auc_score(sub_ys_true[:, label], sub_ys_pred[:, label]))
        else:
            prec, rec, _ = precision_recall_curve(sub_ys_true[:, label].astype(int), sub_ys_pred[:, label])
            auc_precision_recall = auc(rec, prec)
            metrics_temp["1"].append(auc_precision_recall)
            if label == (dims-1):
                metrics.append(np.mean(metrics_temp["1"]))
                metrics.append(hamming_loss(sub_ys_true, (sub_ys_pred > 0.5).astype(float)))
        
    return sub_ys_true, sub_ys_pred, metrics
    
def majority_vote(ys_true, ys_pred, test_ids):

    ys_pred_bi = (ys_pred>0.5).astype(float) # binarize
    df = pd.DataFrame({"y_true": ys_true, "y_pred": ys_pred_bi, "subject_id": test_ids})

    # Group by subject id
    df_grouped = df.groupby("subject_id")

    # Create new dataframe with majority vote
    df_majority = df_grouped.agg(lambda x: x.value_counts().index[0])

    # Convert to numpy arrays
    ys_true = df_majority["y_true"].values
    ys_pred = df_majority["y_pred"].values

    return ys_true, ys_pred

def best_hp(path: str, ncv_i: int, fold: int, n_train: int) -> dict:
    """Returns the best hyperparameters for a given fold."""

    # file path to score file
    path = os.path.dirname(path.rstrip("/"))
    file_name = f"{path}/hp_ncv-{ncv_i}_fold-{fold}_ntrain-{n_train}.csv"

    df = pd.read_csv(file_name)

    # get the best hyperparameters and turn into dict
    min_idx = df["val_loss"].idxmin()
    df = df.drop(columns=["val_loss", "val_metric"])
    best_dict = df.loc[min_idx].to_dict()

    return best_dict

def set_hp(cfg: dict, hp_key: dict, ncv_i: int, fold: int, n_train: int) -> dict:
    """Adds the hyperparameters to the config file."""

    for k in hp_key:

        if k in cfg["model"]:
            cfg["model"][k] = hp_key[k]

        elif k in cfg["training"]:
            cfg["training"][k] = hp_key[k]

        else:
            raise ValueError("Hyper-grid contains unknown parameters.")

    cfg["training"]["ncv"] = ncv_i
    cfg["training"]["fold"] = fold
    cfg["training"]["n_train"] = n_train
    cfg["training"]["hp_key"] = hp_key

    return cfg

def update_score_file(val_loss: int, val_metric: int, hp_key: dict, ncv_i: int, fold: int, n_train: int, path: str) -> None:
    """Updates the score file with the new tested hyperparameters and associated validation loss."""

    path = os.path.dirname(path.rstrip("/"))
    file_name = f"{path}/hp_ncv-{ncv_i}_fold-{fold}_ntrain-{n_train}.csv"

    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame()
        
    fold_df = pd.DataFrame({**hp_key, "val_loss": val_loss, "val_metric": val_metric}, index=[0])

    df = pd.concat([df, fold_df], ignore_index=True)

    df.to_csv(file_name, index=False)

    return 

def retrieve_good_subjects(participants_tsv: list, return_sex_label: bool=False, SSL_PRE: bool=False) -> np.ndarray:
    """"Specifies routine to retrieve subjects with valid labels."""

    good_subs_list = []
    sex_label_list = []

    running_N = 0
    for tsv in participants_tsv:
        subjects_df = pd.read_csv(tsv, sep="\t")
        good_subs_df = subjects_df[subjects_df["age"] != 150]

        # manual exclusion based on >20% of channels*epochs having std>4sigma
        to_del = [1102, 1606, 1742]

        subjects = good_subs_df.participant_id.values
        subjects = np.array([s.lstrip("sub-") for s in subjects]).astype(int) + running_N
        to_del_ind = np.where(np.isin(subjects, to_del))[0]
        # delete bad subjects
        del_mask = np.isin(subjects, to_del, invert=True)
        subjects = subjects[del_mask]
        # delete from sex label
        sex_label = (good_subs_df.sex.values=="F").astype(int)
        sex_label = np.delete(sex_label, to_del_ind)
        
        good_subs_list.append(subjects)
        sex_label_list.append(sex_label)

        running_N += len(subjects_df)

    good_subs = np.concatenate(good_subs_list)
    sex_label = np.concatenate(sex_label_list)
        
    if return_sex_label:
        return good_subs, sex_label
    
    return good_subs

def dict_from_yaml(file_path: str) -> dict:
    
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        
    if "convert_to_TF" not in yaml_data["model"]:
        yaml_data["model"]["convert_to_TF"] = False

    return yaml_data

def split_indices_and_prep_dataset(
        cfg, subjects, dataset, test_dataset, n_train, n_val, n_test, setting, world_size, n_folds, fold, ncv_i):
    
    val_ss = cfg["dataset"]["val_subsample"]
    test_ss = cfg["dataset"]["test_subsample"]
    target = cfg["training"]["target"]
    salt = cfg["training"]["random_seed"] + 4999*ncv_i
    
    to_stratify = get_stratification_vector(dataset, target, n_train, subset=subjects)
    
    if setting in ["SSL_PRE", "GEN_EMB"]: # Do not subsample and use complete training set.
        train_ind, val_ind, test_ind = subjects, np.array([1]), np.array([1])
        
    elif val_ss: # validation set is provided manually: Skip folding data.
        train_ind = subjects
        ind_path = os.path.join(cfg['dataset']['path'], 'indices', f"{val_ss}_indices.npy") 
        val_ind = np.load(ind_path)
        
    elif n_folds==1: # Skip Cross-Validation
        train_ind, val_ind = train_test_split(subjects, test_size=n_val, stratify=to_stratify, 
                                              random_state=9*n_train + salt)
        
    else: # Do Stratified-K-Fold, with state=n_train to recreate splits.
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=salt)
        for i, (train_index, val_index) in enumerate(skf.split(np.arange(len(to_stratify)), to_stratify)):
            if i == fold: # Grab [fold]
                train_ind, val_ind = train_index, val_index
                # val_ind, train_ind = train_index, val_index #REV

        if n_train < len(train_ind): # If necessary, subsample training set.
            to_stratify_train = to_stratify[train_ind] 
             # avoid n=1 issues by replacing uniquely occurring strings 
            unique_str, counts = np.unique(to_stratify_train, return_counts=True)
            more_than_once = unique_str[counts > 1]
            only_once = unique_str[counts == 1]
            replacement_dict = {key: np.random.choice(more_than_once) for key in only_once}
            to_stratify_train = np.array([replacement_dict.get(i, i) for i in to_stratify_train])
            try: 
                train_ind, _ = train_test_split(train_ind, train_size=n_train, stratify=to_stratify_train,
                                                random_state=99*fold + 9*n_train + salt)
            except:
                train_ind = np.random.choice(train_ind, replace=False, size=n_train)
            
        # If requested, subsample for test-set (rather than using a predefined hold-out test-set).
        if test_ss or test_dataset:
            test_ind = np.array([1]) 
        else:
            assert (n_val+n_test) <= len(val_ind), "Reduce n_val+n_test; too large for number of folds!"
            val_ind, test_ind = train_test_split(val_ind, train_size=n_val, test_size=n_test, stratify=to_stratify[val_ind], 
                                                 random_state=99*fold + 9*n_train + salt)

        # From indices (np.arange) to subject IDs
        train_ind = subjects[train_ind]
        val_ind = subjects[val_ind]
        test_ind = subjects[test_ind]

    if test_ss or test_dataset: # In case we have a seperate test dataset or test subsample.
        ind_path = os.path.join(cfg['dataset']['path'], 'indices', f"{test_ss}_indices.npy")        
        test_ind = np.sort(np.load(ind_path))

    dataset.set_epoch_indices(train_ind, val_ind, test_ind)
    sub_ids = dataset.get_subject_ids(world_size)

    if test_dataset: 
        test_dataset.test_ind = test_ind
        test_dataset.set_epoch_indices(np.arange(1), np.arange(1), test_ind)
        test_sub_ids = test_dataset.get_subject_ids(world_size)
        sub_ids["test"] = test_sub_ids["test"]
        
    print(len(train_ind), len(val_ind), len(test_ind))

    return train_ind, val_ind, test_ind, dataset, test_dataset, sub_ids

def get_stratification_vector(dataset, target: list, n_train: int, subset: np.ndarray=np.array([])):
    # matches = np.isin(dataset.subject_ids, subset)
    # subject_ids = dataset.subject_ids[matches]
    # str_labels = dataset.labels[matches].astype(int).astype(str)
    # labels = reduce(np.char.add, str_labels.T)

    # unique_ids, indices = np.unique(subject_ids, return_index=True)

    # return labels[indices]

    # summarize variables per subject (as we're splitting on subject-lvl)
    # "PAT": dataset.pathology[matches], dataset.pathology

    # This does NOT work for multi-label setting.
    
    if "TUEG" in dataset.file_path:
        include = ["PAT"]
        if len(subset)>0:
            matches = np.isin(dataset.subject_ids, subset)
            df = pd.DataFrame({"PAT": dataset.pathology[matches].astype(int).squeeze(), "subject_id": dataset.subject_ids[matches]})
        else:
            df = pd.DataFrame({"PAT": dataset.pathology.astype(int).squeeze(), "subject_id": dataset.subject_ids})
        df_grouped = df.groupby("subject_id")
        df = df_grouped.mean()
    else: # "HBN" in dataset.file_path or "TUAB" in dataset.file_path:
        include = ["AGE", "SEX", "PAT"]
        if len(subset)>0:
            matches = np.isin(dataset.subject_ids, subset)
            # str_labels = dataset.labels[matches].astype(int)
            # labels = reduce(np.char.add, str_labels.T)
            df = pd.DataFrame({"AGE": dataset.age[matches], "SEX": dataset.sex[matches], "PAT": dataset.pathology[matches].astype(int).squeeze(), "subject_id": dataset.subject_ids[matches]})
        else:
            df = pd.DataFrame({"AGE": dataset.age, "SEX": dataset.sex, "PAT": dataset.pathology.astype(int).squeeze(), "subject_id": dataset.subject_ids})

        df_grouped = df.groupby("subject_id")
        df = df_grouped.mean()
        df["SEX"] = df["SEX"].values.astype(int)
        df["PAT"] = df["PAT"].values.astype(int)
        n_age_bins = 2 if "TUAB" in dataset.file_path else 3

    print("Labels found:", np.unique(df["PAT"].values))

    # determine how to stratify. prioritize pathology > age > sex
    # if n_train <= 3: # only on the target
    #     include = [target]
    #     n_age_bins = 2 if target == "AGE" else 0
    # elif 4 <= n_train <= 7: # target + [pathology or age]
    #     if target in ["AGE", "PAT"]:
    #         include = ["AGE", "PAT"]
    #         n_age_bins = 2
    #     elif target == "SEX":
    #         include = ["SEX", "PAT"]
    #         n_age_bins = 0
    #else: # stratify based on all three
    if "AGE" in include:
        age = pd.Series(df.AGE.values)

        age_bins = pd.qcut(age, q=n_age_bins, labels=["B" + str(i) for i in range(n_age_bins)])
        age_bins = age_bins.values.astype(str)

    if include == ["AGE"]:
        return np.array(age_bins)
    elif include == ["SEX"]:
        return np.array(df.SEX.values.astype(str))
    elif include == ["PAT"]:
        return np.array(df.PAT.values.astype(str))
    elif set(include) == set(["AGE", "SEX"]):
        return np.char.add(age_bins, df.SEX.values.astype(str))
    elif set(include) == set(["AGE", "PAT"]):
        return np.char.add(age_bins, df.PAT.values.astype(str))
    elif set(include) == set(["SEX", "PAT"]):
        return np.char.add(df.SEX.values.astype(str), df.PAT.values.astype(str))
    elif set(include) == set(["SEX", "AGE", "PAT"]):
        return np.char.add(
            np.char.add(age_bins, df.PAT.values.astype(str)), 
            df.SEX.values.astype(str))


def save_cv_results(setting, cfg, ys_true, ys_pred, test_metric, test_loss, hp, n_train, fold, ncv_i):

    results = {
        # "ys_true_sub": ys_true,
        # "ys_pred_sub": ys_pred,
        "MAE/BACC": test_metric[0],
        "R2/AUC": test_metric[1],
        "fold": fold,
        "ncv_i": ncv_i,
        "best_hp": str(hp),
        "test_loss": test_loss
    }
    for i in range(ys_true.shape[1]):
        results["ys_true_sub_l" + str(i)] = ys_true[:,i]
        results["ys_pred_sub_l" + str(i)] = ys_pred[:,i]

    rp = cfg['training']['results_save_path'] + "/" + setting
    if not os.path.exists(rp):
        os.makedirs(rp)

    df = pd.DataFrame(results)
    df.to_csv(f"{rp}/{cfg['model']['model_name']}_ncv_{ncv_i}_fold_{fold}_ntrain_{n_train}.csv")


def load_DDP_state_dict(model, path, device, DDP=False):

    state_dict = torch.load(path, device)

    # if DDP:
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         new_key = "module." + key
    #         new_state_dict[new_key] = value
    #     state_dict = new_state_dict

    model.load_state_dict(state_dict)

    return model

def load_data(cfg, setting):

    def find_correct_dataset(cfg, setting):
        if setting in ["SSL_PRE", "GEN_EMB"]: # SSL channel-wise Pretraining: [n_epochs*n_channels, n_EEG_samples]
            if cfg["training"]["loss_function"] in ["SubCLR"]:
                dataset = TUAB_H5_SubCLR(cfg, setting)
            else:
                dataset = TUAB_H5_SSL(cfg, setting)
        elif setting in ["SSL_FT", "SV"]: # Finetune or Supervise: [n_epochs, n_channels, n_EEG_samples]
            dataset = TUAB_H5(cfg, setting)
        elif setting in ["SSL_NL"]: # Nonlinear eval: [n_epochs, n_channels, n_embedding_samples]
            dataset = TUAB_H5_features(cfg, setting)
        elif setting in ["SSL_LIN"]:
            dataset = []
        return dataset
    
    dataset = find_correct_dataset(cfg, setting)

    # Check whether a seperate training dataset is used. 
    if cfg["dataset"]["test_name"]:
        cfg_test = deepcopy(cfg)
        cfg_test["dataset"]["name"] = cfg_test["dataset"]["test_name"]
        test_dataset = find_correct_dataset(cfg_test, setting)
    else:
        test_dataset = None

    return dataset, test_dataset


def set_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
