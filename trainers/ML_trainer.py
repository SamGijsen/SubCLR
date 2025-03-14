import numpy as np
import pandas as pd
import warnings

from utils.utils import score, split_indices_and_prep_dataset, save_cv_results
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning


from models.baseline_models import hc_model_classification, LogReg, LinReg, OVR_LogReg
from models.baseline_models import RF, RFreg, XGBReg, make_RFB

# Suppress ConvergenceWarning globally
warnings.filterwarnings("ignore", category=ConvergenceWarning)

@ignore_warnings(category=ConvergenceWarning)
def linear_eval_cv(
        cfg,
        subjects,
        dataset,
        test_dataset,
        n_train,
        n_val,
        n_test,
        setting,
        world_size,
        n_folds,
        fold,
        ncv_i):

    _, _, _, dataset, test_dataset, sub_ids = split_indices_and_prep_dataset(
        cfg, subjects, dataset, test_dataset, n_train, n_val, n_test, setting, world_size, n_folds, fold, ncv_i)
    target = cfg["training"]["target"]

    # X = dataset.features[:].astype(np.float32)
    # if np.isinf(X).any():
    #     print("!X contains INF!")
    # if np.isnan(X).any():
    #     print("!X contains NAN!")
    # X = np.where(np.isinf(X), np.nan, X)

    # if setting != "RFB":
    #     X = X.reshape(X.shape[0], -1)

    # data and label prep
    y = dataset.labels[:] # n_features, n_labels
    
    multitarget = (y.shape[1] > 1)
    if multitarget:
        model, param_grid = OVR_LogReg
    else:
        if cfg["model"]["n_classes"] == 1:
           model, param_grid = LinReg 
        else:
            model, param_grid = LogReg
    y = y.squeeze()

    averaging = False
    # if averaging:
    #     df = pd.DataFrame(X)
    #     df["subject_ids"] = dataset.subject_ids
    #     df["y"] = y
    #     df = df.groupby("subject_ids").mean()

    #     y = df["y"].values
    #     df = df.drop("y", axis=1)
    #     X = df.values.reshape(-1, X.shape[1])

    #     reference_ids = df.index.values
    #     train_ind = np.where(np.isin(reference_ids, train_ind))[0]
    #     val_ind = np.where(np.isin(reference_ids, val_ind))[0]
    #     test_ind = np.where(np.isin(reference_ids, test_ind))[0]

    #     X_train = X[train_ind]
    #     X_val = X[val_ind]
    #     X_test = X[test_ind]

    #     y_train = y[train_ind]
    #     y_val = y[val_ind]
    #     y_test = y[test_ind]

    # else:
    X_train = dataset.features[dataset.train_epochs].astype(np.float32)
    X_val = dataset.features[dataset.val_epochs].astype(np.float32)
    X_train = np.where(np.isinf(X_train), np.nan, X_train) # map INFs to NANs for imputer
    X_val = np.where(np.isinf(X_val), np.nan, X_val)
    
    if setting == "RFB":
        RFB, X_train = make_RFB(cfg["model"]["in_channels"], X=X_train)
        _, X_val = make_RFB(cfg["model"]["in_channels"], X=X_val)
        model, param_grid = RFB
    else:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
    
    # if setting == "RFB": # Riemann-FB requires features as dataframe
    #     RFB, X = make_RFB(cfg["model"]["in_channels"], X=X)
    #     model, param_grid = RFB
    #     X_train = X.iloc[dataset.train_epochs]
    #     X_val = X.iloc[dataset.val_epochs]
    # else:
    #     X_train = X[dataset.train_epochs]
    #     X_val = X[dataset.val_epochs]
        
    y_train = y[dataset.train_epochs]
    y_val = y[dataset.val_epochs]

    if test_dataset: # Test data and labels from *test_dataset*
        y = test_dataset.labels[:]
        X = test_dataset.features[:].astype(np.float32)
        X = np.where(np.isinf(X), np.nan, X)
        
        if setting == "RFB":
            _, X = make_RFB(cfg["model"]["in_channels"], X=X)
            X_test = X.iloc[test_dataset.test_epochs]
        else:
            X = X.reshape(X.shape[0], -1)
            X_test = X[test_dataset.test_epochs]
            
        y_test = y[test_dataset.test_epochs]

        to_del = determine_invalid_data(y_test)
        if setting == "RFB":
            X_test = X_test.drop(to_del)
        else:
            X_test = np.delete(X_test, to_del, axis=0)
        y_test = np.delete(y_test, to_del, axis=0).squeeze()
        test_ids = np.delete(sub_ids["test"], to_del, axis=0)
    else:
        X_test = dataset.features[dataset.test_epochs].astype(np.float32)
        X_test = np.where(np.isinf(X_test), np.nan, X_test)
        if setting == "RFB":
            _, X_test = make_RFB(cfg["model"]["in_channels"], X=X_test)

        y_test = y[dataset.test_epochs]
        test_ids = sub_ids["test"]

    print(X_train.shape, X_val.shape, X_test.shape)
    
    # PCA ------
    # pca = PCA(n_components=256)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_val = pca.transform(X_val)
    # X_test = pca.transform(X_test)
    # end of PCA -----

    # Do train/evaluation split and grid-search
    fold_indices = np.concatenate((np.ones(len(X_train)), np.zeros(len(X_val)))) 
    cv_setup = PredefinedSplit(test_fold=fold_indices)
    gs = GridSearchCV(model, param_grid, cv=cv_setup, refit=False, n_jobs=cfg["training"]["num_workers"])

    if setting == "RFB":
        gs.fit(pd.concat([X_train, X_val], ignore_index=True), np.concatenate((y_train, y_val)))
    else:
        gs.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))

    allscores=gs.cv_results_['mean_test_score']
    print(allscores, flush=True)

    # Fetch results and find best parameters
    results = gs.cv_results_
    best_index = results['rank_test_score'].argmin()
    best_params = results['params'][best_index]

    if multitarget: # Keep also the 'estimator__' prefix (e.g. estimator__C)
        updated_params = {}
        for k,v in best_params.items():
            name = k.split('__')[-2:]
            name = name[0] + '__' + name[1]
            updated_params[name] = v
        best_params = updated_params
    else:
        best_params = {k.split('__')[-1]: v for k, v in best_params.items()}

    #model[-1].set_params(**best_params)
    # Using best set of parameters, re-fit only to train-set.
    if "probability" in model[-1].get_params():
        model[-1]["probability"] = True
    model[-1].set_params(**best_params)
    if n_train < 500:
        model.fit(X_train, y_train)
    else:
        if setting == "RFB":
            model.fit(pd.concat([X_train, X_val], ignore_index=True), np.concatenate((y_train, y_val)))
        else:
            model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))

    try:
        if multitarget:
            n_iter = model.steps[-1][1].estimators_[0].n_iter_
        else:
            n_iter = model.steps[-1][1].n_iter_
        print("Number of iters:", n_iter, flush=True)
    except:
        n_iter = 0 # Catch closed-form estimators.
    best_params["n_iter"] = n_iter

    # Predict the test-set given the trained model
    try:
        y_pred = model.predict_proba(X_test)[:,1]
    except:
        y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1, dataset.labels[:].shape[1])

    # Go from epoch-level prediction to subject-level
    sub_ys_true, sub_ys_pred, metrics = score(y_test, y_pred, test_ids, cfg["model"]["n_classes"], cfg["training"]["subject_level_prediction"],
    logits=False)

    save_cv_results("SSL_LIN", cfg, sub_ys_true, sub_ys_pred, metrics, 0., best_params, n_train, fold, ncv_i)


@ignore_warnings(category=ConvergenceWarning)
def baseline_cv(
        cfg,
        subjects,
        dataset,
        test_dataset,
        n_train,
        n_val,
        n_test,
        setting,
        world_size,
        n_folds,
        fold,
        ncv_i):

    train_ind, val_ind, test_ind, dataset, test_dataset, sub_ids = split_indices_and_prep_dataset(
        cfg, subjects, dataset, test_dataset, n_train, n_val, n_test, setting, world_size, n_folds, fold, ncv_i)
    test_ind = test_ind.astype(int)

    # ML models should yield shape [n_subjects, n_features]
    X = dataset.features[:].astype(np.float32)
    X = np.where(np.isinf(X), np.nan, X) 
        
    y = dataset.labels[:].squeeze()

    # Map from subject ID to index in dataset
    reference_ids = dataset.subject_ids[:].astype(int)
    train_ind = np.where(np.isin(reference_ids, train_ind))[0]
    val_ind = np.where(np.isin(reference_ids, val_ind))[0]
    
    if setting == "HC":
        model, param_grid = hc_model_classification
        X_train = X[train_ind]
        X_val = X[val_ind]
    elif setting == "RFB":
        HBN = False #("HBN" in cfg["dataset"]["path"])
        RFB, X = make_RFB(cfg["model"]["in_channels"], flex_rank=HBN, X=X)
        model, param_grid = RFB
        X_train = X.iloc[train_ind]
        X_val = X.iloc[val_ind]

    y_train = y[train_ind]
    y_val = y[val_ind]

    if test_dataset: # Test data and labels from *test_dataset*
        y = test_dataset.labels[:]
        X = test_dataset.features[:].astype(np.float32)
        X = np.where(np.isinf(X), np.nan, X) 
        if setting == "RFB":
            _, X = make_RFB(cfg["model"]["in_channels"], flex_rank=HBN, X=X)
            X_test = X.iloc[test_ind]
        else:
            X_test = X[test_ind]
        y_test = y[test_ind]
        
        to_del = determine_invalid_data(y_test)
        if setting == "HC":
            X_test = np.delete(X_test, to_del, axis=0)
        else:
            X_test = X_test.drop(to_del)
        y_test = np.delete(y_test, to_del, axis=0).squeeze()
        test_ind = np.delete(test_ind, to_del, axis=0)

        test_ind = np.where(np.isin(test_dataset.subject_ids[:].astype(int), test_ind))[0]
    else:
        test_ind = np.where(np.isin(reference_ids, test_ind))[0]
        if setting == "RFB":
            X_test = X.iloc[test_ind]
        else:
            X_test = X[test_ind]
        y_test = y[test_ind]
        
    print(X_train.shape, X_val.shape, X_test.shape)
        
    fold_indices = np.concatenate((np.ones(len(X_train)), np.zeros(len(X_val)))) 
    cv_setup = PredefinedSplit(test_fold=fold_indices)
    gs = GridSearchCV(model, param_grid, cv=cv_setup, refit=False, n_jobs=cfg["training"]["num_workers"])

    if setting == "HC":
        gs.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    else:
        gs.fit(pd.concat([X_train, X_val], ignore_index=True), np.concatenate((y_train, y_val)))

    allscores=gs.cv_results_['mean_test_score']
    print("Grid scores: ", allscores, flush=True)

    results = gs.cv_results_
    best_index = results['rank_test_score'].argmin()
    best_params = results['params'][best_index]
    
    model_params = {}
    fb_transformer_n_compo = None
    # Extract n_compo for the filter bank transformer and other parameters for the model
    for k, v in best_params.items():
        if "estimator__" in k:
            model_params[k.replace("estimator__", "")] = v
        elif k == "fb_transformer__projection_params__n_compo":
            fb_transformer_n_compo = v

    if fb_transformer_n_compo is not None:
        if 'fb_transformer' in model.named_steps:
            current_projection_params = model.named_steps['fb_transformer'].get_params().get('projection_params', {})
            if current_projection_params is None:
                current_projection_params = {}
            current_projection_params['n_compo'] = fb_transformer_n_compo
            model.named_steps['fb_transformer'].set_params(projection_params=current_projection_params)

    if "probability" in model.named_steps["estimator"].get_params():
        model_params["probability"] = True
    model.named_steps['estimator'].set_params(**model_params)
            
    # best_params = {k.split('__')[-1]: v for k, v in best_params.items()}
    # model[-1].set_params(**best_params)
    model.fit(X_train, y_train)
    # if setting == "RFB":
    #     model.fit(pd.concat([X_train, X_val], ignore_index=True), np.concatenate((y_train, y_val)))
    # else:
    #     model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    
    y_pred = model.predict_proba(X_test)[:,1]
    
    sub_ys_true, sub_ys_pred, metrics = score(y_test, y_pred, test_ind, cfg["model"]["n_classes"], 
    convert_to_subject=True, logits=False)

    save_cv_results("SSL_LIN", cfg, sub_ys_true, sub_ys_pred, metrics, 0., best_params, n_train, fold, ncv_i)

def subject_level_features(X, func='mean', axis=0):
    aggs = {'mean': np.nanmean, 'median': np.nanmedian}
    return np.vstack([aggs[func](x, axis=axis, keepdims=True) for x in X])

def check_valid(arr: np.array) -> bool:
    """"Check if numpy array contains any nans or infinite values."""
    return not (np.isnan(arr).any() or np.isinf(arr).any())


def determine_invalid_data(labels: np.array) -> np.array:

    # In case of single-label we filter out -999.
    # In case of multi-label we filter out the samples for which no label applies.

    if labels.shape[1] == 1:
        to_del = np.where(labels == -999)[0]
    else:
        to_del = np.where(np.all(labels == 0, axis=1))[0]

    return to_del