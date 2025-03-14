from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Ridge , LogisticRegression
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor

import coffeine
import numpy as np
import pandas as pd
# from sklearn.linear_model import RidgeCV
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.ensemble import RandomForestRegressor

#alpha_grid = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8])
alpha_grid = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8])
#alpha_grid = np.array([1e-6, 1e-4, 1e-2, 1, 1e2, 1e4])
alpha_grid = np.logspace(-6, 5, 45)

# Linear evaluation models
LogReg = ( # (pipeline, grid) for Logistic Regression classifier
    Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("varth", VarianceThreshold()), 
        ("scale", StandardScaler()),
        ("model_LR", LogisticRegression(max_iter=2000, class_weight='balanced'))
        ]),
    {"model_LR__C" : alpha_grid}, 
)

LinReg = ( # (pipeline, grid) for Ridge Regression
    Pipeline([
        ("varth", VarianceThreshold()), 
        ("scale", StandardScaler()),
        ("model_Ridge", Ridge(max_iter=1000))
        ]),
    {"model_Ridge__alpha" : alpha_grid}, 
)

SVR = ( # (pipeline, grid) for SVR
    Pipeline([
        ("varth", VarianceThreshold()), 
        ("scale", StandardScaler()),
        ("model_LinearSVR", LinearSVR(max_iter=1000))
        ]),
    {"model_LinearSVR__C" : alpha_grid}, 
)

# SVC = ( # (pipeline, grid) for SVC
#     Pipeline([
#         ("varth", VarianceThreshold()), 
#         ("scale", StandardScaler()),
#         ("model_LinearSVC", LinearSVC(max_iter=1000))
#         ]),
#     {"model_LinearSVC__C" : alpha_grid}, 
# )

RF = ( # (pipeline, grid) for RF
    Pipeline([
        ("varth", VarianceThreshold()), 
        ("scale", StandardScaler()),
        ("model_RF", RandomForestClassifier(class_weight='balanced'))
        ]),
    # {"model_RF__max_depth" : [4, 8, 16, 32, None],
    #  "model_RF__max_features": ["sqrt", "log2"],
    #  "model_RF__min_samples_split": [2, 5],
    #  "model_RF__n_estimators": [500]}, 
    {"model_RF__max_depth" : [4, 16, None],
     "model_RF__max_features": ["sqrt", "log2"],
     "model_RF__n_estimators": [100]}, 
)

RFreg = ( # (pipeline, grid) for RF
    Pipeline([
        ("varth", VarianceThreshold()), 
        ("scale", StandardScaler()),
        ("model_RF", RandomForestRegressor())
        ]),
    {"model_RF__max_depth" : [4, 16, None],
     "model_RF__max_features": ["sqrt", "log2"],
     "model_RF__n_estimators": [100]}, 
)

XGBReg = (
# (pipeline, grid) for GradientBoosting regressor
    Pipeline([
    ("varth", VarianceThreshold()),
    ("scale", StandardScaler()),
    ("model_GB", XGBRegressor(n_estimators=100, subsample=1.0))
    ]),
    {
        "model_GB__max_depth" : [4, 8, 16, 32, None],
        "model_GB__learning_rate" : [0.05, 0.25],
    } 
)

# Multi-target
OVR_LogReg = ( # (pipeline, grid) for Logistic Regression classifier
    Pipeline([
        ("varth", VarianceThreshold()), 
        ("scale", StandardScaler()),
        ("model_OVR_LR", MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight="balanced")))
        ]),
    {"model_OVR_LR__estimator__C" : alpha_grid}, 
)

# Filterbank model
RFB_config = {  # put other benchmark related config here
    'fb-riemann': {  # it can go in a separate file later
        'frequency_bands': {
            "low": (0.1, 1),
            "delta": (1, 4),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 15.0),
            "beta_low": (15.0, 26.0),
            "beta_mid": (26.0, 35.0),
            "beta_high": (35.0, 49)
        },
        'feature_map': 'fb_covs',
    }
}

from sklearn.base import BaseEstimator, TransformerMixin

class FilterBankTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, names, method='riemann', projection_params=None, vectorization_params=None, kernel=None, combine_kernels=None, categorical_interaction=None):
        self.names = names
        self.method = method
        self.projection_params = projection_params
        self.vectorization_params = vectorization_params
        self.kernel = kernel
        self.combine_kernels = combine_kernels
        self.categorical_interaction = categorical_interaction
        self.filter_bank_transformer = None

    def fit(self, X, y=None):
        # Initialize filter_bank_transformer with current parameters
        self.filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=self.names,
            method=self.method,
            projection_params=self.projection_params,
            vectorization_params=self.vectorization_params,
            categorical_interaction=self.categorical_interaction
        )
        # Fit the transformer
        self.filter_bank_transformer.fit(X, y)
        return self

    def transform(self, X):
        # Transform the data
        return self.filter_bank_transformer.transform(X)
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            keys = parameter.split('__', 1)
            if len(keys) > 1 and hasattr(self, keys[0]):
                # Handle nested parameters
                param_name, sub_param_name = keys
                if param_name == 'projection_params':
                    if self.projection_params is None:
                        self.projection_params = {}
                    self.projection_params[sub_param_name] = value
            else:
                # Handle non-nested parameters
                setattr(self, parameter, value)
        return self

    # def set_params(self, **parameters):
    #     # For setting the parameters dynamically
    #     for parameter, value in parameters.items():
    #         print(parameter, value)
    #         setattr(self, parameter, value)
    #     return self

    def get_params(self, deep=True):
        # Get parameters for this estimator
        return {
            "names": self.names,
            "method": self.method,
            "projection_params": self.projection_params,
            "vectorization_params": self.vectorization_params,
            "categorical_interaction": self.categorical_interaction
        }


def make_RFB(num_channels, freq_bands= RFB_config['fb-riemann']['frequency_bands'], flex_rank=False, X=None):
    
    rank = num_channels - 1

    if flex_rank:
        fb_riemann_model_classification = (Pipeline([
            ("fb_transformer", FilterBankTransformerWrapper(names=list(freq_bands))),
            ("scale", StandardScaler()),
            #('estimator', SVC(kernel='rbf'))
            ("estimator", LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]), {
            "fb_transformer__projection_params__n_compo": [55, 80, rank],
            "estimator__C": alpha_grid,
            # "estimator__C" : [1., 1e2, 1e4], 
            # "estimator__gamma": np.array([1e-4, 1e-2, 1.,]),
        })
        
    else:
        filter_bank_transformer = coffeine.make_filter_bank_transformer(
                    names=list(freq_bands),
                    method='riemann',
                    projection_params=dict(scale='auto', n_compo=rank)
                )
        fb_riemann_model_classification = (Pipeline([
            ("fb_transformer", filter_bank_transformer),
            ("scale", StandardScaler()),
            #('estimator', SVC(kernel='rbf'))
            ("estimator", LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]), {
            "estimator__C": alpha_grid,
            # "estimator__C" : [1., 1e2, 1e4], 
            # "estimator__gamma": np.array([1e-4, 1e-2, 1.,]),
        })
    
    if X is not None:
        covs = [X[s] for s in range(X.shape[0])]
        covs = np.array(covs)
        X_df = pd.DataFrame(
            {band: list(covs[:, ii]) for ii, band in enumerate(freq_bands)}
        )
    else:
        X_df = None

    return fb_riemann_model_classification, X_df


# num_channels = 21
# frequency_bands = RFB_config['fb-riemann']['frequency_bands']
# rank = num_channels - 1
# filter_bank_transformer = coffeine.make_filter_bank_transformer(
#             names=list(frequency_bands),
#             method='riemann',
#             projection_params=dict(scale='auto', n_compo=rank)
#         )

# fb_riemann_model_classification = (
# # (pipeline, grid) for Ridge Regression
# Pipeline([
#     ("fb_transformer", filter_bank_transformer), 
#     ("scale", StandardScaler()),
#     #("model_LR", LogisticRegression(max_iter=1000))
#     ('estimator', SVC(kernel='rbf'))
#         ]),
#    # {"model_LR__C" : alpha_grid}, 
#         "estimator__C" : alpha_grid, 
#         "estimator__gamma": np.array([1e-8, 1e-6, 1e-4, 1e-2, 1., 1e2]),
# )

# Hand-crafted models
hc_model_classification = (
# (pipeline, grid) for RF classifier
    Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("varth", VarianceThreshold()),
    ("scale", StandardScaler()),
    ("estimator", RandomForestClassifier(n_estimators=500))
    ]),
    {
        "estimator__max_depth" : [4, 8, 16, None],
        "estimator__max_features" : ["sqrt", "log2"],
    } 
)


