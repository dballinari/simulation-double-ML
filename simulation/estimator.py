import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple


def regression_prediction(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, **kwargs) -> np.ndarray:
    # fit random forest regression
    model = RandomForestRegressor(**kwargs)
    model.fit(x_train, y_train)
    # predict outcomes
    y_pred = model.predict(x_test)
    return y_pred

def classification_prediction(x_train: np.ndarray, w_train: np.ndarray, x_test: np.ndarray, **kwargs) -> np.ndarray:
    # fit random forest classification
    model = RandomForestClassifier(**kwargs)
    model.fit(x_train, w_train)
    # predict treatment probabilities
    w_pred = model.predict_proba(x_test)[:,1]
    return w_pred

def estimate_ate(y: np.ndarray, w: np.ndarray, x: np.ndarray, nfolds: int=2, **kwargs) -> float:
    # compute pseudo-outcomes
    tau, tau_naive = _estimate_pseudo_outcomes(y, w, x, nfolds, **kwargs)
    # estimate ATE using doubly robust estimator
    ate = np.mean(tau)
    ate_var = np.var(tau)
    # estimate ATE using naive estimator
    ate_naive = np.mean(tau_naive)
    ate_naive_var = np.var(tau)
    return ate, ate_var, ate_naive, ate_naive_var

def _estimate_pseudo_outcomes(y: np.ndarray, w: np.ndarray, x: np.ndarray, nfolds: int=2, **kwargs) -> Tuple[np.ndarray]:
    # function to estimate pseudo-outcomes using cross-fitting
    # split sample into folds
    n = x.shape[0]
    idx = np.random.choice(np.arange(n), size=n, replace=False)
    idx = np.array_split(idx, nfolds)
    # initialize pseudo-outcomes
    tau = np.zeros(n)
    tau_naive = np.zeros(n)
    # loop over folds
    for i in range(nfolds):
        # split sample into train and test
        idx_test = idx[i]
        idx_train = np.concatenate(idx[:i] + idx[(i+1):])
        x_train = x[idx_train,:]
        y_train = y[idx_train]
        w_train = w[idx_train]
        x_test = x[idx_test,:]
        y_test = y[idx_test]
        w_test = w[idx_test]
        # if train and/or test sample have no treated or no non-treated, set tau to nan
        if (np.sum(w_train==1)==0) or (np.sum(w_train==0)==0) or (np.sum(w_test==1)==0) or (np.sum(w_test==0)==0):
            tau[idx_test] = np.nan
            continue
        # predict outcomes using data on the treated
        y_pred_treated = regression_prediction(x_train[w_train==1,:], y_train[w_train==1], x_test, **kwargs)
        # predict outcomes using data on the non-treated
        y_pred_not_treated = regression_prediction(x_train[w_train==0,:], y_train[w_train==0], x_test, **kwargs)
        # predict treatment probabilities
        w_pred = classification_prediction(x_train, w_train, x_test, **kwargs)
        # compute pseudo-outcomes on test set
        tau[idx_test] = y_pred_treated-y_pred_not_treated + w_test*(y_test-y_pred_treated)/(w_pred+1e-10) - (1-w_test)*(y_test-y_pred_not_treated)/(1-w_pred+1e-10)
        # compute naive pseudo-outcome on test set
        tau_naive[idx_test] = y_pred_treated-y_pred_not_treated
    return tau, tau_naive