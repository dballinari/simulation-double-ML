import numpy as np
from scipy import stats
from scipy.special import expit, logit
from typing import Tuple

# Definition of DGPs as in the paper Okasa (2022) [https://arxiv.org/abs/2201.12692] with constant ATE

# Define constants
MIN_COVARIATES = 6

def simulate_data(n: int, p: int, mode: int=1, corr: float=0.0, sigma: float=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if p < MIN_COVARIATES:
         raise ValueError(f"Number of covariates must be at least {MIN_COVARIATES}")

    dgps = {
         1: _dgp1,
         2: _dgp2,
         3: _dgp3,
    }
    if mode not in dgps.keys():
         raise ValueError(f"Mode {mode} not recognized")
    return dgps[mode](n, p, corr, sigma)

def _dgp1(n: int, p: int, corr: float=0.0, sigma: float=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Set-up C in Nie and Wager (2017) [https://arxiv.org/abs/1712.04912]
    Sigma = _get_covariance_matrix(p, corr)
    x = sim_covariates(n, p, type='normal', cov=Sigma)
    e = 1/(1+np.exp(x[:,0]+x[:,1]))
    b = 2*np.log(1+np.exp(x[:,0]+x[:,1]+x[:,2]))
    tau = 1
    w = np.random.binomial(1, e, size=n)
    y = b + (w-0.5) * tau + sigma * np.random.normal(size=x.shape[0])
    ate = tau
    return x, w, y, ate

def _dgp2(n: int, p: int, corr: float=0.0, sigma: float=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Set-up D in Nie and Wager (2017) [https://arxiv.org/abs/1712.04912]
    Sigma = _get_covariance_matrix(p, corr)
    x = sim_covariates(n, p, type='normal', cov=Sigma)
    e = 1/(1+np.exp(-x[:,0])+np.exp(-x[:,1]))
    b = 0.5*np.maximum(x[:,0]+x[:,1]+x[:,2], np.repeat(0, n)) + 0.5*np.maximum(x[:,3]+x[:,4], np.repeat(0, n))
    tau = np.maximum(x[:,0]+x[:,1]+x[:,2], np.repeat(0, n)) - np.maximum(x[:,3]+x[:,4], np.repeat(0, n))
    w = np.random.binomial(1, e, size=n)
    y = b + (w-0.5) * tau + sigma * np.random.normal(size=x.shape[0])
    ate = np.sqrt(np.sum(np.dot(Sigma[:3,:3], np.ones(3)))/(2*np.pi))\
          - np.sqrt(np.sum(np.dot(Sigma[3:5,3:5], np.ones(2)))/(2*np.pi))
    return x, w, y, ate

def _dgp3(n: int, p: int, corr: float=0.0, sigma: float=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Adapted set-up D in Nie and Wager (2017) [https://arxiv.org/abs/1712.04912]
    Sigma = _get_covariance_matrix(p, corr)
    x = sim_covariates(n, p, type='normal', cov=Sigma)
    e = 1/(1+np.exp(-x[:,0])+np.exp(-x[:,1]))
    b = 0.5*np.maximum(x[:,0]+x[:,1]+x[:,2], np.repeat(0, n))
    tau = np.maximum(x[:,0]+x[:,1]+x[:,2], np.repeat(0, n)) - np.maximum(x[:,3]+x[:,4], np.repeat(0, n))
    w = np.random.binomial(1, e, size=n)
    y = b + (w-0.5) * tau + sigma * np.random.normal(size=x.shape[0])
    ate = np.sqrt(np.sum(np.dot(Sigma[:3,:3], np.ones(3)))/(2*np.pi))\
          - np.sqrt(np.sum(np.dot(Sigma[3:5,3:5], np.ones(2)))/(2*np.pi))
    return x, w, y, ate

def _get_covariance_matrix(p: int, corr: float=0.0) -> np.ndarray:
    # define covariance matrix where entries are 0.7^(|i-j|) for i,j=1,...,p
    i, j = np.indices((p, p))
    Sigma = corr ** np.abs(i - j)
    return Sigma


def sim_covariates(n: int, p: int, type: str='uniform', cov: np.array=None) -> np.ndarray:
    if type == 'uniform':
        # simulate nxp covariates from a uniform distribution
        x = np.random.uniform(size=(n, p))
    elif type == 'normal':
        if cov is None:
            cov = np.eye(p)
        # simulate nxp covariates from a multivariate normal distribution with mean 0 and covariance matrix cov
        x = np.random.multivariate_normal(np.zeros(p), cov, size=n)
    return x
    