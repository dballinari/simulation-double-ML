import numpy as np
from scipy import stats
from typing import Tuple

# Definition of DGPs as in the paper Okasa (2022) [https://arxiv.org/abs/2201.12692] with constant ATE

# Define constants
MIN_COVARIATES = 6

def sim_outcomes(n: int, p: int, alpha: float, beta: int, gamma: int, true_ate: float, cate_type: str='constant') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if p < MIN_COVARIATES:
        raise ValueError(f"Number of covariates must be at least {MIN_COVARIATES}")
    # simulate nxp covariates from a uniform distribution
    x = sim_covariates(n, p)
    # simulate treatment assignment
    w = _sim_treatment_assignment(x, alpha, beta, gamma)
    # simulate outcomes
    y0 = outcomes_not_treated(x)
    y1 = outcomes_treated(x, true_ate, cate_type)
    # observed outcomes
    y = y0*(1-w) + y1*w + np.random.normal(size=x.shape[0])
    return x, w, y

def propensity_scores(x: np.ndarray, alpha: float, beta: int, gamma: int) -> np.ndarray:
    # define sinus function of the product of the first 4 covariates
    f = np.sin(np.prod(x[:,:4], axis=1)*np.pi)
    # propensity scores as beta distribution at f
    ps = alpha*(1 + stats.beta.cdf(f, beta, gamma))
    return ps

def sim_covariates(n: int, p: int, type: str='uniform') -> np.ndarray:
    if type == 'uniform':
        # simulate nxp covariates from a uniform distribution
        x = np.random.uniform(size=(n, p))
    elif type == 'normal':
        # define covariance matrix where entries are 0.7^(|i-j|) for i,j=1,...,p
        cov = np.zeros((p,p))
        for i in range(p):
            for j in range(p):
                cov[i,j] = 0.7**(np.abs(i-j))
        # simulate nxp covariates from a multivariate normal distribution with mean 0 and covariance matrix cov
        x = np.random.multivariate_normal(np.zeros(p), cov, size=n)
    return x

def outcomes_not_treated(x: np.ndarray) -> np.ndarray:
    mu = _base_outcomes(x)
    return mu

def outcomes_treated(x: np.ndarray, true_ate: float, cate_type: str='constant') -> np.ndarray:
    tau = _cate(cate_type)(x)
    mu = true_ate*tau + _base_outcomes(x)
    return mu

def _base_outcomes(x: np.ndarray) -> np.ndarray:
    return np.sin(np.prod(x[:,:2], axis=1)*np.pi) + 2*(x[:,3]-0.5)**2 + 0.5*x[:,4] 


def _cate(type: str, amplitude: int=3):
    # returns CATE with expected value equal to 1
    cates = {
        'complex': lambda x: (1 + (1/(1+np.exp(-20*(x[:,0]-0.5))) - 0.5))*(1 + (1/(1+np.exp(-20*(x[:,1]-0.5))) - 0.5)), # as in the paper https://arxiv.org/abs/1510.04342
        'sine': lambda x: (np.cos(amplitude*x[:,0]*np.pi) + np.sin(amplitude*x[:,1]*np.pi))*amplitude*np.pi/2, # in expectation equal to 2/(amplitude*pi)
        'constant': lambda x: 1,
    }
    if type not in cates.keys():
        raise ValueError(f"CATE type {type} not recognized")
    return cates[type]
            
            

def _sim_treatment_assignment(x: np.ndarray, alpha: float, beta: int, gamma: int) -> np.ndarray:
    # simulate treatment assignment
    ps = propensity_scores(x, alpha, beta, gamma)
    # treatment assignment as bernoulli distribution at ps
    w = np.random.binomial(1, ps)
    return w
    