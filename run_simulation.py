from simulation.dgps import sim_outcomes, sim_covariates, propensity_scores
from simulation.estimator import estimate_ate

import argparse
import numpy as np
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt

# parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_simulations', type=int, default=100)
argparser.add_argument('--n', type=int, default=1000)
argparser.add_argument('--p', type=int, default=10)
argparser.add_argument('--alpha', type=float, default=1/2)
argparser.add_argument('--beta', type=int, default=2)
argparser.add_argument('--gamma', type=int, default=4)
argparser.add_argument('--true_ate', type=float, default=1.0)
argparser.add_argument('--cate_type', type=str, default='complex')
argparser.add_argument('--n_folds', type=int, default=5)
argparser.add_argument('--n_estimators', type=int, default=100)
argparser.add_argument('--seed', type=int, default=123)
argparser.add_argument('--n_jobs', type=int, default=None)
argparser.add_argument('--min_samples_leaf', type=int, default=1)
args = argparser.parse_args()


if __name__=='__main__':
    # set seed
    np.random.seed(args.seed)
    estimates_ate = np.zeros(args.num_simulations)
    estimates_ate_var = np.zeros(args.num_simulations)
    estimates_ate_reg_split = np.zeros(args.num_simulations)
    estimates_ate_reg_split_var = np.zeros(args.num_simulations)
    estimates_ate_reg = np.zeros(args.num_simulations)
    estimates_ate_reg_var = np.zeros(args.num_simulations)
    proportion_treated = np.zeros(args.num_simulations)
    # define export file names
    file_name = f'NSim{args.num_simulations}_NObs{args.n}_NVars{args.p}_NFolds{args.n_folds}_Alpha{args.alpha}_Beta{args.beta}_Gamma{args.gamma}_ATE{args.true_ate}_Type{args.cate_type}_NTrees{args.n_estimators}_Seed{args.seed}'
    # add progress bar
    progress_bar = tqdm(total=args.num_simulations)
    for i in range(args.num_simulations):
        # simulate data
        x, w, y = sim_outcomes(n=args.n, p=args.p, alpha=args.alpha, beta=args.beta, gamma=args.gamma, true_ate=args.true_ate, cate_type=args.cate_type)
        # save proportion of treated
        proportion_treated[i] = np.mean(w)
        # estimate ATE
        ate, ate_var, ate_reg_split, ate_reg_split_var, ate_reg, ate_reg_var = estimate_ate(y, w, x, nfolds=args.n_folds, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs, min_samples_leaf=args.min_samples_leaf)
        # save ate estimates
        estimates_ate[i] = ate
        estimates_ate_var[i] = ate_var
        estimates_ate_reg_split[i] = ate_reg_split
        estimates_ate_reg_split_var[i] = ate_reg_split_var
        estimates_ate_reg[i] = ate_reg
        estimates_ate_reg[i] = ate_reg_var
        # update progress bar
        progress_bar.update(1)
    # close progress bar
    progress_bar.close()
    np.savez(f'results/ate_{file_name}.npz', 
             ate=estimates_ate, 
             ate_var=estimates_ate_var,
             ate_reg_split=estimates_ate_reg_split, 
             ate_reg_split_var=estimates_ate_reg_split_var,
             ate_reg=estimates_ate_reg,
             ate_reg_var=estimates_ate_reg_var,
             proportion_treated=proportion_treated, 
             simulation_settings=vars(args))
