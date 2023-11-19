from simulation.dgps import simulate_data
from simulation.estimator import estimate_ate
import os
import argparse
import numpy as np
from tqdm import tqdm
import git

# parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_simulations', type=int, default=100)
argparser.add_argument('--n', type=int, default=1000)
argparser.add_argument('--p', type=int, default=10)
argparser.add_argument('--corr', type=float, default=0.0)
argparser.add_argument('--sigma', type=int, default=1)
argparser.add_argument('--mode', type=int, default=1)
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
    true_ate = np.zeros(args.num_simulations)
    # define export file names
    file_name = f'Mode{args.mode}_NSim{args.num_simulations}_NObs{args.n}_NVars{args.p}_NFolds{args.n_folds}_Corr{args.corr}_Sigma{args.sigma}_NTrees{args.n_estimators}_Seed{args.seed}'
    # add progress bar
    progress_bar = tqdm(total=args.num_simulations)
    for i in range(args.num_simulations):
        # simulate data
        x, w, y, ate_true = simulate_data(n=args.n, p=args.p, mode=args.mode, corr=args.corr, sigma=args.sigma)
        # save proportion of treated
        proportion_treated[i] = np.mean(w)
        # save true ATE
        true_ate[i] = ate_true
        # estimate ATE
        ate, ate_var, ate_reg_split, ate_reg_split_var, ate_reg, ate_reg_var = estimate_ate(y, w, x, nfolds=args.n_folds, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs, min_samples_leaf=args.min_samples_leaf)
        # save ate estimates
        estimates_ate[i] = ate
        estimates_ate_var[i] = ate_var
        estimates_ate_reg_split[i] = ate_reg_split
        estimates_ate_reg_split_var[i] = ate_reg_split_var
        estimates_ate_reg[i] = ate_reg
        estimates_ate_reg_var[i] = ate_reg_var
        # update progress bar
        progress_bar.update(1)
    # close progress bar
    progress_bar.close()
    # if results folder does not exist, create it
    if not os.path.exists('results'):
        os.makedirs('results')
    np.savez(f'results/{file_name}.npz', 
             ate=estimates_ate, 
             ate_var=estimates_ate_var,
             ate_reg_split=estimates_ate_reg_split, 
             ate_reg_split_var=estimates_ate_reg_split_var,
             ate_reg=estimates_ate_reg,
             ate_reg_var=estimates_ate_reg_var,
             proportion_treated=proportion_treated, 
             true_ate=true_ate,
             simulation_settings=vars(args),
             git_hash=git.Repo(search_parent_directories=True).head.object.hexsha,
             )
