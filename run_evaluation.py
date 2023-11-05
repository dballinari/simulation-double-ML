import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 

def plot_estimator_distribution(estimator: np.ndarray, true_value: float, ax: plt.Axes, title: str):
    mean_estimator = np.nanmean(estimator)
    std_estimator = np.nanstd(estimator)
    x = np.linspace(-5*std_estimator + mean_estimator, 5*std_estimator + mean_estimator, 100)
    pdf = 1/np.sqrt(2*np.pi*std_estimator)*np.exp(-((x-mean_estimator)/std_estimator)**2/2)
    hist_values, _, _ = ax.hist(estimator, bins=50, alpha=0.5, density=True)
    pdf *= np.max(hist_values)/np.max(pdf)
    ax.plot(x, pdf, 'r--')
    ax.axvline(true_value, color='k', linewidth=1)
    ax.vlines(mean_estimator, ymin=0, ymax=max(pdf), color='r', linestyle='dashed')
    ax.set_title(title)

def evaluate_estimation(ate: np.ndarray, ate_true: float, ate_var: float, n_obs: int, level: float=0.95) -> dict:
    bias = ate - ate_true
    quantile_normal = scipy.stats.norm.ppf((1+level)/2)
    return {
        'mean_bias': np.nanmean(bias),
        'rmse': np.nanstd(bias),
        'mae': np.nanmean(np.abs(bias)),
        'std_estimate': np.nanstd(ate),
        'coverage': np.mean((ate_true > ate - quantile_normal*np.sqrt(ate_var/n_obs)) & (ate_true < ate + quantile_normal*np.sqrt(ate_var/n_obs))),
        'jaque_bera': scipy.stats.jarque_bera(ate)[0],
        'jaque_bera_pvalue': scipy.stats.jarque_bera(ate)[1],
        'kurtosis': scipy.stats.kurtosis(ate),
        'skewness': scipy.stats.skew(ate)
    }


results = np.load('results/ate_1000_2000_20_0.5_2_4_1.0_sine_200_123.npz', allow_pickle=True)
simulation_settings = results.f.simulation_settings.tolist()

print('Double-ML:')
for key, value in evaluate_estimation(results.f.ate, simulation_settings['true_ate'], results.f.ate_var, simulation_settings['n'], level=0.95).items():
    print(f'\t{key}: {value}')
print('\nRegression adjustment sample-splitting')
for key, value in evaluate_estimation(results.f.ate_reg_split, simulation_settings['true_ate'], results.f.ate_reg_split_var, simulation_settings['n'], level=0.95).items():
    print(f'\t{key}: {value}')
print('\nRegression adjustment')
for key, value in evaluate_estimation(results.f.ate_reg, simulation_settings['true_ate'], results.f.ate_reg_var, simulation_settings['n'], level=0.95).items():
    print(f'\t{key}: {value}')

fig, ax = plt.subplots(ncols=3)
plot_estimator_distribution(results.f.ate, simulation_settings['true_ate'], ax[0], 'ATE')
plot_estimator_distribution(results.f.ate_reg_split, simulation_settings['true_ate'], ax[1], 'ATE reg. adj. cross-fitting')
plot_estimator_distribution(results.f.ate_reg, simulation_settings['true_ate'], ax[2], 'ATE reg. adj.')
plt.show()