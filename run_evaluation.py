import numpy as np
import matplotlib.pyplot as plt

def plot_estimator_distribution(estimator: np.ndarray, true_value: float, ax: plt.Axes, title: str):
    mean_estimator = np.nanmean(estimator)
    std_estimator = np.nanstd(estimator)
    x = np.linspace(-5*std_estimator + mean_estimator, 5*std_estimator + mean_estimator, 100)
    ax.hist(estimator, bins=50, alpha=0.5, density=True)
    ax.plot(x, 1/np.sqrt(2*np.pi*std_estimator)*np.exp(-((x-mean_estimator)/std_estimator)**2/2), 'r--')
    ax.axvline(true_value, color='k', linestyle='dashed', linewidth=1)
    ax.set_title(title)

def evaluate_estimation(ate: np.ndarray, ate_true: float) -> dict:
    bias = ate - ate_true
    return {
        'mean_bias': np.nanmean(bias),
        'rmse': np.nanstd(bias),
        'mae': np.nanmean(np.abs(bias)),
        'std_estimate': np.nanstd(ate),
    }


results = np.load('results/ate_2000_2000_10_0.5_2_4_1.0_complex_200_123.npz', allow_pickle=True)

print(evaluate_estimation(results.f.ate, results.f.simulation_settings.tolist()['true_ate']))
print(evaluate_estimation(results.f.ate_naive, results.f.simulation_settings.tolist()['true_ate']))

fig, ax = plt.subplots(ncols=2)
plot_estimator_distribution(results.f.ate, results.f.simulation_settings.tolist()['true_ate'], ax[0], 'ATE')
plot_estimator_distribution(results.f.ate, results.f.simulation_settings.tolist()['true_ate'], ax[1], 'ATE semi-naive')
plt.show()