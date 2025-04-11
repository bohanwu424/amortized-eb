import tensorflow_datasets as tfds
import tensorflow as tf
import torch
from torch.distributions import Normal
from torch.distributions.gamma import Gamma
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from NormalMeans import train_model, evaluate_model
from scipy.stats import norm

#------------------------------------------------------------------------------------
#Efron-Morris Data
datasets = tfds.load(name='efron_morris75', split='train', as_supervised=False)
data = tfds.as_numpy(datasets)

data_list = []
for item in data:
    data_dict = {name: value.decode('utf-8') if isinstance(value, bytes) else value.item() if value.size == 1 else value
                 for name, value in item.items()}
    data_list.append(data_dict)

# Create a DataFrame
df = pd.DataFrame(data_list)


# Normal transformation and its inverse
def h(x): return np.sqrt(45) * np.arcsin(2 * x - 1)


def h_inverse(x): return (np.sin(x / np.sqrt(45)) + 1) / 2


# Mean confidence interval calculation
def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    m, se = np.mean(a), np.std(a, ddof=1) / np.sqrt(len(a))
    h = se * norm.ppf((1 + confidence) / 2)
    return m, m - h, m + h


# Simulation parameters
n_simulations = 100
mse_results = {key: [] for key in ['AEB', 'AVAR', 'AJS', 'James-Stein', 'MLE']}
estimators = {key: [] for key in ['AEB', 'AVAR', 'AJS', 'James-Stein', 'MLE']}

# Simulation loop
for sim in range(n_simulations):
    np.random.seed(sim)
    torch.manual_seed(sim)

    y = torch.tensor(h(df['BattingAverage'].astype(float)), dtype=torch.float32).unsqueeze(1)
    z = torch.tensor(h(df['RemainingAverage'].astype(float)), dtype=torch.float32).unsqueeze(1)
    sigma2 = torch.tensor(1)

    model_aeb = train_model(y, sigma2, approx=False)
    model_ajs = train_model(y, sigma2, amortized='Mean')
    model_avar = train_model(y, sigma2, amortized='Variance')

    mse_aeb, hat_z_bayes = evaluate_model(model_aeb, y, z, sigma2)
    mse_avar, hat_z_avar = evaluate_model(model_avar, y, z, sigma2)
    mse_ajs, hat_z_ajs = evaluate_model(model_ajs, y, z, sigma2)

    z_bar = torch.mean(y)
    abs2 = torch.sum((y - z_bar) ** 2)
    shrinkage_factor = 1 - (18 - 3) / abs2
    hat_z_js = shrinkage_factor * y + (1 - shrinkage_factor) * z_bar
    mse_js = F.mse_loss(hat_z_js, z).item()

    mse_mle = F.mse_loss(y, z).item()

    mse_results['AEB'].append(mse_aeb)
    mse_results['AVAR'].append(mse_avar)
    mse_results['AJS'].append(mse_ajs)
    mse_results['James-Stein'].append(mse_js)
    mse_results['MLE'].append(mse_mle)

    estimators['AEB'].append(h_inverse(hat_z_bayes.numpy()).flatten())
    estimators['AVAR'].append(h_inverse(hat_z_avar.numpy()).flatten())
    estimators['AJS'].append(h_inverse(hat_z_ajs.numpy()).flatten())
    estimators['James-Stein'].append(h_inverse(hat_z_js.numpy()).flatten())
    estimators['MLE'].append(h_inverse(y.numpy()).flatten())

# Reporting results
for key in mse_results:
    mean_mse, lower, upper = mean_confidence_interval(mse_results[key])
    print(f"{key} MSE: Mean={mean_mse:.4f}, 95% CI=({lower:.4f}, {upper:.4f})")

# Calculate mean and confidence bands for each estimator at each y point
mean_estimators = {key: np.mean(np.stack(vals), axis=0) for key, vals in estimators.items()}
ci_lower_estimators = {key: np.percentile(np.stack(vals), 2.5, axis=0) for key, vals in estimators.items()}
ci_upper_estimators = {key: np.percentile(np.stack(vals), 97.5, axis=0) for key, vals in estimators.items()}


y_vals = df['BattingAverage'].astype(float).values
z_true = df['RemainingAverage'].astype(float).values
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
colors = {"AEB": "#1f77b4", "AJS": "#9467bd", "AVAR": "#ff7f0e", "James-Stein": "#2ca02c", "MLE": "#d62728"}

for i, (title, z_hat) in enumerate(mean_estimators.items()):
    row, col = divmod(i, 3)
    ci_lower = ci_lower_estimators[title]
    ci_upper = ci_upper_estimators[title]
    yerr = [z_hat - ci_lower, ci_upper - z_hat]
    yerr = np.array(yerr)
    yerr = np.abs(yerr)

    mse = np.mean((z_hat - z_true) ** 2)

    ax[row, col].scatter(y_vals, z_true, color='grey', alpha=0.6, label='True z')
    ax[row, col].scatter(y_vals, z_hat, color=colors[title], alpha=0.6, label='Estimated z')
    ax[row, col].errorbar(y_vals, z_hat, yerr=yerr, fmt='o', color=colors[title], alpha=0.6, label='Error bar')
    ax[row, col].plot(y_vals, y_vals, linestyle=':', color='black', alpha=0.5)
    ax[row, col].set_title(f"{title} (MSE={mse:.4f})", fontsize=16)
    ax[row, col].set_xlabel('y', fontsize=14)
    ax[row, col].set_ylabel('z', fontsize=14)
    ax[row, col].legend()
    ax[row, col].grid(True)

# Turn off any unused subplots
for j in range(i + 1, len(ax.flat)):
    fig.delaxes(ax.flat[j])

plt.tight_layout()
plot_filename = '../Plots/NormalMeans/nm_EfronMorris.pdf'
plt.savefig(plot_filename)
plt.show()
#------------------------------------------------------------------------------------
#
