import torch
from torch.distributions import Normal, Gamma
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from NormalMeans import  train_model, evaluate_model

# Import R's utility package and the EbayesThresh package
utils = importr('utils')
try:
    ebayesthresh = importr('EbayesThresh')
except:
    utils.install_packages('EbayesThresh')
    ebayesthresh = importr('EbayesThresh')


def bootstrap_mse(y, true_z, sigma2, S, n_bootstraps=100):
    mse_samples = []
    valid = False

    while not valid:
        mse_samples.clear()
        for _ in range(n_bootstraps):
            indices = torch.randint(0, len(y), (S,))  # Subsample S samples from y
            y_bootstrap = y[indices]
            true_z_bootstrap = true_z[indices]

            # Train model on the bootstrap sample
            model_aeb = train_model(y_bootstrap, sigma2, approx=False, epochs=1000, depth=5, hidden_dim=20)
            mse_aeb, _ = evaluate_model(model_aeb, y_bootstrap, true_z_bootstrap, sigma2)

            # Check for NaN
            if np.isnan(mse_aeb):
                mse_samples.clear()
                break
            mse_samples.append(mse_aeb)

        if mse_samples:
            valid = True

    # Calculate confidence intervals and mean
    ci_lower = np.percentile(mse_samples, 2.5)
    ci_upper = np.percentile(mse_samples, 97.5)
    mean_mse = np.mean(mse_samples)
    return mean_mse, ci_lower, ci_upper



torch.manual_seed(0)
np.random.seed(0)

# Constants for simulation
n_values = np.logspace(np.log10(1000), np.log10(1000000), num=20).astype(int)
alpha, beta = 2.0, 1.0
sigma2 = torch.tensor(1.0)

# Simulate data and compute bootstrap MSE with error handling
from torch.distributions import Normal, Gamma
import pandas as pd

# Placeholder DataFrame
df_results = pd.DataFrame(columns=['n', 'Minimax MSE', 'Mean AEB MSE', 'CI Lower', 'CI Upper'])

for n in n_values:
    s = 1 / np.sqrt(n)
    variances = Gamma(torch.tensor([alpha]), torch.tensor([beta])).sample((n,)).squeeze()
    true_z = torch.zeros(n)
    non_zero_indices = torch.randperm(n)[:int(s * n)]
    true_z[non_zero_indices] = Normal(5, torch.sqrt(variances[non_zero_indices])).sample()
    true_z = true_z.reshape(-1, 1)
    y = Normal(true_z, torch.sqrt(sigma2)).sample()
    model_aeb = train_model(y, sigma2, approx=False, epochs=3000, depth=10, hidden_dim=30)
    mse_aeb, _ = evaluate_model(model_aeb, y, true_z, sigma2)
    #  mean_mse_aeb, ci_lower, ci_upper = bootstrap_mse(y, true_z, sigma2, S=int(np.floor(n**(2/3))))
    minimax_mse = 2 * s * np.log(1 / s)

    df_results = df_results.append({
        'n': n,
        'Minimax MSE': minimax_mse,
        'AEB MSE': mse_aeb,
     #   'Mean AEB MSE': mean_mse_aeb,
     #   'CI Lower': ci_lower,
      #  'CI Upper': ci_upper
    }, ignore_index=True)

# Plot results with confidence intervals
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(df_results['n'], df_results['AEB MSE'], linestyle='-', label='AEB MSE')
ax.plot(df_results['n'], df_results['Minimax MSE'], label='Minimax MSE', linestyle='--')
ax.set_title('MSE Comparison as a Function of $n$ ($s = 1/\sqrt{n}$)', fontsize=16)
ax.set_xlabel('Number of Samples ($n$)', fontsize=14)
ax.set_ylabel('MSE', fontsize=14)
ax.set_xscale('log')
ax.grid(True)
ax.legend()
plt.tight_layout()
plot_filename = '../Plots/NormalMeans/Risk_vs_n.pdf'
plt.savefig(plot_filename)
plt.close()