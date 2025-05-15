import torch
from torch.distributions import Normal
from torch.distributions.gamma import Gamma
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

#R functionality
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector

# Import R package
utils = importr('utils')

try:
    ebayesthresh = importr('EbayesThresh')
except:
    utils.install_packages('EbayesThresh')
    ebayesthresh = importr('EbayesThresh')

from NormalMeans import train_model, evaluate_model

# Set random seed
n_values = np.logspace(np.log10(500), np.log10(100000), 50, dtype=int)
mean = 5
alpha, beta = 2.0, 1
sigma2 = torch.tensor(1.0)

mse_results = {
    'n': n_values,
    'AEB': np.zeros(len(n_values)),
    'AVAR': np.zeros(len(n_values)),
    'Cauchy': np.zeros(len(n_values)),
    'Laplace': np.zeros(len(n_values)),
    'Minimax': np.zeros(len(n_values))
}

def evaluate_mse(model_func, y, true_z, sigma2):
    model = model_func(y, sigma2)
    mse, _ = evaluate_model(model, y, true_z, sigma2)
    return mse

num_simulations = 20

for seed in range(num_simulations):
    torch.manual_seed(seed)
    np.random.seed(seed)

    for i, n in enumerate(n_values):
        s = 1 / np.sqrt(n)
        variances = Gamma(torch.tensor([alpha]), torch.tensor([beta])).sample((n,)).squeeze()
        true_z = torch.zeros(n)
        non_zero_indices = torch.randperm(n)[:int(s * n)]
        true_z[non_zero_indices] = Normal(mean, torch.sqrt(variances[non_zero_indices])).sample()
        true_z = true_z.reshape(-1, 1)
        y = Normal(true_z, torch.sqrt(sigma2)).sample()

        mse_results['AEB'][i] += evaluate_mse(lambda y, sigma2: train_model(y, sigma2, approx=False,patience=50,delta=1e-1), y, true_z, sigma2)
        mse_results['AVAR'][i] += evaluate_mse(lambda y, sigma2: train_model(y, sigma2, amortized="Variance",patience=50,delta=1e-1), y, true_z, sigma2)

        ro.globalenv['y'] = FloatVector(y)
        ro.globalenv['sigma2'] = FloatVector([sigma2.item()])
        ro.globalenv['seed'] = seed
        r_code = """
        set.seed(seed)
        hat_z_laplace <- ebayesthresh(x = y, prior = "laplace", a = NA, sdev = sqrt(sigma2))
        hat_z_cauchy <- ebayesthresh(x = y, prior = "cauchy", a = NA, sdev = sqrt(sigma2))
        list(hat_z_laplace = hat_z_laplace, hat_z_cauchy = hat_z_cauchy)
        """
        result = ro.r(r_code)
        hat_z_laplace = torch.tensor(np.array(result[0]), dtype=torch.float32).unsqueeze(1)
        hat_z_cauchy = torch.tensor(np.array(result[1]), dtype=torch.float32).unsqueeze(1)
        mse_results['Laplace'][i] += F.mse_loss(hat_z_laplace, true_z).item()
        mse_results['Cauchy'][i] += F.mse_loss(hat_z_cauchy, true_z).item()

        minimax_mse = 2 * s * np.log(1 / s) if s <= 0.2 else 0
        mse_results['Minimax'][i] += minimax_mse if minimax_mse != 0 else 0

# Compute average results over simulations
for key in mse_results.keys():
    if key != 'n':
        mse_results[key] /= num_simulations

df_mse = pd.DataFrame(mse_results)
#save results
df_mse.to_csv('../Results/NormalMeans/mse_vs_n.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(df_mse['n'], df_mse['AEB'], label='AEB-both', color='blue', linestyle='-', marker='o')
plt.plot(df_mse['n'], df_mse['AVAR'], label='AEB-variance', color='orange', linestyle='--', marker='s')
plt.plot(df_mse['n'], df_mse['Cauchy'], label='Cauchy', color='green', linestyle='-.', marker='D')
plt.plot(df_mse['n'], df_mse['Laplace'], label='Laplace', color='red', linestyle=':', marker='x')
plt.plot(df_mse['n'], df_mse['Minimax'], label='Minimax', color='black', linestyle='-', marker='*')

plt.xscale('log')
plt.xlabel('n', fontsize=32)
plt.ylabel('MSE', fontsize=32)
plt.title(r'MSE vs. Sample Size ($s_n = n^{-1/2}$)', fontsize=36, pad=20)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=20, loc='upper rigth', frameon=False)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plot_filename = '../Plots/NormalMeans/mse_vs_n.pdf'
plt.savefig(plot_filename)
plt.show()