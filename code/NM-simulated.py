import torch
from torch.distributions import Normal, Gamma, Bernoulli
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

#R functionality
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector

# Import R's utility package
utils = importr('utils')

# Install and load the 'EbayesThresh' package if not already installed
try:
    ebayesthresh = importr('EbayesThresh')
    ebnm = importr('ebnm')
except:
    utils.install_packages('EbayesThresh')
    ebayesthresh = importr('EbayesThresh')

    utils.install_packages('ebnm')
    ebnm = importr('ebnm')

from NormalMeans import train_model, evaluate_model, plot_estimator, evaluate_mm

#--------------------------------------------------------------------------------------------------
#Block1: MSE Table
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
n = 5000
s_values = [0.01, 0.05, 0.1, 0.2, 0.5,1]
means = [0, 3, 4, 5, 7]
alpha, beta = 2.0, 1
sigma2 = torch.tensor(1.0)

#MSE results
results = []

for s in s_values:
    for mean in means:
        variances = Gamma(torch.tensor([alpha]), torch.tensor([beta])).sample((n,)).squeeze()

        # Generating sparse signals: s% are non-zero, sampled from a normal distribution
        true_z = torch.zeros(n)
        non_zero_indices = torch.randperm(n)[:int(s * n)]
        true_z[non_zero_indices] = Normal(mean, torch.sqrt(variances[non_zero_indices])).sample()
        true_z = true_z.reshape(-1, 1)

        # Generate observations
        y = Normal(true_z, torch.sqrt(sigma2)).sample()

        # Training and evaluation
        model_ajs = train_model(y, sigma2, amortized="Mean")
        mse_ajs, hat_z_ajs = evaluate_model(model_ajs, y, true_z, sigma2)

        model_aeb = train_model(y, sigma2, approx=False)
        mse_aeb, hat_z_bayes = evaluate_model(model_aeb, y, true_z, sigma2)

        model_avar = train_model(y, sigma2, amortized="Variance")
        mse_avar, hat_z_avar = evaluate_model(model_avar, y, true_z, sigma2)

        # James-Stein estimator
        z_bar = torch.mean(y)
        abs2 = torch.sum((y - z_bar) ** 2)
        shrinkage_factor = 1 - (y.shape[0] - 3) / abs2
        hat_z_js = shrinkage_factor * y + (1 - shrinkage_factor) * z_bar
        mse_js = F.mse_loss(hat_z_js, true_z).item()


        # Johnstone-Silverman estimators
        ro.globalenv['y'] = FloatVector(y)
        ro.globalenv['sigma2'] = FloatVector([sigma2.item()])

        # Define the R code as a string
        r_code = """
                hat_z_laplace <- ebayesthresh(x = y, prior = "laplace", a = NA, sdev = sqrt(sigma2))
                hat_z_cauchy <- ebayesthresh(x = y, prior = "cauchy", a = NA, sdev = sqrt(sigma2))
                fit_npmle <- ebnm(y, sqrt(sigma2), prior_family = "npmle", control = list(verbose = TRUE))
                hat_z_npmle <- fit_npmle$posterior$mean
                list(hat_z_laplace = hat_z_laplace, hat_z_cauchy = hat_z_cauchy, hat_z_npmle = hat_z_npmle)
                """
        result = ro.r(r_code)

        # Convert the results from R to PyTorch
        hat_z_laplace = torch.tensor(np.array(result[0]), dtype=torch.float32).unsqueeze(1)
        hat_z_cauchy = torch.tensor(np.array(result[1]), dtype=torch.float32).unsqueeze(1)
        hat_z_npmle = torch.tensor(np.array(result[2]), dtype=torch.float32).unsqueeze(1)

        # Calculate the MSE for Laplace/Cauchy/NPMLE estimators
        mse_laplace = F.mse_loss(hat_z_laplace, true_z, reduction='mean').item()
        mse_cauchy = F.mse_loss(hat_z_cauchy, true_z, reduction='mean').item()
        mse_npmle = F.mse_loss(hat_z_npmle, true_z, reduction='mean').item()

        # MoM estimator
        mse_mm, hat_z_mm = evaluate_mm(y, true_z, sigma2)

        # Maximum likelihood estimator (MLE)
        mse_mle = F.mse_loss(y, true_z).item()

        # Minimax MSE
        if s <= 0.2:
            minimax_mse = 2 * s * np.log(1 / s)
        else:
            minimax_mse = 'Not Applicable'

        # Collecting results for each s
        results.append({
            's': s,
            'mean': mean,
            'Minimax': minimax_mse,
            'AEB-both': mse_aeb,
            'AEB-mean': mse_ajs,
            'AEB-variance': mse_avar,
            'Cauchy': mse_cauchy,
            'Laplace': mse_laplace,
            'NPMLE': mse_npmle,
            'James-Stein': mse_js,
            'MoM': mse_mm,
            'MLE': mse_mle
        })

        estimators = {
            'AEB-both': hat_z_bayes,
            'AEB-mean': hat_z_ajs,
            'AEB-variance': hat_z_avar,
            "Cauchy": hat_z_cauchy,
            "Laplace": hat_z_laplace,
            'NPMLE': hat_z_npmle,
            "James-Stein": hat_z_js,
            "MoM": hat_z_mm,
            "MLE": y
        }

        fig, ax = plt.subplots(3, 3, figsize=(14, 12))
        axes = ax.flatten()

        # Define colors
        colors = {
            'AEB-both': "#1f77b4",
            'AEB-mean': "#9467bd",
            'AEB-variance': "#17becf",
            "Cauchy": "#2ca02c",
            "Laplace": "#ff7f0e",
            "NPMLE": "#bcbd22",
            "James-Stein": "#2ca02c",
            "MoM": "#8c564b",
            "MLE": "#d62728"
        }

        for i, (title, z_data) in enumerate(estimators.items()):
            ax = axes[i]
            ax.scatter(y.numpy(), true_z.numpy(), color='grey', alpha=0.5, label='True $z$')
            ax.scatter(y.numpy(), z_data.numpy(), color=colors[title], alpha=0.6, label='Estimated $z$')
            ax.plot(y.numpy(), y.numpy(), linestyle=':', color='black', alpha=0.5)

            ax.set_title(title, fontsize=24)

            if i % 3 == 0:
                ax.set_ylabel("Estimated $z$", fontsize=20)
            else:
                ax.set_yticklabels([])

            if i // 3 == 2:
                ax.set_xlabel("Observed $y$", fontsize=20)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis='both', labelsize=18)
            ax.grid(True)
            ax.legend(fontsize=12, loc='upper left')

        for j in range(len(estimators), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plot_filename = f'../Plots/NormalMeans/nm_hetero_estimators_mean_{mean}_s_{s}.pdf'
        plt.savefig(plot_filename)
        plt.close()

        with torch.no_grad():
            mu_aeb, tau_aeb = model_aeb(y)

        # Plot the amortized prior means and variances
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        with torch.no_grad():
            plot_estimator(ax, y, mu_aeb, "Amortized Prior Means", "y", "μ", line_label='$\hat f_\mu(y) = y$',
                           plot_diagonal=True)

        plt.tight_layout()
        plot_filename = f'../Plots/NormalMeans/nm_hetero_prior_mean_{mean}_s_{s}.pdf'
        plt.savefig(plot_filename)
        plt.close()

        with torch.no_grad():
            mu, tau = model_aeb(y)
            exp_tau = torch.exp(tau)
            weights = sigma2 / (sigma2 + exp_tau)
        plt.figure(figsize=(10, 5))
        plt.scatter(y, weights, color='blue', alpha=0.6)
        plt.title("Amortized EB Shrinkage", fontsize=40)
        plt.xlabel("$y$", fontsize=36)
        plt.ylabel("Shrinkage weight", fontsize=36)
        plt.grid(True)
        plt.tight_layout()
        plot_filename = f'../Plots/NormalMeans/nm_hetero_shrinkage_mean_{mean}_s_{s}.pdf'
        plt.savefig(plot_filename)
        plt.close()

df_results = pd.DataFrame(results)
df_results = df_results.round(3)

df_results.columns = ['s', 'mean', 'Minimax', 'AEB-both', 'AEB-mean','AEB-variance',
                        'Cauchy', 'Laplace','NPMLE', 'James-Stein','MoM', 'MLE']

csv_filename = '../Results/NormalMeans/mse_results_table.csv'

df_results.to_csv(csv_filename, index=False)

#--------------------------------------------------------------------------------------------------
#Block2: MSE vs. s
torch.manual_seed(0)
np.random.seed(0)

n = 5000
s_values = np.logspace(-3, 0, 50)
mean = 5
alpha, beta = 2.0, 1
sigma2 = torch.tensor(1.0)

def evaluate_mse(model_func, y, true_z, sigma2):
    model = model_func(y, sigma2)
    mse, _ = evaluate_model(model, y, true_z, sigma2)
    return mse

mse_results = {
    's': s_values,
    'AEB-both': [],
    'AEB-variance': [],
    'Cauchy': [],
    'Laplace': [],
    'AEB-mean': [],
    'James-Stein': [],
    'NPMLE': []
}

for s in s_values:
    variances = Gamma(torch.tensor([alpha]), torch.tensor([beta])).sample((n,)).squeeze()
    true_z = torch.zeros(n)
    non_zero_indices = torch.randperm(n)[:int(s * n)]
    true_z[non_zero_indices] = Normal(mean, torch.sqrt(variances[non_zero_indices])).sample()
    true_z = true_z.reshape(-1, 1)
    y = Normal(true_z, torch.sqrt(sigma2)).sample()

    mse_results['AEB-both'].append(evaluate_mse(lambda y, sigma2: train_model(y, sigma2, approx=False), y, true_z, sigma2))
    mse_results['AEB-mean'].append(evaluate_mse(lambda y, sigma2: train_model(y, sigma2, amortized="Mean"), y, true_z, sigma2))
    mse_results['AEB-variance'].append(evaluate_mse(lambda y, sigma2: train_model(y, sigma2, amortized="Variance"), y, true_z, sigma2))

    ro.globalenv['y'] = FloatVector(y)
    ro.globalenv['sigma2'] = FloatVector([sigma2.item()])
    r_code = """
    set.seed(0)
    hat_z_laplace <- ebayesthresh(x = y, prior = "laplace", a = NA, sdev = sqrt(sigma2))
    hat_z_cauchy <- ebayesthresh(x = y, prior = "cauchy", a = NA, sdev = sqrt(sigma2))
    fit_npmle <- ebnm(y, sqrt(sigma2), prior_family = "npmle", control = list(verbose = TRUE))
    hat_z_npmle <- fit_npmle$posterior$mean
    list(hat_z_laplace = hat_z_laplace, hat_z_cauchy = hat_z_cauchy, hat_z_npmle = hat_z_npmle)
    """
    result = ro.r(r_code)
    hat_z_laplace = torch.tensor(np.array(result[0]), dtype=torch.float32).unsqueeze(1)
    hat_z_cauchy = torch.tensor(np.array(result[1]), dtype=torch.float32).unsqueeze(1)
    hat_z_npmle = torch.tensor(np.array(result[2]), dtype=torch.float32).unsqueeze(1)

    mse_results['Laplace'].append(F.mse_loss(hat_z_laplace, true_z, reduction='mean').item())
    mse_results['Cauchy'].append(F.mse_loss(hat_z_cauchy, true_z,reduction='mean').item())
    mse_results['NPMLE'].append(F.mse_loss(hat_z_npmle, true_z,reduction='mean').item())

    tau_js2 = (torch.norm(y) ** 2 / (y.shape[0] - 2)) - sigma2
    hat_z_js = tau_js2 / (sigma2 + tau_js2) * y
    mse_results['James-Stein'].append(F.mse_loss(hat_z_js, true_z).item())

df_mse = pd.DataFrame(mse_results)


plt.figure(figsize=(12, 7))

plt.plot(df_mse['s'], df_mse['AEB-both'], label='AEB-Both', color='blue', linestyle='-', marker='o', linewidth=2, markersize=8)
plt.plot(df_mse['s'], df_mse['AEB-variance'], label='AEB-Variance', color='orange', linestyle='--', marker='s', linewidth=2, markersize=8)
plt.plot(df_mse['s'], df_mse['Cauchy'], label='Cauchy', color='green', linestyle='-.', marker='D', linewidth=2, markersize=8)
plt.plot(df_mse['s'], df_mse['Laplace'], label='Laplace', color='red', linestyle=':', marker='x', linewidth=2, markersize=8)
plt.plot(df_mse['s'], df_mse['AEB-mean'], label='AEB-Mean', color='purple', linestyle='-', marker='^', linewidth=2, markersize=8)
plt.plot(df_mse['s'], df_mse['NPMLE'], label='NPMLE', color='black', linestyle='-', marker='*', linewidth=2, markersize=10)
plt.plot(df_mse['s'], df_mse['James-Stein'], label='James-Stein', color='brown', linestyle='--', marker='v', linewidth=2, markersize=8)

plt.xlabel('Sparsity level $s$', fontsize=32)
plt.ylabel('MSE', fontsize=32)
plt.title('MSE vs. Sparsity', fontsize=36, pad=20)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=20, loc='upper left', frameon=False)
plt.grid(True, which="both", linestyle='--', linewidth=1.0, alpha=0.6)

plt.tight_layout()
plot_filename = '../Plots/NormalMeans/mse_vs_s.pdf'
plt.savefig(plot_filename)
plt.show()
