import torch
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import scipy.stats as stats

# Import R's utility package
utils = importr('utils')

import matplotlib.pyplot as plt

# Install and load the 'EbayesThresh' package if not already installed
try:
    ebayesthresh = importr('EbayesThresh')
except:
    utils.install_packages('EbayesThresh')
    ebayesthresh = importr('EbayesThresh')

from NormalMeans import train_model, evaluate_model

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


# Load the RData file
file_path = '../data/prostz.RData'
ro.r['load'](file_path)
# Fetch the data from R's global environment
data_in_r = ro.r['prostz']

y = torch.tensor(np.array(data_in_r), dtype=torch.float32).view(-1,1)

sigma2 = torch.tensor(1.0)


# James-Stein Estimator (shink towards the mean)
z_bar = torch.mean(y)
abs2 = torch.sum((y - z_bar) ** 2)
shrinkage_factor = 1 - (y.shape[0] - 3) / abs2
hat_z_js = shrinkage_factor * y + (1 - shrinkage_factor) * z_bar

# Train and evaluate the models
model_ajs = train_model(y, sigma2, amortized="Mean")
_, hat_z_ajs = evaluate_model(model_ajs, y, y, sigma2)  # use a dummy true z = y

model_avar = train_model(y, sigma2, amortized="Variance")
_, hat_z_avar = evaluate_model(model_avar, y, y, sigma2)

model_aeb = train_model(y, sigma2, amortized="Both")
_, hat_z_aeb = evaluate_model(model_aeb, y, y, sigma2)

credible_interval = 0.95
alpha = 0.05

q_star = -stats.norm.ppf((1 - credible_interval)/2)

with torch.no_grad():
    mu, tau = model_aeb(y)
    post_var = 1 / (torch.exp(-tau) + 1 / sigma2)
    post_lb = hat_z_aeb - q_star * torch.sqrt(post_var)
    post_ub = hat_z_aeb + q_star * torch.sqrt(post_var)
    post_sd = torch.sqrt(post_var)

# Convert tensors to numpy arrays
y_np = y.cpu().numpy().flatten()
hat_z_js_np = hat_z_js.cpu().numpy().flatten()
hat_z_ajs_np = hat_z_ajs.cpu().numpy().flatten()
hat_z_avar_np = hat_z_avar.cpu().numpy().flatten()
hat_z_aeb_np = hat_z_aeb.cpu().numpy().flatten()
lower_bands_aeb_np = post_lb.cpu().numpy().flatten()
upper_bands_aeb_np = post_ub.cpu().numpy().flatten()
post_sd_np = post_sd.cpu().numpy().flatten()

# Sort the values for proper plotting
sorted_indices = np.argsort(y_np)
y_sorted = y_np[sorted_indices]
hat_z_js_sorted = hat_z_js_np[sorted_indices]
hat_z_ajs_sorted = hat_z_ajs_np[sorted_indices]
hat_z_avar_sorted = hat_z_avar_np[sorted_indices]
hat_z_aeb_sorted = hat_z_aeb_np[sorted_indices]
lower_bands_sorted = lower_bands_aeb_np[sorted_indices]
upper_bands_sorted = upper_bands_aeb_np[sorted_indices]
post_sd_sorted = post_sd_np[sorted_indices]

# Compute two-sided p-values for each estimate
p_values_js = 2 * (1 - stats.norm.cdf(np.abs(hat_z_js_np)))
p_values_ajs = 2 * (1 - stats.norm.cdf(np.abs(hat_z_ajs_np)))
p_values_avar = 2 * (1 - stats.norm.cdf(np.abs(hat_z_avar_np)))
p_values_aeb = 2 * (1 - stats.norm.cdf(np.abs(hat_z_aeb_np)))

# Evaluate p-values at multiple points between the lower and upper bands
z_range = np.linspace(lower_bands_aeb_np, upper_bands_aeb_np, num=500)

#For probability of significant effects
z_threshold = 1.5
prob_aeb = 1 - (stats.norm.cdf((z_threshold - np.abs(hat_z_aeb_np)) / post_sd_np) - stats.norm.cdf((-z_threshold - np.abs(hat_z_aeb_np)) / post_sd_np))

#Compute the local false sign rate
local_false_sign_rate = np.where(
    y_np > 0,
    stats.norm.cdf(-hat_z_aeb_np / post_sd_np),
    stats.norm.cdf(hat_z_aeb_np / post_sd_np)
)
local_false_sign_rate = stats.norm.cdf(hat_z_aeb_np / post_sd_np)


# Amortized EB findings
indices_aeb = np.where((lower_bands_aeb_np > 0) | (upper_bands_aeb_np < 0))[0]

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 8))

# Plot the posterior credible bands
axs[0].scatter(y_sorted, hat_z_aeb_sorted, color='black', marker='o', label='Estimate $\hat{z}^{\mathrm{AEB}}$')
axs[0].plot(y_sorted, hat_z_aeb_sorted,  color='blue')
axs[0].fill_between(y_sorted, lower_bands_sorted, upper_bands_sorted, color='blue', alpha=0.2, label=f'{100*credible_interval}% Credible Band')
axs[0].axhline(y=0, color='grey', linestyle='--')

# Significant genes
significant_y = y.cpu().numpy()[indices_aeb]
significant_hat_z = hat_z_aeb.cpu().numpy().flatten()[indices_aeb]
axs[0].scatter(significant_y, significant_hat_z, color='red', marker='o', label='Significant Genes')

axs[0].set_xlabel('y', fontsize=14)
axs[0].set_ylabel('Estimates for $z$', fontsize=14)
axs[0].set_title('AEB Estimates and Credible Band', fontsize=16)
axs[0].legend(fontsize=12, loc='upper left')
axs[0].grid(True)

# Plot the posterior probability of |z| > stats.norm.ppf(alpha / 2).
axs[1].plot(y_sorted, local_false_sign_rate[sorted_indices], color='red')
axs[1].axhline(y=0, color='grey', linestyle='--')
axs[1].axhline(y=0.5, color='grey', linestyle='--')
axs[1].axhline(y=1, color='grey', linestyle='--')
axs[1].set_xlabel('y', fontsize=14)
axs[1].set_ylabel(f'$P(|z| > {round(z_threshold, 2)} \mid y)$', fontsize=14)
axs[1].set_title('Probability of Significant Findings', fontsize=16)
axs[1].grid(True)

plt.tight_layout()
plot_filename = '../Plots/Prostate/AEB-credible-bands.pdf'
plt.savefig(plot_filename)
plt.show()

#-------------------------------
# Label shuffling
n_trials = 20
suspect_gene_counts = []

for i in range(n_trials):
    y_shuffled = y[torch.randperm(y.shape[0])]

    # Train on shuffled data
    model_aeb_null = train_model(y_shuffled, sigma2, amortized="Both")
    _, hat_z_aeb_null = evaluate_model(model_aeb_null, y_shuffled, y_shuffled, sigma2)

    with torch.no_grad():
        mu_null, tau_null = model_aeb_null(y_shuffled)
        post_var_null = 1 / (torch.exp(-tau_null) + 1 / sigma2)
        post_lb_null = hat_z_aeb_null - q_star * torch.sqrt(post_var_null)
        post_ub_null = hat_z_aeb_null + q_star * torch.sqrt(post_var_null)

    lower_bands_null_np = post_lb_null.cpu().numpy().flatten()
    upper_bands_null_np = post_ub_null.cpu().numpy().flatten()
    indices_null = np.where((lower_bands_null_np > 0) | (upper_bands_null_np < 0))[0]

    # Intersect with original significant genes
    shared_significant_genes = np.intersect1d(indices_aeb, indices_null)
    suspect_gene_counts.append(len(shared_significant_genes))

# Summary
avg_suspects = np.mean(suspect_gene_counts)
std_suspects = np.std(suspect_gene_counts)

print(f"\n[Label Shuffling Check - {n_trials} Trials]")
print(f"Original significant genes (AEB): {len(indices_aeb)}")
print(f"Average shared significant genes across shuffles: {avg_suspects:.2f}")
print(f"Standard deviation: {std_suspects:.2f}")