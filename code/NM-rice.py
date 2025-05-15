import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#try if this works
try:
    dataset = fetch_ucirepo(id=545)
    X = dataset.data.features.values
    y = dataset.data.targets.values
except:
    file_path = '../data/Rice_Cammeo_Osmancik.xlsx'
    dataset = pd.read_excel(file_path)
    X = dataset.iloc[:, :-1].values  # Features
    y = dataset.iloc[:, -1].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Accuracy on clean test data
y_pred_clean = model.predict(X_test)
accuracy_clean = accuracy_score(y_test, y_pred_clean)
print(f'Accuracy on clean test data: {accuracy_clean:.4f}')

d = X_test.shape[1]
n_test = X_test.shape[0]

class F_mu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(F_mu, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, y):
        return self.fc(y)

class F_S(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(F_S, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim * output_dim)
        )
        self.output_dim = output_dim

    def forward(self, y):
        batch_size = y.size(0)
        L_params = self.fc(y).view(batch_size, self.output_dim, self.output_dim)
        L = torch.tril(L_params)
        S = torch.matmul(L, L.transpose(1, 2))
        return S
class F_mu_Constant(torch.nn.Module):
    def __init__(self, mu_0):
        super(F_mu_Constant, self).__init__()
        self.mu_0 = torch.nn.Parameter(mu_0)

    def forward(self, y):
        batch_size = y.size(0)
        return self.mu_0.unsqueeze(0).repeat(batch_size, 1)
class F_S_Constant(torch.nn.Module):
    def __init__(self, S_0):
        super(F_S_Constant, self).__init__()
        self.S_0 = torch.nn.Parameter(S_0)

    def forward(self, y):
        batch_size = y.size(0)
        return self.S_0.unsqueeze(0).repeat(batch_size, 1, 1)

def denoise(y, f_mu, f_S, Sigma):
    mu = f_mu(y)
    S = f_S(y)
    S_plus_Sigma = S + Sigma
    adjustment = torch.linalg.solve(S_plus_Sigma, Sigma @ (mu - y).unsqueeze(-1)).squeeze(-1)
    z_estimate = y + adjustment
    return z_estimate

def SURE(f_mu, f_S, y, Sigma):
    y.requires_grad_(True)
    z_estimate = denoise(y, f_mu, f_S, Sigma)
    adjustment = z_estimate - y
    error = torch.sum(adjustment ** 2)
    v = torch.randint(0, 2, y.shape, device=device).float() * 2 - 1
    Jv = torch.autograd.grad(
        outputs=z_estimate,
        inputs=y,
        grad_outputs=v,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    Sigma_v = (Sigma @ v.unsqueeze(-1)).squeeze(-1)
    div_term = torch.sum(Sigma_v * Jv)
    loss = error + 2 * div_term
    return loss / n_test

def create_batches(X, batch_size):
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training setup
f_mu = F_mu(input_dim=d, output_dim=d).to(device)
f_S = F_S(input_dim=d, output_dim=d).to(device)
optimizer = optim.Adam(list(f_mu.parameters()) + list(f_S.parameters()), lr=1e-3, weight_decay=0.05)
batch_size = 64
epochs = 3000

# Store results
results = {"noise_level": [], "accuracy_clean": [], "accuracy_noisy": [], "accuracy_denoised": [], "mse_noisy": [], "mse_denoised": []}

# Evaluate for multiple noise levels
noise_levels = np.linspace(0.1, 2.5, 10)
#noise_levels = [1]
amortization_types = ["Gaussian", "mean", "covariance", "both"]

# Store all results in a single dataframe
all_results = []


for noise_std_test in noise_levels:
    np.random.seed(0)
    noise_test = noise_std_test * np.random.randn(*X_test.shape)
    X_test_noisy = X_test + noise_test

    # Convert to tensors
    X_test_noisy_tensor = torch.tensor(X_test_noisy, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    Sigma = torch.stack([noise_std_test ** 2 * torch.eye(d) for _ in range(n_test)]).to(device)

    # Accuracy on  noisy input
    y_pred_noisy = model.predict(X_test_noisy)
    accuracy_noisy = accuracy_score(y_test, y_pred_noisy)

    # Accuracy on clean data
    accuracy_clean = accuracy_score(y_test, model.predict(X_test))

    for amortization_type in amortization_types:
        if amortization_type == "Gaussian":
            f_mu = F_mu_Constant(torch.zeros(d, device=device))
            f_S = F_S_Constant(torch.eye(d, device=device))
        elif amortization_type == "mean":
            f_mu = F_mu(input_dim=d, output_dim=d).to(device)
            f_S = F_S_Constant(torch.eye(d, device=device))
        elif amortization_type == "covariance":
            f_mu = F_mu_Constant(torch.zeros(d, device=device))
            f_S = F_S(input_dim=d, output_dim=d).to(device)
        else:  # Amortizing both
            f_mu = F_mu(input_dim=d, output_dim=d).to(device)
            f_S = F_S(input_dim=d, output_dim=d).to(device)

        optimizer = optim.Adam(list(f_mu.parameters()) + list(f_S.parameters()) if callable(f_mu) else list(f_S.parameters()), lr=1e-3, weight_decay=0.05)
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = SURE(f_mu, f_S, X_test_noisy_tensor, Sigma)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(loss.item())

        # Denoising
        with torch.no_grad():
            X_test_denoised_tensor = denoise(X_test_noisy_tensor, f_mu, f_S, Sigma)
            X_test_denoised = X_test_denoised_tensor.cpu().numpy()

        # Evaluate
        y_pred_denoised = model.predict(X_test_denoised)
        accuracy_denoised = accuracy_score(y_test, y_pred_denoised)

        mse_noisy = nn.MSELoss()(X_test_tensor, X_test_noisy_tensor).item()
        mse_denoised = nn.MSELoss()(X_test_tensor, X_test_denoised_tensor).item()

        all_results.append({
            "amortization_type": amortization_type,
            "noise_level": noise_std_test,
            "accuracy_clean": accuracy_clean,
            "accuracy_noisy": accuracy_noisy,
            "accuracy_denoised": accuracy_denoised,
            "mse_noisy": mse_noisy,
            "mse_denoised": mse_denoised
        })

# Save all results to a single CSV file
all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv('../Results/NormalMeans-Rice/results_rice.csv', index=False)

# Plotting
all_results_df = pd.read_csv('../Results/NormalMeans-Rice/results_rice.csv')
plt.figure(figsize=(15, 7))

# First Plot: MSE vs Noise Level (Left plot)
plt.subplot(1, 2, 1)

# Plot for Amortizing Mean, Covariance, and Mean + Covariance
for amortization_type in amortization_types:
    subset = all_results_df[all_results_df['amortization_type'] == amortization_type]
    plt.plot(subset["noise_level"], subset["mse_denoised"], label=f'{amortization_type}', marker='o')

plt.xlabel('Noise Level', fontsize=24)
plt.ylabel('MSE', fontsize=24)
plt.legend(fontsize=20)
plt.title('Denoising MSE', fontsize=32, pad=20)

# Second Plot: Accuracy vs Noise Level (Right plot)
plt.subplot(1, 2, 2)

# Plot for Noisy data
plt.plot(all_results_df["noise_level"].unique(),
         all_results_df.groupby("noise_level")["accuracy_noisy"].mean(),
         label='noisy', linestyle='--', marker='o')

# Plot for Amortizing Mean, Covariance, and Mean + Covariance
for amortization_type in amortization_types:
    subset = all_results_df[all_results_df['amortization_type'] == amortization_type]
    plt.plot(subset["noise_level"], subset["accuracy_denoised"], label=f'{amortization_type}', marker='o')

# Plot for Original/clean test data
plt.plot(all_results_df["noise_level"].unique(),
         all_results_df.groupby("noise_level")["accuracy_clean"].mean(),
         label='clean', linestyle='-', marker='x')

plt.xlabel('Noise Level', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.legend(fontsize=20)
plt.title('Test Accuracy', fontsize=32, pad=20)
plt.tight_layout()
plt.savefig('../Plots/NormalMeans/results_rice.pdf', dpi=300)
plt.show()
