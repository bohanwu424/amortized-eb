import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import hypergeom
from NMhetero import train_model, evaluate_model
from ucimlrepo import fetch_ucirepo
import torch
from modeling_util import VAE, loss_function


# Fetch dataset
communities_and_crime = fetch_ucirepo(id=211)

# Data (as pandas dataframes)
X_crime = communities_and_crime.data.features
y_crime = communities_and_crime.data.targets

# Merge features and target to have a complete dataframe
crime_df = pd.concat([X_crime, y_crime], axis=1)

# List all column names
column_names = crime_df.columns.tolist()
print("All column names in the dataset:")
for column in column_names:
    print(column)

# Filter out rows where nonViolPerPop is NaN
crime_df = crime_df.dropna(subset=['nonViolPerPop'])

# Convert nonViolPerPop to float
crime_df['nonViolPerPop'] = crime_df['nonViolPerPop'].astype(float)

# Select features (removing 'nonViolPerPop', 'population' if present)
X = crime_df.iloc[:, 2:126].drop(columns=['nonViolPerPop', 'population'], errors='ignore')

# Remove columns with NaNs
X = X.dropna(axis=1)

# Select only columns where the type is float64
X_sub = X.loc[:, X.applymap(lambda x: np.issubdtype(type(x), np.float64)).all()]

# Get the names of the selected columns
selected_column_names = X_sub.columns.tolist()

# Standardize the features
scaler = StandardScaler()
X_sub_scaled = scaler.fit_transform(X_sub)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_sub_scaled, dtype=torch.float32)


def run_experiment(B, num_simulations=100):
    # Perform hypergeometric subsampling
    ncrimes = (crime_df['nonViolPerPop'] / 100_000 * crime_df['pop']).round().astype(int)
    npopulation = crime_df['pop'].astype(int)
    N_i = [hypergeom(npop, nc, B).rvs() for nc, npop in zip(ncrimes, npopulation)]
    N_i = np.array(N_i)

    # Calculate the estimated non-violent crime rate
    p_hat = N_i / B

    # Variance stabilizing transformation (VST)
    y = np.sqrt(p_hat)

    # Convert y to PyTorch tensor
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    p_i = torch.tensor(crime_df['nonViolPerPop'].values / 100_000, dtype=torch.float32).view(-1, 1)

    # Define sigma2
    sigma2 = torch.full((len(y), 1), 1 / (4 * B), dtype=torch.float32)

    mse_results = {
        "AJS without covariates": [],
        "AVAR without covariates": [],
        "AEB without covariates": [],
        "AJS with covariates": [],
        "Reduced dimensions (Mean)": [],
        "Reduced dimensions (Variance)": [],
        "Reduced dimensions (Both)": []
    }

    for _ in range(num_simulations):
        np.random.seed(_)

        # Train the models
        model_ajs = train_model(y_tensor, sigma2, hidden_dim=30, depth=10, epochs=3000, patience=200, lr_init=0.01, delta=1e-3, amortized="Mean", print_NaN=True)
        model_avar = train_model(y_tensor, sigma2, hidden_dim=30, depth=10, epochs=3000, patience=200, lr_init=0.01, delta=1e-3, amortized="Variance", print_NaN=True)
        model_aeb = train_model(y_tensor, sigma2, hidden_dim=30, depth=10, epochs=3000, patience=200, lr_init=0.01, delta=1e-3, amortized="Both", print_NaN=True)

        # Evaluate the models
        _, hat_z_ajs = evaluate_model(model_ajs, y_tensor, y_tensor, sigma2)
        _, hat_z_avar = evaluate_model(model_avar, y_tensor, y_tensor, sigma2)
        _, hat_z_aeb = evaluate_model(model_aeb, y_tensor, y_tensor, sigma2)

        # Calculate Mean Squared Errors
        mse_results["AJS without covariates"].append(F.mse_loss(hat_z_ajs**2, p_i).item() * 1_000_000)
        mse_results["AVAR without covariates"].append(F.mse_loss(hat_z_avar**2, p_i).item() * 1_000_000)
        mse_results["AEB without covariates"].append(F.mse_loss(hat_z_aeb**2, p_i).item() * 1_000_000)

        # Train and evaluate the model with covariates
        model_ajs_covariate = train_model(y_tensor, sigma2, x=X_tensor, hidden_dim=30, depth=10, epochs=3000, patience=200, lr_init=0.01, delta=1e-3, amortized="Mean", print_NaN=True)
        _, hat_z_ajs_covariate = evaluate_model(model_ajs_covariate, y_tensor, y_tensor, sigma2, x=X_tensor)
        mse_results["AJS with covariates"].append(F.mse_loss(hat_z_ajs_covariate ** 2, p_i).item() * 1_000_000)

        # Training parameters for VAE
        input_dim = X_tensor.shape[1]
        latent_dim = 3
        learning_rate = 1e-2
        batch_size = len(X_tensor)
        num_epochs = 3000

        # Initialize the VAE and optimizer
        vae = VAE(input_dim, latent_dim)
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, verbose=True)

        # DataLoader for VAE
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop for VAE
        vae.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for data in dataloader:
                x = data[0]
                optimizer.zero_grad()
                recon_x, mu, logvar = vae(x)
                loss = loss_function(recon_x, x, mu, logvar)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            scheduler.step(total_loss / len(dataloader.dataset))
            if epoch % 100 == 0:
                print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader.dataset)}')

        with torch.no_grad():
            # Transform the covariates
            X_reduced = vae.encode(X_tensor)

            # Calculate reconstruction error
            recon_X, _, _ = vae(X_tensor)
            recon_error = nn.functional.mse_loss(recon_X, X_tensor).item()
        print(f'Reconstruction Error: {recon_error}')

        model_reduced_mean = train_model(y_tensor, sigma2=sigma2, x=X_reduced, hidden_dim=30, depth=10, epochs=3000,
                                         patience=50, lr_init=0.001, delta=1e-3, amortized="Mean", print_NaN=True)
        model_reduced_variance = train_model(y_tensor, sigma2=sigma2, x=X_reduced, hidden_dim=30, depth=10,
                                             epochs=3000, patience=50, lr_init=0.001, delta=1e-3,
                                             amortized="Variance", print_NaN=True)
        model_reduced_both = train_model(y_tensor, sigma2=sigma2, x=X_reduced, hidden_dim=30, depth=10, epochs=3000,
                                         patience=50, lr_init=0.001, delta=1e-3, amortized="Both", print_NaN=True)
        # Evaluate the models with reduced covariates
        _, hat_z_reduced_mean = evaluate_model(model_reduced_mean, y_tensor, y_tensor, sigma2=sigma2, x=X_reduced)
        _, hat_z_reduced_variance = evaluate_model(model_reduced_variance, y_tensor, y_tensor, sigma2=sigma2,
                                                   x=X_reduced)
        _, hat_z_reduced_both = evaluate_model(model_reduced_both, y_tensor, y_tensor, sigma2=sigma2, x=X_reduced)

        # Calculate Mean Squared Errors with reduced covariates
        mse_results["Reduced dimensions (Mean)"].append(F.mse_loss(hat_z_reduced_mean ** 2, p_i).item() * 1_000_000)
        mse_results["Reduced dimensions (Variance)"].append(
            F.mse_loss(hat_z_reduced_variance ** 2, p_i).item() * 1_000_000)
        mse_results["Reduced dimensions (Both)"].append(F.mse_loss(hat_z_reduced_both ** 2, p_i).item() * 1_000_000)

    summary_results = {key: {"mean": np.mean(val), "stderr": np.std(val) / np.sqrt(num_simulations) * 2} for
                       key, val in mse_results.items()}

    mse_df = pd.DataFrame(summary_results).T
    mse_df.columns = ["Mean MSE", "2 * StdErr"]
    mse_df.to_csv(f"../Results/NormalMeans/mse_crimes_B{B}.csv", index=True)
    print(f"MSE results saved to mse_crimes_B{B}.csv")

run_experiment(200, num_simulations=100)
run_experiment(500, num_simulations=100)