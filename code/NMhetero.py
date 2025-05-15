import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler

#%%
class AmortizedMean(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth):
        super(AmortizedMean, self).__init__()

        def create_layers(n):
            layers = []
            for i in range(n):
                layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
                layers.append(nn.ReLU())
            return layers

        self.f_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *create_layers(depth),
            nn.Linear(hidden_dim, 1)
        )

        self.logtau2 = nn.Parameter(torch.tensor(0.0),  requires_grad=True)

    def forward(self, x):
        mu = self.f_mu(x)
        tau2 = torch.exp(self.logtau2)
        return mu, tau2

class AmortizedVar(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth):
        super(AmortizedVar, self).__init__()

        def create_layers(n):
            layers = []
            for i in range(n):
                layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
                layers.append(nn.ReLU())
            return layers

        self.f_tau = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *create_layers(depth),
            nn.Linear(hidden_dim, 1)
        )

        self.mu = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        mu = self.mu
        tau = self.f_tau(x)
        return mu, tau

class AmortizedBoth(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth = 30, input_dim_tau = None):
        if input_dim_tau is None:
            input_dim_tau = input_dim
        super(AmortizedBoth, self).__init__()

        def create_layers(n):
            layers = []
            for i in range(n):
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU()
                nn.Linear(hidden_dim * 2, hidden_dim)
                nn.ReLU(),
            return layers

        self.f_mu = nn.Sequential(
            nn.Linear(input_dim_tau, hidden_dim),
            nn.ReLU(),
            *create_layers(depth),
            nn.Linear(hidden_dim, 1)
        )
        # Defining the tau network
        self.f_tau = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *create_layers(depth),
            nn.Linear(hidden_dim, 1)
)

    def forward(self, x, x_tau = None):
        if x_tau is None:
            x_tau = x
        mu = self.f_mu(x)
        tau = self.f_tau(x_tau)
        return mu, tau

def nm_ajs_loss(y, mu, tau2, sigma2, sigma4 = None, mu_grad = None):
    var_sum_sq = (sigma2 + tau2).pow(2)
    if sigma4 is None:
        sigma4 = sigma2.pow(2)
    if mu_grad is None:
        mu_grad =torch.autograd.grad(outputs=mu, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    term1 = (sigma4/ var_sum_sq) * ((mu - y).pow(2))
    term2 = 2 * (sigma4/ (tau2 + sigma2)) * mu_grad
    term3 = 2 * (sigma2 * tau2 / (tau2 + sigma2))
    total_loss = (term1 + term2 + term3).sum()
    return (total_loss/torch.sum(sigma2) -1)

def nm_avar_loss(y, mu, f_tau, sigma2, sigma4 = None, tau_grad = None):
    tau2 = torch.exp(f_tau)
    var_sum_sq = (sigma2 + tau2).pow(2)
    if sigma4 is None:
        sigma4 = sigma2.pow(2)
    if tau_grad is None:
        tau_grad = torch.autograd.grad(outputs=f_tau, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    term1 = ((y - mu  + tau2 * tau_grad).pow(2) ) / var_sum_sq
    term2 = -(tau2 * tau_grad).pow(2) / var_sum_sq
    term3 = - 2/ (tau2 + sigma2)
    total_loss = (sigma4*(term1 + term2 + term3)).sum()
    return total_loss/torch.sum(sigma2) + 1


def nm_aeb_loss(y, mu, f_tau, sigma2, sigma4 = None, mu_grad = None, tau_grad = None):
    tau2 = torch.exp(f_tau)
    if sigma4 is None:
        sigma4 = sigma2.pow(2)

    # Gradient calculations
    if mu_grad is None:
        mu_grad =torch.autograd.grad(outputs=mu, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    if tau_grad is None:
        tau_grad =torch.autograd.grad(outputs=f_tau, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]

    # Calculating new loss components based on the provided formula
    term1 = ((y - mu  + tau2 * tau_grad).pow(2) ) / ((tau2 + sigma2).pow(2))
    term2 = -((tau2 * tau_grad).pow(2)) / ((tau2 + sigma2).pow(2))
    term3 = 2 * (mu_grad - 1) / (tau2 + sigma2)

    # Summing all components to form the total loss
    total_loss = (sigma4*(term1 + term2 + term3)).sum()
    return total_loss/(torch.sum(sigma2)) +1

def train_model(y, sigma2, x=None, hidden_dim=30, depth=5, epochs=3000, patience=100, lr_init=0.001, delta=1e-2, amortized="Both", print_epoch=True, print_NaN=False, x_tau=None, clip_value=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y.to(device)
    sigma2 = sigma2.to(device)


    y_with_grad = y.clone().detach().requires_grad_(True).to(device)
    if x is not None:
        x = x.to(device)
        inputs_tau = inputs = torch.cat((x, y_with_grad), dim=1)
    else:
        inputs_tau = inputs = y_with_grad

    if x_tau is not None:
        x_tau = x_tau.to(device)
        inputs_tau = torch.cat((x_tau, y_with_grad), dim=1)

    while True:
        if amortized == "Mean":
            model = AmortizedMean(input_dim=inputs.size(1), hidden_dim=hidden_dim, depth=depth).to(device)
        elif amortized == "Variance":
            model = AmortizedVar(input_dim=inputs.size(1), hidden_dim=hidden_dim, depth=depth).to(device)
        else:
            model = AmortizedBoth(input_dim=inputs.size(1), hidden_dim=hidden_dim, depth=depth, input_dim_tau=inputs_tau.size(1)).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr_init)
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=False)

        best_loss = np.inf
        counter = 0
        is_nan_encountered = False
        sigma4 = sigma2.pow(2)

        for epoch in range(epochs):
            optimizer.zero_grad()

            if amortized == "Mean":
                mu, tau2 = model(inputs)
                loss = nm_ajs_loss(y_with_grad, mu, tau2, sigma2, sigma4)
            elif amortized == "Variance":
                mu, tau = model(inputs)
                loss = nm_avar_loss(y_with_grad, mu, tau, sigma2, sigma4)
            else:
                mu, tau = model(inputs, inputs_tau)
                loss = nm_aeb_loss(y_with_grad, mu, tau, sigma2, sigma4)

            if torch.isnan(loss).any():
                if print_NaN:
                    print(f'NaN loss encountered at epoch {epoch}, restarting training...')
                is_nan_encountered = True
                break

            scaler.scale(loss).backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(loss.item())

            if epoch % 100 == 0 and print_epoch:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

            if abs(loss.item() - best_loss) < delta:
                counter += 1
            else:
                counter = 0
                best_loss = loss.item()

            if counter >= patience and print_epoch:
                print(f"Stopping early at epoch {epoch}. Best loss: {best_loss}")
                break

        if not is_nan_encountered:
            break

    return model

def evaluate_model(model, y, true_z, sigma2, x=None, x_tau=None):
    device = next(model.parameters()).device
    y, true_z, sigma2 = y.to(device), true_z.to(device), sigma2.to(device)

    inputs = torch.cat((x, y), dim=1) if x is not None else y
    inputs_tau = torch.cat((x_tau, y), dim=1) if x_tau is not None else inputs

    model.eval()  # Put model in evaluation mode

    with torch.no_grad():
        if isinstance(model, AmortizedBoth):
            hat_mu, tau = model(inputs, inputs_tau)
        elif isinstance(model, AmortizedMean):
            hat_mu, tau_hat2 = model(inputs)
            tau = torch.log(tau_hat2)  # Assume tau_hat2 is log-scale
        else:
            hat_mu, ftau = model(inputs)
            tau = ftau  # Use ftau directly if model doesn't include variance

        tau_hat2 = torch.exp(tau)
        total_variance = sigma2 + tau_hat2
        post_mean = (tau_hat2 / total_variance) * y + (sigma2 / total_variance) * hat_mu
        post_var = tau_hat2 * sigma2 / total_variance
        mse_bayes = F.mse_loss(post_mean, true_z).item()

    return mse_bayes, post_mean, post_var

def train_and_evaluate_model(y, sigma2, true_z, x, hidden_dim, depth, epochs, patience, lr_init, delta, amortized):
    # Train the model
    model = train_model(y, sigma2, x, hidden_dim, depth, epochs, patience, lr_init, delta, amortized, print_epoch = False, print_NaN = True)

    # Evaluate
    mse,_,_ = evaluate_model(model, y, true_z, sigma2)

    return mse
