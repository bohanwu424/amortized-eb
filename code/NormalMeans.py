import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

#Amortizing Means
class AmortizedMean(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth):
        super(AmortizedMean, self).__init__()

        def create_layers(n):
            layers = []
            for i in range(n):
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU()
                nn.Linear(hidden_dim * 2, hidden_dim)
                nn.ReLU(),
            return layers

        self.f = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *create_layers(depth),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        output = self.f(x)
        return output

def compute_tau_squared(y, model, sigma_squared):
    model.eval()
    device = next(model.parameters()).device
    y_tensor = y.clone().detach().to(device).requires_grad_(True)
    y_pred = model(y_tensor)
    squared_errors = (y_pred - y_tensor) ** 2
    mse = squared_errors.sum().item()

    f_grad = torch.autograd.grad(
        outputs=y_pred,
        inputs=y_tensor,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    div = torch.sum(f_grad)
    n = y.size(0)
    tau_squared = max((mse / (n - div)) - sigma_squared, 0)
    return tau_squared
def nm_hom_loss(y, y_pred, sigma2):
    n = y.numel()
    ls_loss = torch.sum((y_pred - y) ** 2)
    f_grad = torch.autograd.grad(outputs=y_pred, inputs=y, grad_outputs=torch.ones_like(y),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
    div = torch.sum(f_grad)
    tau_hat2 = max((ls_loss / (n - div)) - sigma2, 0)
    total_loss = 1/y.size(0)*(ls_loss + 2*(tau_hat2 + sigma2)*div + n/sigma2*tau_hat2**2)
    return total_loss

#Amortizing Variances
class AmortizedVar(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth):
        super(AmortizedVar, self).__init__()

        def create_layers(n):
            layers = []
            for i in range(n):
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU()
                nn.Linear(hidden_dim * 2, hidden_dim)
                nn.ReLU(),
            return layers

        self.f = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *create_layers(depth),
            nn.Linear(hidden_dim, 1)
    )

    def forward(self, x):
        output = self.f(x)
        return output

def nm_var_loss(y, ftau, sigma2):
    tau = torch.exp(ftau)
    f_grad = torch.autograd.grad(outputs=ftau, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True, only_inputs=True)[0]
    weights = 1 / (sigma2 + tau) ** 2
    hat_mu_numerator = torch.sum((y + sigma2 * f_grad * tau) * weights)
    hat_mu_denominator = torch.sum(weights)
    hat_mu = hat_mu_numerator / hat_mu_denominator

    risk_terms = ((y - hat_mu) ** 2) / ((sigma2 + tau) ** 2)
    risk = 2 * torch.sum((sigma2 * (y - hat_mu) * f_grad * tau / (sigma2 + tau) ** 2) - (sigma2 / (sigma2 + tau))) + torch.sum(risk_terms)

    return risk, hat_mu

#Amortizing Means and Variances
class ConstantLayer(nn.Module):
    def __init__(self, output_value=1.0):
        super(ConstantLayer, self).__init__()
        self.output = nn.Parameter(torch.tensor([output_value]), requires_grad=True)

    def forward(self, x):
        return self.output.expand(x.size(0), -1)

class AmortizedBoth(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth = 30):
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
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *create_layers(depth),
            nn.Linear(hidden_dim, 1)
        )
        # Defining the tau network
        self.f_tau = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *create_layers(depth),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        mu = self.f_mu(x)
        tau = self.f_tau(x)
        return mu, tau



def nm_hetero_loss(y, mu, tau, sigma_squared, approx=False):
    sigma2 = torch.tensor(sigma_squared, dtype=torch.float32)
    var_tau = torch.exp(tau)

    # Gradient calculations
    mu_grad =torch.autograd.grad(outputs=mu, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True, only_inputs=True)[0]
    tau_grad =torch.autograd.grad(outputs=tau, inputs=y, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Calculating new loss components based on the provided formula
    term1 = ((y - mu  + var_tau * tau_grad)**2 ) / ((var_tau + sigma2)**2)
    term2 = -(var_tau * tau_grad)**2 / ((var_tau + sigma2)**2 )
    term3 = 2 * (mu_grad - 1) / (var_tau + sigma2)
    if approx:
        term3 =  2 * (mu_grad**2 - 1) / (var_tau + sigma2)

    # Summing all components to form the total loss
    total_loss = term1.sum() + term2.sum() + term3.sum()
    return total_loss


def train_model(y, sigma2, hidden_dim=30, depth=5, epochs=3000, patience=100, lr_init=0.001, delta=1e-2, approx=False,
                amortized="Mean and Variance"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y.to(device)
    sigma2 = sigma2.to(device)

    while True:  # Loop to allow for retraining if NaN loss occurs
        if amortized == "Mean":
            model = AmortizedMean(input_dim=y.size(1), hidden_dim=hidden_dim, depth=depth).to(device)
        elif amortized == "Variance":
            model = AmortizedVar(input_dim=y.size(1), hidden_dim=hidden_dim, depth=depth).to(device)
        else:
            model = AmortizedBoth(input_dim=y.size(1), hidden_dim=hidden_dim, depth=depth).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=0.05)
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=False)

        best_loss = np.inf
        counter = 0
        is_nan_encountered = False

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_with_grad = y.clone().detach().to(device).requires_grad_(True)
            if amortized == "Mean":
                loss = nm_hom_loss(y_with_grad, model(y_with_grad), sigma2)
            elif amortized == "Variance":
                loss, _ = nm_var_loss(y_with_grad, model(y_with_grad), sigma2)
            else:
                mu, tau = model(y_with_grad)
                loss = nm_hetero_loss(y_with_grad, mu, tau, sigma2, approx=approx)

            if torch.isnan(loss).any():  # Check if the loss is NaN
                print(f'NaN loss encountered at epoch {epoch}, restarting training...')
                is_nan_encountered = True
                break

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(loss.item())

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

            if abs(loss.item() - best_loss) < delta:
                counter += 1
            else:
                counter = 0
                best_loss = loss.item()

            if counter >= patience:
                print(f"Stopping early at epoch {epoch}. Best loss: {best_loss}")
                break

        if not is_nan_encountered:
            break

    return model

def plot_estimator(ax, y, estimates, title, xlabel, ylabel, line_label=None, plot_diagonal=True):
    ax.scatter(y.numpy().flatten(), estimates.numpy().flatten(), color='blue', alpha=0.6)
    if plot_diagonal:
        ax.plot(y.numpy().flatten(), y.numpy().flatten(), linestyle=':', color='red', label=line_label)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True)




def evaluate_model(model, y, true_z, sigma2):
    device = next(model.parameters()).device
    y = y.to(device)
    true_z = true_z.to(device)
    sigma2 = sigma2.to(device)

    # Ensure model is in evaluation mode
    model.eval()
    if isinstance(model, AmortizedBoth):
        with torch.no_grad():
            mu, log_tau = model(y)
            tau = torch.exp(log_tau)
            weights = sigma2 / (sigma2 + tau)
            hat_z_bayes = weights * mu + (1 - weights) * y
            mse_bayes = F.mse_loss(hat_z_bayes, true_z).item()
            return mse_bayes, hat_z_bayes

    elif isinstance(model, AmortizedMean):
        tau_hat2 = compute_tau_squared(y, model, sigma2)
        total_variance = sigma2 + tau_hat2

        with torch.no_grad():
            y_pred = model(y.unsqueeze(1)).squeeze(1)
            hat_z_bayes = (tau_hat2 / total_variance) * y + (sigma2 / total_variance) * y_pred
            mse_bayes = F.mse_loss(hat_z_bayes, true_z).item()
            return mse_bayes, hat_z_bayes

    else:
        y_with_grad = y.clone().detach().to(device).requires_grad_(True)
        ftau = model(y_with_grad)
        _, hat_mu = nm_var_loss(y_with_grad, ftau, sigma2)
        with torch.no_grad():
            tau_hat2 = torch.exp(ftau)
            total_variance = sigma2 + tau_hat2
            hat_z_bayes =  (tau_hat2 / total_variance) * y + (sigma2 / total_variance) * hat_mu
            mse_bayes = F.mse_loss(hat_z_bayes, true_z).item()
            return mse_bayes, hat_z_bayes

def mm_estimator(y, sigma2):
    y = torch.as_tensor(y, dtype=torch.float32)
    sigma2 = torch.as_tensor(sigma2, dtype=torch.float32)

    mean_y = y.mean()

    y_diff = y - mean_y
    y_diff_squared_minus_sigma2 = y_diff ** 2 - sigma2
    positive_part = torch.clamp(y_diff_squared_minus_sigma2, min=0)

    denominator = positive_part + sigma2
    weight_mean = sigma2 / denominator
    weight_y = positive_part / denominator

    hat_z_mom = weight_mean * mean_y + weight_y * y

    return hat_z_mom


def evaluate_mm(y, true_z, sigma2):
    hat_z = mm_estimator(y, sigma2)
    mse = F.mse_loss(hat_z, true_z).item()
    return mse, hat_z


