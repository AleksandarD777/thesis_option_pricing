"""PINN components for European call pricing under Black-Scholes."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

from src.black_scholes import black_scholes_call
from src.metrics import mae, max_absolute_error, rmse


@dataclass
class LossWeights:
    """Weights applied to the PINN loss components."""

    pde: float = 1.0
    terminal: float = 1.0
    boundary_left: float = 1.0
    boundary_right: float = 1.0


@dataclass
class TrainingConfig:
    """Training settings for the Black-Scholes PINN."""

    epochs: int = 5000
    n_interior: int = 2000
    n_terminal: int = 800
    n_boundary: int = 400
    lr: float = 1e-3
    print_every: int = 500
    seed: int | None = None
    loss_weights: LossWeights | None = None
    value_scale: float | None = None


class PINN(nn.Module):
    """Feed-forward neural network used as a PINN approximation to option value."""

    def __init__(self, input_dim=2, hidden_dim=64, num_hidden=3, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def call_payoff(S, K):
    """Return the European call payoff max(S - K, 0)."""
    return torch.clamp(S - K, min=0.0)


def normalize_inputs(S, t, S_max, T):
    """Normalize stock and time inputs for the PINN."""
    return torch.cat([S / S_max, t / T], dim=1)


def set_seed(seed):
    """Set the PyTorch and NumPy seeds when reproducibility is requested."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def model_price(model, S, t, S_max, T, value_scale=1.0):
    """Return the rescaled option price predicted by the PINN."""
    return value_scale * model(normalize_inputs(S, t, S_max, T))


def pde_residual(model, S, t, S_max, K, T, r, sigma):
    """Return the Black-Scholes PDE residual for the scaled PINN output."""
    del K
    S = S.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    # The PDE is homogeneous, so the residual can be computed on the scaled output.
    value = model(normalize_inputs(S, t, S_max, T))

    value_t = torch.autograd.grad(
        value, t, grad_outputs=torch.ones_like(value), create_graph=True
    )[0]
    value_s = torch.autograd.grad(
        value, S, grad_outputs=torch.ones_like(value), create_graph=True
    )[0]
    value_ss = torch.autograd.grad(
        value_s, S, grad_outputs=torch.ones_like(value_s), create_graph=True
    )[0]

    return value_t + 0.5 * sigma**2 * S**2 * value_ss + r * S * value_s - r * value


def sample_interior(n, S_max, T, device):
    """Sample interior collocation points for the PDE residual."""
    S = torch.rand(n, 1, device=device) * S_max
    t = torch.rand(n, 1, device=device) * T
    return S, t


def sample_terminal(n, S_max, T, device):
    """Sample terminal-condition points at maturity."""
    S = torch.rand(n, 1, device=device) * S_max
    t = torch.full((n, 1), T, device=device)
    return S, t


def sample_boundary_left(n, T, device):
    """Sample left-boundary points where S is zero."""
    S = torch.zeros(n, 1, device=device)
    t = torch.rand(n, 1, device=device) * T
    return S, t


def sample_boundary_right(n, S_max, T, device):
    """Sample right-boundary points where S equals S_max."""
    S = torch.full((n, 1), S_max, device=device)
    t = torch.rand(n, 1, device=device) * T
    return S, t


def train_pinn(
    model,
    K,
    T,
    r,
    sigma,
    S_max,
    epochs=5000,
    n_interior=2000,
    n_terminal=800,
    n_boundary=400,
    lr=1e-3,
    device=None,
    print_every=500,
    loss_weights=None,
    seed=None,
    value_scale=None,
):
    """Train a European call PINN and return the model plus training metrics."""
    if device is None:
        device = next(model.parameters()).device
    set_seed(seed)
    weights = loss_weights or LossWeights()
    scale = float(value_scale if value_scale is not None else 1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "total_loss": [],
        "pde_loss": [],
        "terminal_loss": [],
        "boundary_left_loss": [],
        "boundary_right_loss": [],
    }
    start = perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        S_f, t_f = sample_interior(n_interior, S_max, T, device)
        residual = pde_residual(model, S_f, t_f, S_max, K, T, r, sigma)
        loss_pde = torch.mean(residual**2)

        S_T, t_T = sample_terminal(n_terminal, S_max, T, device)
        value_terminal = model(normalize_inputs(S_T, t_T, S_max, T))
        loss_terminal = torch.mean((value_terminal - call_payoff(S_T, K) / scale) ** 2)

        S_l, t_l = sample_boundary_left(n_boundary, T, device)
        value_left = model(normalize_inputs(S_l, t_l, S_max, T))
        loss_left = torch.mean(value_left**2)

        S_r, t_r = sample_boundary_right(n_boundary, S_max, T, device)
        value_right = model(normalize_inputs(S_r, t_r, S_max, T))
        right_boundary = (S_max - K * torch.exp(-r * (T - t_r))) / scale
        loss_right = torch.mean((value_right - right_boundary) ** 2)

        loss = (
            weights.pde * loss_pde
            + weights.terminal * loss_terminal
            + weights.boundary_left * loss_left
            + weights.boundary_right * loss_right
        )
        loss.backward()
        optimizer.step()

        history["total_loss"].append(float(loss.item()))
        history["pde_loss"].append(float(loss_pde.item()))
        history["terminal_loss"].append(float(loss_terminal.item()))
        history["boundary_left_loss"].append(float(loss_left.item()))
        history["boundary_right_loss"].append(float(loss_right.item()))
        if print_every and epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    metrics = {
        "history": history,
        "runtime_sec": perf_counter() - start,
        "final_total_loss": history["total_loss"][-1] if history["total_loss"] else np.nan,
        "loss_weights": {
            "pde": weights.pde,
            "terminal": weights.terminal,
            "boundary_left": weights.boundary_left,
            "boundary_right": weights.boundary_right,
        },
        "value_scale": scale,
    }
    return model, metrics


def train_pinn_from_config(model, K, T, r, sigma, S_max, config=None, device=None):
    """Train a PINN using a TrainingConfig object."""
    config = config or TrainingConfig()
    return train_pinn(
        model,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        S_max=S_max,
        epochs=config.epochs,
        n_interior=config.n_interior,
        n_terminal=config.n_terminal,
        n_boundary=config.n_boundary,
        lr=config.lr,
        device=device,
        print_every=config.print_every,
        loss_weights=config.loss_weights,
        seed=config.seed,
        value_scale=config.value_scale,
    )


def price_pinn(model, S0, t0, S_max, T, device=None, value_scale=1.0):
    """Evaluate a trained PINN at one stock/time point."""
    if device is None:
        device = next(model.parameters()).device
    S = torch.tensor([[S0]], dtype=torch.float32, device=device)
    t = torch.tensor([[t0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model_price(model, S, t, S_max, T, value_scale=value_scale).item()


def evaluate_pinn_grid(model, S_values, t_values, S_max, T, device=None, value_scale=1.0):
    """Evaluate a trained PINN on a rectangular stock-time grid."""
    if device is None:
        device = next(model.parameters()).device

    S_values = np.asarray(S_values, dtype=np.float32)
    t_values = np.asarray(t_values, dtype=np.float32)
    S_mesh, t_mesh = np.meshgrid(S_values, t_values, indexing="ij")

    S_tensor = torch.tensor(S_mesh.reshape(-1, 1), dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_mesh.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        values = model_price(model, S_tensor, t_tensor, S_max, T, value_scale=value_scale)

    return S_mesh, t_mesh, values.cpu().numpy().reshape(S_mesh.shape)


def compare_t0_curve(model, S_values, K, T, r, sigma, S_max, device=None, value_scale=1.0):
    """Compare the PINN and analytical Black-Scholes prices along the t=0 curve."""
    S_values = np.asarray(S_values, dtype=float)
    _, _, pinn_values = evaluate_pinn_grid(
        model,
        S_values,
        np.array([0.0]),
        S_max,
        T,
        device=device,
        value_scale=value_scale,
    )
    pinn_prices = pinn_values[:, 0]
    analytical_prices = np.array(
        [black_scholes_call(S0, K, T, r, sigma) for S0 in S_values],
        dtype=float,
    )
    errors = pinn_prices - analytical_prices

    return {
        "S": S_values,
        "pinn_price": pinn_prices,
        "analytical_price": analytical_prices,
        "error": errors,
        "abs_error": np.abs(errors),
        "mae": mae(errors),
        "rmse": rmse(errors),
        "max_abs_error": max_absolute_error(errors),
    }


def plot_loss_history(metrics, ax=None):
    """Plot total and component training losses from a metrics dictionary."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    history = metrics["history"]
    ax.plot(history["total_loss"], label="total")
    ax.plot(history["pde_loss"], label="PDE", alpha=0.8)
    ax.plot(history["terminal_loss"], label="terminal", alpha=0.8)
    ax.plot(history["boundary_left_loss"], label="left boundary", alpha=0.8)
    ax.plot(history["boundary_right_loss"], label="right boundary", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PINN Training Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    return ax


def plot_t0_comparison(comparison, ax=None):
    """Plot PINN and analytical prices along the t=0 stock-price curve."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(comparison["S"], comparison["analytical_price"], label="Black-Scholes", color="#222222", linewidth=2.0)
    ax.plot(comparison["S"], comparison["pinn_price"], label="PINN", color="#2ca02c", linestyle="--", linewidth=2.0)
    ax.set_xlabel("Stock price")
    ax.set_ylabel("Call price")
    ax.set_title("PINN vs Black-Scholes at t=0")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    return ax
