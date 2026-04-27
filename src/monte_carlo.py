"""Monte Carlo pricing routines for option experiments."""

import numpy as np


def _rng(seed=None):
    """Return a NumPy random number generator, preserving old random behavior by default."""
    return np.random if seed is None else np.random.default_rng(seed)


def monte_carlo_european_call(S0, K, T, r, sigma, n_paths, seed=None):
    """Price a vanilla European call with risk-neutral Monte Carlo simulation."""
    rng = _rng(seed)
    z = rng.standard_normal(n_paths)

    s_t = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(s_t - K, 0)

    return np.exp(-r * T) * np.mean(payoff)


def monte_carlo_asian_call(S0, K, T, r, sigma, n_paths, n_steps, seed=None):
    """Price an arithmetic-average Asian call with Monte Carlo simulation."""
    rng = _rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_steps, n_paths))

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(increments, axis=0)])
    s_paths = S0 * np.exp(log_paths)

    average_price = np.mean(s_paths[1:], axis=0)
    payoffs = np.maximum(average_price - K, 0)

    return np.exp(-r * T) * np.mean(payoffs)


def monte_carlo_basket_call(s1, s2, K, T, r, sigma1, sigma2, corr, n_paths, seed=None):
    """Price an equally weighted two-asset basket call by Monte Carlo simulation."""
    rng = _rng(seed)
    z1 = rng.standard_normal(n_paths)
    z2 = rng.standard_normal(n_paths) * np.sqrt(1 - corr**2) + corr * z1

    st1 = s1 * np.exp((r - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * z1)
    st2 = s2 * np.exp((r - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * z2)

    payoffs = np.maximum(0.5 * st1 + 0.5 * st2 - K, 0)

    return np.exp(-r * T) * np.mean(payoffs), z1, z2


def compute_mc_call(r, sigma, S0, T, K, n_paths, seed=None):
    """Compatibility wrapper for the original notebook argument order."""
    return monte_carlo_european_call(S0, K, T, r, sigma, n_paths, seed=seed)


def compute_asian_call(r, sigma, S0, T, K, n_paths, n_steps, seed=None):
    """Compatibility wrapper for the original notebook argument order."""
    return monte_carlo_asian_call(S0, K, T, r, sigma, n_paths, n_steps, seed=seed)


def compute_basket_call(r, s1, s2, sigma1, sigma2, K, T, corr, n_paths, seed=None):
    """Compatibility wrapper for the original notebook argument order."""
    return monte_carlo_basket_call(s1, s2, K, T, r, sigma1, sigma2, corr, n_paths, seed=seed)

