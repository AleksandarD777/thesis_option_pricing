"""Closed-form Black-Scholes pricing formulas."""

import numpy as np
from scipy.stats import norm


def black_scholes_call(S0, K, T, r, sigma):
    """Return the Black-Scholes price of a vanilla European call option."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def compute_bs_call(r, sigma, S0, T, K):
    """Compatibility wrapper for the original notebook argument order."""
    return black_scholes_call(S0, K, T, r, sigma)

