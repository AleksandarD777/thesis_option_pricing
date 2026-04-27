"""Reusable benchmark pipeline for European call pricing methods."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.black_scholes import black_scholes_call
from src.finite_differences import crank_nicolson_call
from src.metrics import mae, max_absolute_error, rmse
from src.monte_carlo import monte_carlo_european_call


@dataclass
class OptionParams:
    """Parameters shared by all European call pricing methods."""

    S0: float = 100.0
    K: float = 110.0
    T: float = 2.0
    r: float = 0.035


@dataclass
class CNGrid:
    """Crank-Nicolson grid settings."""

    S_max: float = 600.0
    M: int = 200
    N: int = 200


@dataclass
class MonteCarloConfig:
    """Monte Carlo settings for repeated benchmark runs."""

    n_paths: int = 10000
    n_repeats: int = 5
    seed: int = 12345


def volatility_grid(start=0.10, stop=0.45, step=0.03):
    """Return a rounded volatility grid including the final endpoint when it lands on the grid."""
    n_steps = int(np.floor((stop - start) / step + 1e-12))
    return np.round(start + step * np.arange(n_steps + 1), 10)


def price_with_runtime(pricer):
    """Run a pricing callable and return its value and elapsed seconds."""
    start = perf_counter()
    price = pricer()
    runtime = perf_counter() - start
    return float(price), runtime


def monte_carlo_repeated(S0, K, T, r, sigma, config):
    """Run repeated Monte Carlo prices and return mean, standard deviation, and total runtime."""
    prices = []
    start = perf_counter()
    for repeat in range(config.n_repeats):
        seed = None if config.seed is None else config.seed + repeat
        price = monte_carlo_european_call(S0, K, T, r, sigma, config.n_paths, seed=seed)
        prices.append(price)

    runtime = perf_counter() - start
    ddof = 1 if len(prices) > 1 else 0
    return float(np.mean(prices)), float(np.std(prices, ddof=ddof)), runtime


def error_columns(price, analytical_price):
    """Return absolute and relative error against the analytical benchmark."""
    if price is None or np.isnan(price):
        return np.nan, np.nan

    abs_error = abs(price - analytical_price)
    rel_error = abs_error / abs(analytical_price) if analytical_price != 0 else np.nan
    return float(abs_error), float(rel_error)


def run_volatility_benchmark(
    params=None,
    sigmas=None,
    cn_grid=None,
    mc_config=None,
    pinn_pricer=None,
    pinn_sigmas=None,
):
    """Run a volatility sweep and return row dictionaries suitable for CSV export."""
    params = params or OptionParams()
    sigmas = volatility_grid() if sigmas is None else sigmas
    cn_grid = cn_grid or CNGrid()
    mc_config = mc_config or MonteCarloConfig()
    pinn_sigma_set = None
    if pinn_sigmas is not None:
        pinn_sigma_set = {round(float(sigma), 10) for sigma in pinn_sigmas}

    rows = []
    for sigma in sigmas:
        sigma = float(sigma)

        analytical_price, analytical_runtime = price_with_runtime(
            lambda: black_scholes_call(params.S0, params.K, params.T, params.r, sigma)
        )
        mc_price, mc_std, mc_runtime = monte_carlo_repeated(
            params.S0, params.K, params.T, params.r, sigma, mc_config
        )
        cn_price, cn_runtime = price_with_runtime(
            lambda: crank_nicolson_call(
                params.S0,
                params.K,
                params.T,
                params.r,
                sigma,
                cn_grid.S_max,
                cn_grid.M,
                cn_grid.N,
            )[0]
        )

        run_pinn = pinn_pricer is not None and (
            pinn_sigma_set is None or round(sigma, 10) in pinn_sigma_set
        )
        if not run_pinn:
            pinn_price = np.nan
            pinn_runtime = np.nan
        else:
            pinn_price, pinn_runtime = price_with_runtime(lambda: pinn_pricer(sigma))

        mc_abs_error, mc_rel_error = error_columns(mc_price, analytical_price)
        cn_abs_error, cn_rel_error = error_columns(cn_price, analytical_price)
        pinn_abs_error, pinn_rel_error = error_columns(pinn_price, analytical_price)

        rows.append(
            {
                "sigma": sigma,
                "analytical_price": analytical_price,
                "analytical_runtime_sec": analytical_runtime,
                "mc_price_mean": mc_price,
                "mc_price_std": mc_std,
                "mc_runtime_sec": mc_runtime,
                "mc_abs_error": mc_abs_error,
                "mc_rel_error": mc_rel_error,
                "cn_price": cn_price,
                "cn_runtime_sec": cn_runtime,
                "cn_abs_error": cn_abs_error,
                "cn_rel_error": cn_rel_error,
                "pinn_price": pinn_price,
                "pinn_runtime_sec": pinn_runtime,
                "pinn_abs_error": pinn_abs_error,
                "pinn_rel_error": pinn_rel_error,
            }
        )

    return rows


def _finite_values(rows, column):
    """Return finite numeric values from one benchmark result column."""
    values = []
    for row in rows:
        value = row.get(column, np.nan)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def summarize_benchmark(rows):
    """Return method-level error and runtime summaries from benchmark rows."""
    method_columns = {
        "Monte Carlo": {
            "abs_error": "mc_abs_error",
            "rel_error": "mc_rel_error",
            "runtime": "mc_runtime_sec",
        },
        "Crank-Nicolson": {
            "abs_error": "cn_abs_error",
            "rel_error": "cn_rel_error",
            "runtime": "cn_runtime_sec",
        },
        "PINN": {
            "abs_error": "pinn_abs_error",
            "rel_error": "pinn_rel_error",
            "runtime": "pinn_runtime_sec",
        },
    }

    summaries = []
    for method, columns in method_columns.items():
        abs_errors = _finite_values(rows, columns["abs_error"])
        rel_errors = _finite_values(rows, columns["rel_error"])
        runtimes = _finite_values(rows, columns["runtime"])

        summaries.append(
            {
                "method": method,
                "n_observations": len(abs_errors),
                "mae": mae(abs_errors) if abs_errors else np.nan,
                "rmse": rmse(abs_errors) if abs_errors else np.nan,
                "max_abs_error": max_absolute_error(abs_errors) if abs_errors else np.nan,
                "mean_relative_error": float(np.mean(rel_errors)) if rel_errors else np.nan,
                "mean_runtime_sec": float(np.mean(runtimes)) if runtimes else np.nan,
                "max_runtime_sec": float(np.max(runtimes)) if runtimes else np.nan,
            }
        )

    return summaries
