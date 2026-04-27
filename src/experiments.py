"""Additional numerical experiments for thesis method comparisons."""

import numpy as np

from src.benchmarking import MonteCarloConfig, OptionParams, error_columns, monte_carlo_repeated, price_with_runtime
from src.black_scholes import black_scholes_call
from src.finite_differences import crank_nicolson_call
from src.metrics import mae, max_absolute_error, rmse


def run_mc_convergence(params=None, sigma=0.25, path_counts=None, n_repeats=5, seed=12345):
    """Run Monte Carlo convergence over increasing path counts."""
    params = params or OptionParams()
    path_counts = path_counts or [1000, 5000, 10000, 50000, 100000]
    analytical_price = black_scholes_call(params.S0, params.K, params.T, params.r, sigma)
    rows = []

    for n_paths in path_counts:
        config = MonteCarloConfig(n_paths=int(n_paths), n_repeats=n_repeats, seed=seed)
        price_mean, price_std, runtime = monte_carlo_repeated(
            params.S0,
            params.K,
            params.T,
            params.r,
            sigma,
            config,
        )
        abs_error, rel_error = error_columns(price_mean, analytical_price)
        rows.append(
            {
                "method": "Monte Carlo",
                "sigma": float(sigma),
                "n_paths": int(n_paths),
                "n_repeats": int(n_repeats),
                "price": price_mean,
                "price_std": price_std,
                "analytical_price": analytical_price,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "runtime_sec": runtime,
            }
        )

    return rows


def run_cn_refinement(params=None, sigma=0.25, S_max=600.0, grid_sizes=None):
    """Run Crank-Nicolson refinement over increasing grid sizes."""
    params = params or OptionParams()
    grid_sizes = grid_sizes or [50, 100, 200, 400]
    analytical_price = black_scholes_call(params.S0, params.K, params.T, params.r, sigma)
    rows = []

    for grid_size in grid_sizes:
        grid_size = int(grid_size)
        price, runtime = price_with_runtime(
            lambda: crank_nicolson_call(
                params.S0,
                params.K,
                params.T,
                params.r,
                sigma,
                S_max,
                grid_size,
                grid_size,
            )[0]
        )
        abs_error, rel_error = error_columns(price, analytical_price)
        rows.append(
            {
                "method": "Crank-Nicolson",
                "sigma": float(sigma),
                "M": grid_size,
                "N": grid_size,
                "grid_size": grid_size,
                "price": price,
                "analytical_price": analytical_price,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "runtime_sec": runtime,
            }
        )

    return rows


def summarize_experiment(rows):
    """Summarize error and runtime metrics by method."""
    methods = sorted({row["method"] for row in rows})
    summary = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        abs_errors = [float(row["abs_error"]) for row in method_rows if np.isfinite(row["abs_error"])]
        runtimes = [float(row["runtime_sec"]) for row in method_rows if np.isfinite(row["runtime_sec"])]
        summary.append(
            {
                "method": method,
                "n_observations": len(method_rows),
                "mae": mae(abs_errors) if abs_errors else np.nan,
                "rmse": rmse(abs_errors) if abs_errors else np.nan,
                "max_abs_error": max_absolute_error(abs_errors) if abs_errors else np.nan,
                "average_runtime_sec": float(np.mean(runtimes)) if runtimes else np.nan,
                "max_runtime_sec": float(np.max(runtimes)) if runtimes else np.nan,
            }
        )

    return summary

