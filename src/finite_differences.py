"""Finite-difference solvers for the Black-Scholes PDE."""

import numpy as np


def explicit_fd_call(S0, K, T, r, sigma, S_max, M, N):
    """Price a vanilla European call with an explicit finite-difference scheme."""
    dt = T / N

    s_grid = np.linspace(0, S_max, M + 1)
    t_grid = np.linspace(0, T, N + 1)
    values = np.zeros((M + 1, N + 1))

    values[:, -1] = np.maximum(s_grid - K, 0)
    values[0, :] = 0
    values[-1, :] = S_max - K * np.exp(-r * (T - t_grid))

    for j in range(N - 1, -1, -1):
        for i in range(1, M):
            alpha = 0.5 * dt * (sigma**2 * i**2 - r * i)
            beta = 1 - dt * (sigma**2 * i**2 + r)
            gamma = 0.5 * dt * (sigma**2 * i**2 + r * i)

            values[i, j] = (
                alpha * values[i - 1, j + 1]
                + beta * values[i, j + 1]
                + gamma * values[i + 1, j + 1]
            )

    price = np.interp(S0, s_grid, values[:, 0])
    return price, s_grid, t_grid, values


def crank_nicolson_call(S0, K, T, r, sigma, S_max, M, N):
    """Price a vanilla European call with the Crank-Nicolson finite-difference scheme."""
    dt = T / N

    s_grid = np.linspace(0, S_max, M + 1)
    t_grid = np.linspace(0, T, N + 1)
    values = np.zeros((M + 1, N + 1))

    values[:, -1] = np.maximum(s_grid - K, 0)
    values[0, :] = 0
    values[-1, :] = S_max - K * np.exp(-r * (T - t_grid))

    i = np.arange(1, M)
    a = -0.25 * dt * (sigma**2 * i**2 - r * i)
    b = 1 + 0.5 * dt * (sigma**2 * i**2 + r)
    c = -0.25 * dt * (sigma**2 * i**2 + r * i)

    d = 0.25 * dt * (sigma**2 * i**2 - r * i)
    e = 1 - 0.5 * dt * (sigma**2 * i**2 + r)
    f = 0.25 * dt * (sigma**2 * i**2 + r * i)

    lhs = np.zeros((M - 1, M - 1))
    rhs_matrix = np.zeros((M - 1, M - 1))

    for k in range(M - 1):
        if k > 0:
            lhs[k, k - 1] = a[k]
            rhs_matrix[k, k - 1] = d[k]
        lhs[k, k] = b[k]
        rhs_matrix[k, k] = e[k]
        if k < M - 2:
            lhs[k, k + 1] = c[k]
            rhs_matrix[k, k + 1] = f[k]

    for j in range(N - 1, -1, -1):
        rhs = rhs_matrix @ values[1:M, j + 1]
        rhs[0] += d[0] * values[0, j + 1] - a[0] * values[0, j]
        rhs[-1] += f[-1] * values[M, j + 1] - c[-1] * values[M, j]

        values[1:M, j] = np.linalg.solve(lhs, rhs)

    price = np.interp(S0, s_grid, values[:, 0])
    return price, s_grid, t_grid, values


def compute_impicit_call(S0, K, T, r, sigma, S_max, M, N):
    """Compatibility wrapper for the original misspelled explicit FD function."""
    return explicit_fd_call(S0, K, T, r, sigma, S_max, M, N)


def compute_CN_call(S0, K, T, r, sigma, S_max, M, N):
    """Compatibility wrapper for the original Crank-Nicolson function name."""
    return crank_nicolson_call(S0, K, T, r, sigma, S_max, M, N)

