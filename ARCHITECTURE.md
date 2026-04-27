# Code Architecture

This document describes the internal structure, data flow, and design decisions of the thesis codebase. For running instructions and output descriptions see [README.md](README.md).

---

## Module Map

```
Thesis/
├── run_all.py                  ← single entry point, orchestrates the three runners
├── run_benchmark.py            ← volatility sweep (BS, MC, CN, PINN)
├── run_efficiency_experiments.py ← MC convergence + CN refinement study
├── run_pinn_reliability.py     ← PINN t=0 curve, seed variability, training effort
│
├── src/
│   ├── black_scholes.py        ← analytical pricer
│   ├── monte_carlo.py          ← simulation pricers (European, Asian, Basket)
│   ├── finite_differences.py   ← explicit FD + Crank-Nicolson solvers
│   ├── pinn.py                 ← PINN architecture, training loop, evaluation
│   ├── benchmarking.py         ← pipeline glue: loops, aggregation, dataclasses
│   ├── metrics.py              ← MAE, RMSE, max absolute error
│   └── plots.py                ← all figure-generation functions
│
├── notebooks/
│   ├── 01_pricing_comparison.ipynb
│   └── 02_pinn_reliability.ipynb
│
├── results/                    ← CSV + JSON outputs (git-ignored)
└── figures/                    ← PNG outputs (git-ignored)
```

---

## Dependency Graph

```
run_all.py
  └── subprocess calls:
        run_benchmark.py
          ├── src.benchmarking   (pipeline loop)
          │     ├── src.black_scholes
          │     ├── src.monte_carlo
          │     ├── src.finite_differences
          │     ├── src.pinn          (optional, --pinn-mode)
          │     └── src.metrics
          └── src.plots
        run_efficiency_experiments.py
          ├── src.experiments    (convergence sweeps)
          │     ├── src.benchmarking
          │     └── src.metrics
          └── src.plots
        run_pinn_reliability.py
          ├── src.pinn
          ├── src.black_scholes
          ├── src.metrics
          └── src.plots
```

`run_all.py` uses `subprocess` rather than direct imports to avoid namespace collisions and to keep each script independently runnable.

---

## src/ Module Reference

### `black_scholes.py`

Two functions wrapping the same closed-form formula:

- `black_scholes_call(S0, K, T, r, sigma)` — primary interface
- `compute_bs_call(r, sigma, S0, T, K)` — reordered parameters for compatibility with older call sites

Used as the ground-truth reference throughout. All errors are measured against this.

---

### `monte_carlo.py`

Three independent pricers, all using risk-neutral GBM:

- `monte_carlo_european_call(S0, K, T, r, sigma, n_paths, seed)` — used in benchmarks
- `monte_carlo_asian_call(S0, K, T, r, sigma, n_paths, n_steps, seed)` — path-dependent, arithmetic average
- `monte_carlo_basket_call(s1, s2, K, T, r, sigma1, sigma2, corr, n_paths, seed)` — two-asset, Cholesky decomposition for correlated normals

The Asian and Basket pricers are present for completeness but are not used in benchmark comparisons. They represent the natural use case where MC has no viable alternatives (no closed form, CN infeasible for path-dependence or high dimension).

No variance reduction techniques (antithetic variates, control variates) are implemented.

---

### `finite_differences.py`

Two solvers for the Black-Scholes PDE on a uniform grid over S ∈ [0, S_max]:

- `explicit_fd_call(S0, K, T, r, sigma, S_max, M, N)` — explicit scheme, conditionally stable (CFL condition not enforced; present for reference only)
- `crank_nicolson_call(S0, K, T, r, sigma, S_max, M, N)` — CN scheme, unconditionally stable, O(Δt², ΔS²) convergence; used in all benchmarks

**How CN works internally:**
1. Build S-grid: ΔS = S_max / (M+1), giving M interior points
2. Terminal condition: V(Sᵢ, T) = max(Sᵢ − K, 0)
3. Boundary conditions: V(0, t) = 0; V(S_max, t) = S_max − K·e^{−r(T−t)}
4. March backwards N steps from t=T to t=0, solving a tridiagonal system at each step
5. Interpolate the grid at S₀ to return the price

**Grid alignment note:** The benchmark uses S_max=600, M=300, giving ΔS=2.0. With S₀=100 and ΔS=2, S₀ falls exactly on a grid node (index 50), eliminating interpolation error. The CN refinement experiment reveals a related artefact: at M=150 (ΔS=4.0) the error is near-zero for the same reason, while at M=200 (ΔS=3.0, S₀=100/3=33.33 grid steps) interpolation is required and error increases. This is documented in the refinement figure but does not affect the main benchmark.

Returns a tuple: `(price, grid, S_grid, t_grid)`. Only `price` is used in the benchmark.

---

### `pinn.py`

**Architecture:**

```
Input: [S/S_max, t/T]  →  shape (batch, 2)
Hidden: num_hidden=3 layers, hidden_dim=64 neurons each, Tanh activation
Output: scaled price V/value_scale  →  shape (batch, 1)
```

Input normalisation maps S to [0,1] and t to [0,1]. Output is scaled by `value_scale` (default K=110) so the network learns V/K rather than V — this improves numerical conditioning because V/K ∈ [0, ~5] rather than [0, ~550].

**Loss function** (four components):

| Term | Weight (default) | Description |
|------|-----------------|-------------|
| PDE residual | 1.0 | BS PDE enforced at 2000 random interior points |
| Terminal | 5.0 | Payoff max(S−K,0) enforced at 800 points at t=T |
| Left boundary | 1.0 | V(0,t)=0 at 400 points |
| Right boundary | 5.0 | V(S_max,t)=S_max−Ke^{−r(T−t)} at 400 points |

Terminal and right boundary are up-weighted (5×) because the call payoff is concentrated there. The PDE residual uses automatic differentiation (torch autograd) to compute ∂V/∂t, ∂V/∂S, ∂²V/∂S².

**Design limitation:** Each model is trained for a single fixed σ. σ is not an input to the network. A volatility-conditioned design — adding σ as a third input — would allow one training run to cover the full parameter space, which is the natural extension of this work.

**Key functions:**
- `train_pinn(model, K, T, r, sigma, S_max, epochs, ...)` → returns trained model + metrics dict
- `price_pinn(model, S0, t0, S_max, T, ...)` → single-point price at (S₀, t=0)
- `compare_t0_curve(model, S_curve, ...)` → full t=0 curve comparison against BS
- `plot_t0_comparison(comparison, ax)` → matplotlib axes with PINN vs BS curve

**Checkpointing:** Model state dicts saved as `.pt` files named `pinn_sigma_{σ:.4f}_Smax_{S_max:.1f}_scale_{value_scale:.1f}.pt`.

---

### `benchmarking.py`

Pipeline glue. Owns no pricing logic — only orchestration, timing, and aggregation.

**Dataclasses:**

```python
OptionParams(S0=100, K=110, T=2.0, r=0.035)
CNGrid(S_max=600, M=200, N=200)          # defaults; run_benchmark.py overrides to M=N=300
MonteCarloConfig(n_paths=10000, n_repeats=5, seed=12345)  # overridden to 100k/10 in runner
```

**Key functions:**
- `volatility_grid(start, stop, step)` — returns rounded numpy array avoiding float precision drift
- `monte_carlo_repeated(...)` — runs n_repeats MC calls with sequential seeds (seed, seed+1, ...), returns mean, std, total runtime
- `run_volatility_benchmark(...)` — main loop: iterates σ values, calls all pricers, computes errors, returns list of row dicts
- `summarize_benchmark(rows)` — aggregates to per-method MAE, RMSE, max error, mean/max runtime

The PINN pricer is injected as a callable `pinn_pricer(sigma) -> float`, making `benchmarking.py` independent of torch.

---

### `metrics.py`

Three functions, each taking a list/array of error values:

- `mae(errors)` → mean absolute error
- `rmse(errors)` → root mean squared error
- `max_absolute_error(errors)` → max |error|

No confidence intervals or statistical tests.

---

### `plots.py`

All figure generation is centralised here. Figures are never created in the runner scripts directly.

**Style constants:**
```python
METHOD_COLORS = {"Black-Scholes": "#222222", "Monte Carlo": "#1f77b4",
                 "Crank-Nicolson": "#d62728", "PINN": "#2ca02c"}
METHOD_MARKERS = {"Monte Carlo": "o", "Crank-Nicolson": "^", "PINN": "s"}
DPI = 300
```

**Key design:** `_finite_xy(rows, x_col, y_col)` filters NaN values before plotting. This means PINN automatically appears in benchmark figures when data exists and is silently skipped when columns are NaN — no conditional logic needed in the callers.

**Figure functions:**

| Function | Output file(s) |
|----------|----------------|
| `save_benchmark_figures(rows, dir)` | calls the four below |
| `save_price_vs_volatility` | `price_vs_volatility.png` |
| `save_error_vs_volatility(relative=False)` | `abs_error_vs_volatility.png` — log y-scale, MC ±1σ shaded band |
| `save_error_vs_volatility(relative=True)` | `relative_error_vs_volatility.png` |
| `save_runtime_vs_volatility` | `runtime_vs_volatility.png` — log-scale bar chart, mean ± std per method |
| `save_mc_std_vs_volatility` | `mc_std_vs_volatility.png` |
| `save_mc_convergence` | `mc_convergence_error_vs_paths.png` — O(1/√N) reference slope |
| `save_cn_refinement` | `cn_refinement_error_vs_grid.png` — O(1/M²) reference slope |
| `save_error_vs_runtime` | configurable filename — distinct markers per method, Pareto efficient frontier |
| `save_pinn_error_by_stock` | `pinn_error_vs_stock.png` |
| `save_pinn_reliability_summary(rows, dir, individual_rows)` | `pinn_reliability_summary.png` — single panel, error bars + seed dots |

---

## Data Flow

```
run_benchmark.py
    │
    ├─ volatility_grid() → [σ₁, σ₂, ..., σ₃₆]
    │
    └─ run_volatility_benchmark()
          │
          ├── for each σ:
          │     ├── black_scholes_call()      → analytical_price
          │     ├── monte_carlo_repeated()    → mc_price_mean, mc_price_std
          │     ├── crank_nicolson_call()     → cn_price
          │     └── pinn_pricer(σ)           → pinn_price  [optional]
          │
          └── rows: list of dicts, one per σ
                │
                ├── write_csv()              → volatility_benchmark.csv
                ├── summarize_benchmark()    → benchmark_summary.csv
                ├── write_metadata()         → benchmark_metadata.json
                └── save_benchmark_figures() → figures/*.png
```

---

## Configuration Defaults vs Override Points

| Parameter | Default in `benchmarking.py` | Override in `run_benchmark.py` |
|-----------|------------------------------|-------------------------------|
| MC paths | 10,000 | 100,000 |
| MC repeats | 5 | 10 |
| CN M, N | 200 | 300 |
| σ step | 0.03 | 0.01 |
| PINN mode | — | `none` / `selected` / `all` |

The dataclass defaults in `benchmarking.py` are intentionally lighter for use in notebooks and unit tests. The runner scripts use production-grade settings.

---

## PINN Checkpoint Naming

```
results/pinn_checkpoints/pinn_sigma_{σ:.4f}_Smax_{S_max:.1f}_scale_{value_scale:.1f}.pt
```

Example: `pinn_sigma_0.2500_Smax_600.0_scale_110.0.pt`

The checkpoint path encodes S_max and value_scale so models trained with different domain or scaling settings are never mixed up.
