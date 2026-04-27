# European Call Option Pricing — Thesis Benchmark

This folder contains the full computational component of the thesis. It prices a vanilla European call option using four methods — Black-Scholes (analytical), Monte Carlo, Crank-Nicolson finite differences, and Physics-Informed Neural Networks (PINNs) — and compares them on accuracy, computational cost, and reliability across a volatility range of σ ∈ [0.10, 0.45].

For code structure and architecture see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Dependencies

```bash
pip install numpy scipy matplotlib torch
```

All experiments were run inside a dedicated conda environment. Activate it before running any script.

---

## Quickstart — Run Everything

```bash
python run_all.py
```

This runs the volatility benchmark, efficiency experiments, and generates all figures. PINN is excluded by default (fast, ~2 min). To include the full 4-way PINN comparison:

```bash
# First run: trains all 36 PINN models and saves checkpoints (~35 min)
python run_all.py --pinn-mode all --pinn-save-checkpoints --run-pinn-reliability

# Subsequent runs: reloads checkpoints, skips retraining (~2 min)
python run_all.py --pinn-mode all --pinn-load-checkpoints --run-pinn-reliability
```

---

## Run Scripts

There are four runnable scripts. `run_all.py` calls the other three in sequence.

### `run_all.py` — main entry point

| Flag | Effect |
|------|--------|
| `--pinn-mode none` | Skip PINN (default, fast) |
| `--pinn-mode selected` | PINN at σ = 0.10, 0.25, 0.40 only |
| `--pinn-mode all` | PINN at all 36 σ values (~35 min) |
| `--pinn-save-checkpoints` | Save trained models to `results/pinn_checkpoints/` |
| `--pinn-load-checkpoints` | Reload saved models instead of retraining |
| `--run-pinn-reliability` | Also run the deep PINN reliability study |
| `--skip-efficiency` | Skip MC convergence and CN refinement experiments |

---

### `run_benchmark.py` — volatility sweep

Sweeps σ from 0.10 to 0.45 in steps of 0.01 (36 values). Prices each with BS, MC, CN, and optionally PINN. Writes:

```
results/volatility_benchmark.csv    — full per-sigma results
results/benchmark_summary.csv       — MAE, RMSE, runtime per method
results/benchmark_metadata.json     — full experiment configuration
figures/price_vs_volatility.png
figures/abs_error_vs_volatility.png
figures/relative_error_vs_volatility.png
figures/runtime_vs_volatility.png
figures/mc_std_vs_volatility.png
```

Default parameters: S₀=100, K=110, T=2yr, r=3.5%, MC 100k paths × 10 repeats, CN 300×300 grid.

---

### `run_efficiency_experiments.py` — convergence study

Measures how accuracy scales with computational effort. Fixed at σ=0.25. Writes:

```
results/mc_convergence.csv          — error vs path count (1k → 1M paths)
results/cn_refinement.csv           — error vs grid size (M=N: 50 → 500)
results/efficiency_summary.csv      — aggregate stats
figures/mc_convergence_error_vs_paths.png
figures/cn_refinement_error_vs_grid.png
figures/error_vs_runtime.png
figures/mc_error_vs_runtime.png
figures/cn_error_vs_runtime.png
```

---

### `run_pinn_reliability.py` — PINN deep study

Trains PINNs at σ ∈ {0.10, 0.25, 0.40} and runs three analyses: t=0 curve evaluation, repeated-seed variability, and training-effort sensitivity. Writes:

```
results/pinn_reliability.csv        — curve and spot errors per sigma
results/pinn_pointwise_errors.csv   — per-stock-price error at t=0
results/pinn_repeated_runs.csv      — 3 seeds × 3 sigmas
results/pinn_repeated_summary.csv   — mean/std error per sigma
results/pinn_effort.csv             — error vs epochs (1k, 2.5k, 5k)
figures/pinn_t0_curve_sigma_0.10.png
figures/pinn_t0_curve_sigma_0.25.png
figures/pinn_t0_curve_sigma_0.40.png
figures/pinn_error_vs_stock.png
figures/pinn_reliability_summary.png
figures/pinn_effort_error_vs_runtime.png
figures/error_vs_runtime_with_pinn.png
```

---

## Figures Guide

The figures are organised into four thesis blocks:

### Block 1 — Validation
| Figure | What it shows |
|--------|---------------|
| `price_vs_volatility.png` | All four methods pricing the same option. BS, MC, CN agree closely; PINN overestimates, especially at low σ. |

### Block 2 — Accuracy
| Figure | What it shows |
|--------|---------------|
| `abs_error_vs_volatility.png` | Log-scale error vs σ with MC ±1σ shaded band. Key finding: MC error rises with σ, CN error falls — they cross around σ≈0.12. PINN is 2–3 orders of magnitude worse. |
| `relative_error_vs_volatility.png` | Same story in percentage terms. |
| `runtime_vs_volatility.png` | Bar chart of mean runtime per method (log scale). Six orders of magnitude span from BS (~0.1ms) to PINN (~114s). |

### Block 3 — Efficiency
| Figure | What it shows |
|--------|---------------|
| `mc_convergence_error_vs_paths.png` | MC error vs path count with O(1/√N) reference slope. |
| `cn_refinement_error_vs_grid.png` | CN error vs grid size with O(1/M²) reference slope. |
| `error_vs_runtime.png` | MC vs CN efficiency frontier. CN dominates. |
| `error_vs_runtime_with_pinn.png` | Full four-method frontier. PINN sits in the top-right (expensive and inaccurate). |

### Block 4 — PINN Deep Dive
| Figure | What it shows |
|--------|---------------|
| `pinn_t0_curve_sigma_0.25.png` | PINN tracks BS well at moderate volatility. |
| `pinn_t0_curve_sigma_0.10.png` | Visual fit looks good but S₀ marker reveals 158% spot error — a gradient-sharpness failure near the ATM region. |
| `pinn_error_vs_stock.png` | Pointwise error across stock prices. Error spikes at S≈50–150 for low σ. |
| `pinn_reliability_summary.png` | Mean ± std error across 3 seeds with individual seed points. High variance at low σ shows training instability. |

---

## Key Results

| Method | MAE | Mean Rel. Error | Mean Runtime |
|--------|-----|-----------------|--------------|
| Black-Scholes | — (reference) | — | ~0.1 ms |
| Crank-Nicolson | 0.005 | 0.06% | 80 ms |
| Monte Carlo | 0.044 | 0.30% | 16 ms |
| PINN | 2.73 | 32.6% | 114 s |

CN is 8.6× more accurate than MC at 5× the runtime. PINN is 536× less accurate than CN while taking ~1,400× longer.

---

## Notebooks

The notebooks provide a narrative walkthrough and should be read alongside the figures:

- `notebooks/01_pricing_comparison.ipynb` — visual comparison of BS, MC, CN across volatility
- `notebooks/02_pinn_reliability.ipynb` — PINN diagnostics with explanatory commentary on the σ=0.10 failure and PINN's positioning as a research-direction method
