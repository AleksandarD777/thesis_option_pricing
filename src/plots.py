"""Plotting helpers for thesis benchmark outputs."""

from pathlib import Path

import numpy as np

METHOD_COLORS = {
    "Black-Scholes": "#222222",
    "Monte Carlo": "#1f77b4",
    "Crank-Nicolson": "#d62728",
    "PINN": "#2ca02c",
}

METHOD_MARKERS = {
    "Monte Carlo": "o",
    "Crank-Nicolson": "^",
    "PINN": "s",
}

DPI = 300


def _apply_style():
    """Apply a consistent thesis-friendly matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.45,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def _finish_plot(figures_dir, filename):
    """Save the current matplotlib figure with consistent layout settings."""
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(Path(figures_dir) / filename, dpi=DPI, bbox_inches="tight")
    plt.close()


def _finite_xy(rows, x_column, y_column):
    """Return finite x/y arrays from benchmark rows."""
    x_values = []
    y_values = []
    for row in rows:
        try:
            x_value = float(row[x_column])
            y_value = float(row.get(y_column, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(y_value):
            x_values.append(x_value)
            y_values.append(y_value)
    return x_values, y_values


def save_price_vs_volatility(rows, figures_dir):
    """Save the method price comparison across volatility."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    plt.figure(figsize=(9, 5))
    for column, label, style in [
        ("analytical_price", "Black-Scholes", {"color": METHOD_COLORS["Black-Scholes"], "linestyle": "--"}),
        ("mc_price_mean", "Monte Carlo", {"color": METHOD_COLORS["Monte Carlo"], "marker": "o", "alpha": 0.8}),
        ("cn_price", "Crank-Nicolson", {"color": METHOD_COLORS["Crank-Nicolson"]}),
        ("pinn_price", "PINN", {"color": METHOD_COLORS["PINN"], "marker": "s", "linestyle": "None", "markersize": 7}),
    ]:
        x_values, y_values = _finite_xy(rows, "sigma", column)
        if y_values:
            plt.plot(x_values, y_values, label=label, **style)

    plt.xlabel("Volatility")
    plt.ylabel("Call price")
    plt.title("European Call Price vs Volatility")
    plt.legend()
    _finish_plot(figures_dir, "price_vs_volatility.png")


def save_error_vs_volatility(rows, figures_dir, relative=False):
    """Save absolute or relative error by volatility."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    suffix = "rel_error" if relative else "abs_error"
    ylabel = "Relative error" if relative else "Absolute error vs Black-Scholes"

    plt.figure(figsize=(9, 5))

    if not relative:
        mc_x, mc_err, mc_std = [], [], []
        for row in rows:
            try:
                x = float(row["sigma"])
                err = float(row.get("mc_abs_error", np.nan))
                std = float(row.get("mc_price_std", np.nan))
            except (TypeError, ValueError):
                continue
            if np.isfinite(err) and np.isfinite(std):
                mc_x.append(x)
                mc_err.append(err)
                mc_std.append(std)
        if mc_x:
            lower = [max(1e-6, e - s) for e, s in zip(mc_err, mc_std)]
            upper = [e + s for e, s in zip(mc_err, mc_std)]
            plt.fill_between(mc_x, lower, upper, alpha=0.15, color=METHOD_COLORS["Monte Carlo"])

    for column, label, style in [
        (f"mc_{suffix}", "Monte Carlo", {"color": METHOD_COLORS["Monte Carlo"], "marker": "o", "alpha": 0.8}),
        (f"cn_{suffix}", "Crank-Nicolson", {"color": METHOD_COLORS["Crank-Nicolson"]}),
        (f"pinn_{suffix}", "PINN", {"color": METHOD_COLORS["PINN"], "marker": "s", "linestyle": "None", "markersize": 7}),
    ]:
        x_values, y_values = _finite_xy(rows, "sigma", column)
        if y_values:
            plt.plot(x_values, y_values, label=label, **style)

    plt.xlabel("Volatility")
    plt.ylabel(ylabel)
    title = "Relative Error vs Volatility" if relative else "Absolute Error vs Volatility"
    plt.title(title)
    if not relative:
        plt.yscale("log")
    plt.legend()
    filename = "relative_error_vs_volatility.png" if relative else "abs_error_vs_volatility.png"
    _finish_plot(figures_dir, filename)


def save_runtime_vs_volatility(rows, figures_dir):
    """Save mean pricing runtime per method as a bar chart with ±1σ error bars."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    method_configs = [
        ("analytical_runtime_sec", "Black-Scholes"),
        ("mc_runtime_sec", "Monte Carlo"),
        ("cn_runtime_sec", "Crank-Nicolson"),
        ("pinn_runtime_sec", "PINN"),
    ]

    labels, means, stds, colors = [], [], [], []
    for col, label in method_configs:
        _, values = _finite_xy(rows, "sigma", col)
        if values:
            labels.append(label)
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
            colors.append(METHOD_COLORS.get(label, "#333333"))

    plt.figure(figsize=(7, 4))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
            error_kw={"elinewidth": 1.5, "capthick": 1.5})
    plt.yscale("log")
    plt.xticks(x, labels)
    plt.ylabel("Runtime (seconds)")
    plt.title("Mean Pricing Runtime per Method")
    _finish_plot(figures_dir, "runtime_vs_volatility.png")


def save_mc_std_vs_volatility(rows, figures_dir):
    """Save Monte Carlo standard deviation across repeated runs."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    x_values, y_values = _finite_xy(rows, "sigma", "mc_price_std")

    plt.figure(figsize=(9, 5))
    plt.plot(x_values, y_values, marker="o", color=METHOD_COLORS["Monte Carlo"])
    plt.xlabel("Volatility")
    plt.ylabel("Monte Carlo price standard deviation")
    plt.title("Monte Carlo Repeated-Run Variability")
    _finish_plot(figures_dir, "mc_std_vs_volatility.png")


def save_benchmark_figures(rows, figures_dir):
    """Save all benchmark figures used by the thesis comparison."""
    save_price_vs_volatility(rows, figures_dir)
    save_error_vs_volatility(rows, figures_dir, relative=False)
    save_error_vs_volatility(rows, figures_dir, relative=True)
    save_runtime_vs_volatility(rows, figures_dir)
    save_mc_std_vs_volatility(rows, figures_dir)


def save_error_vs_runtime(rows, figures_dir, filename="error_vs_runtime.png"):
    """Save absolute error against runtime with per-method markers and Pareto frontier."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    methods = sorted({row["method"] for row in rows})

    plt.figure(figsize=(10, 6))
    all_points = []

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        x_values, y_values = _finite_xy(method_rows, "runtime_sec", "abs_error")
        if y_values:
            base_method = method.split(" sigma=")[0]
            color = METHOD_COLORS.get(base_method, None)
            marker = METHOD_MARKERS.get(base_method, "o")
            plt.plot(x_values, y_values, marker=marker, label=method, color=color, markersize=7)
            all_points.extend(zip(x_values, y_values))

    if len(all_points) >= 2:
        sorted_points = sorted(all_points, key=lambda p: p[0])
        pareto, min_error = [], float("inf")
        for x, y in sorted_points:
            if y < min_error:
                min_error = y
                pareto.append((x, y))
        if len(pareto) >= 2:
            px, py = zip(*pareto)
            plt.plot(px, py, linestyle="--", color="#aaaaaa", linewidth=1.5,
                     zorder=0, label="efficient frontier")

    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Absolute error vs Black-Scholes")
    plt.title("Accuracy vs Computational Cost")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=8)
    _finish_plot(figures_dir, filename)


def save_mc_convergence(rows, figures_dir):
    """Save Monte Carlo convergence figures."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    x_values, y_values = _finite_xy(rows, "n_paths", "abs_error")
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", color=METHOD_COLORS["Monte Carlo"], label="Monte Carlo")
    if len(x_values) >= 2:
        x_ref = np.array([x_values[0], x_values[-1]], dtype=float)
        y_ref = y_values[-1] * np.sqrt(x_values[-1] / x_ref)
        plt.plot(x_ref, y_ref, linestyle="--", color="gray", alpha=0.6, label="O(1/√N)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of paths")
    plt.ylabel("Absolute error vs Black-Scholes")
    plt.title("Monte Carlo Convergence")
    plt.legend()
    _finish_plot(figures_dir, "mc_convergence_error_vs_paths.png")

    save_error_vs_runtime(rows, figures_dir, filename="mc_error_vs_runtime.png")


def save_cn_refinement(rows, figures_dir):
    """Save Crank-Nicolson refinement figures."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    x_values, y_values = _finite_xy(rows, "grid_size", "abs_error")
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", color=METHOD_COLORS["Crank-Nicolson"], label="Crank-Nicolson")
    if len(x_values) >= 2:
        x_ref = np.array([x_values[0], x_values[-1]], dtype=float)
        y_ref = y_values[-1] * (x_values[-1] / x_ref) ** 2
        plt.plot(x_ref, y_ref, linestyle="--", color="gray", alpha=0.6, label="O(1/M²)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Grid size M=N")
    plt.ylabel("Absolute error vs Black-Scholes")
    plt.title("Crank-Nicolson Grid Refinement")
    plt.legend()
    _finish_plot(figures_dir, "cn_refinement_error_vs_grid.png")

    save_error_vs_runtime(rows, figures_dir, filename="cn_error_vs_runtime.png")


def save_pinn_error_by_stock(rows, figures_dir):
    """Save PINN pointwise t=0 error by stock price."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    sigmas = sorted({float(row["sigma"]) for row in rows})

    plt.figure(figsize=(9, 5))
    for sigma in sigmas:
        sigma_rows = [row for row in rows if float(row["sigma"]) == sigma]
        x_values, y_values = _finite_xy(sigma_rows, "S", "abs_error")
        if y_values:
            plt.plot(x_values, y_values, label=f"sigma={sigma:.2f}")

    plt.xlabel("Stock price")
    plt.ylabel("Absolute error vs Black-Scholes")
    plt.title("PINN t=0 Pointwise Error")
    plt.legend()
    _finish_plot(figures_dir, "pinn_error_vs_stock.png")


def save_pinn_reliability_summary(rows, figures_dir, individual_rows=None):
    """Save PINN reliability: mean ± std error per volatility with individual seed points."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    x_values = [float(row["sigma"]) for row in rows]
    mean_errors = [float(row["mean_abs_error"]) for row in rows]
    std_errors = [float(row["std_abs_error"]) for row in rows]

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        x_values, mean_errors, yerr=std_errors,
        marker="o", capsize=5, linewidth=2,
        color=METHOD_COLORS["PINN"], label="mean ± std (3 seeds)",
    )
    if individual_rows:
        for row in individual_rows:
            plt.scatter(
                float(row["sigma"]), float(row["abs_error"]),
                color=METHOD_COLORS["PINN"], alpha=0.5, s=35, zorder=3,
            )
    plt.xlabel("Volatility")
    plt.ylabel("Absolute error at S₀")
    plt.title("PINN Reliability Across Random Seeds")
    plt.legend()
    _finish_plot(figures_dir, "pinn_reliability_summary.png")
