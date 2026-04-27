"""Run fixed-volatility PINN reliability checks and save thesis outputs."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.black_scholes import black_scholes_call
from src.metrics import mae, max_absolute_error, rmse
from src.pinn import LossWeights, PINN, compare_t0_curve, plot_t0_comparison, price_pinn, train_pinn
from src.plots import save_error_vs_runtime, save_pinn_error_by_stock, save_pinn_reliability_summary
from run_benchmark import parse_sigma_list, pinn_checkpoint_path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    """Parse PINN reliability settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results") / "pinn_reliability.csv")
    parser.add_argument("--pointwise-output", type=Path, default=Path("results") / "pinn_pointwise_errors.csv")
    parser.add_argument("--repeat-output", type=Path, default=Path("results") / "pinn_repeated_runs.csv")
    parser.add_argument("--repeat-summary-output", type=Path, default=Path("results") / "pinn_repeated_summary.csv")
    parser.add_argument("--effort-output", type=Path, default=Path("results") / "pinn_effort.csv")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--sigmas", type=parse_sigma_list, default=parse_sigma_list("0.10,0.25,0.40"))
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=110.0)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--r", type=float, default=0.035)
    parser.add_argument("--S-max", type=float, default=700.0)
    parser.add_argument("--curve-S-max", type=float, default=None)
    parser.add_argument("--curve-extension", type=float, default=1.08)
    parser.add_argument("--curve-points", type=int, default=201)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--value-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeat-seeds", type=parse_int_list, default=parse_int_list("101,202,303"))
    parser.add_argument("--repeat-epochs", type=int, default=5000)
    parser.add_argument("--effort-epochs", type=parse_int_list, default=parse_int_list("1000,2500,5000"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results") / "pinn_checkpoints")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--load-checkpoints", action="store_true")
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=5.0)
    parser.add_argument("--left-boundary-weight", type=float, default=1.0)
    parser.add_argument("--right-boundary-weight", type=float, default=5.0)
    return parser.parse_args()


def parse_int_list(value):
    """Parse a comma-separated integer list."""
    if isinstance(value, list):
        return value
    if not value.strip() or value.strip().lower() in {"none", "null"}:
        return []
    return [int(item.strip()) for item in value.split(",")]


def save_curve_figure(comparison, sigma, figures_dir, training_S_max, S0=None):
    """Save the t=0 PINN-vs-Black-Scholes curve figure for one volatility."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": ":",
            "grid.alpha": 0.45,
        }
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    ax = plot_t0_comparison(comparison)
    ax.axvline(training_S_max, color="gray", linestyle=":", label="training boundary")
    if S0 is not None:
        ax.axvline(S0, color="navy", linestyle="--", linewidth=1.2, label=f"S₀={S0:.0f} (evaluation point)")
    ax.set_title(f"PINN vs Black-Scholes at t=0, sigma={sigma:.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"pinn_t0_curve_sigma_{sigma:.2f}.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_csv(rows, output_path):
    """Write reliability rows to CSV."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path):
    """Read CSV rows if the file exists."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def region_error_metrics(comparison, mask):
    """Compute error metrics on a selected region of the t=0 curve."""
    errors = comparison["error"][mask]
    if len(errors) == 0:
        return {"mae": np.nan, "rmse": np.nan, "max_abs_error": np.nan}
    return {
        "mae": mae(errors),
        "rmse": rmse(errors),
        "max_abs_error": max_absolute_error(errors),
    }


def pointwise_error_rows(comparison, sigma, training_S_max):
    """Return pointwise PINN error rows along the t=0 stock grid."""
    rows = []
    for S, pinn_price, analytical_price, abs_error in zip(
        comparison["S"],
        comparison["pinn_price"],
        comparison["analytical_price"],
        comparison["abs_error"],
    ):
        rel_error = abs_error / abs(analytical_price) if abs(analytical_price) > 1e-12 else np.nan
        rows.append(
            {
                "sigma": float(sigma),
                "S": float(S),
                "pinn_price": float(pinn_price),
                "analytical_price": float(analytical_price),
                "abs_error": float(abs_error),
                "rel_error": float(rel_error) if np.isfinite(rel_error) else np.nan,
                "region": "in_domain" if S <= training_S_max else "extrapolation",
            }
        )
    return rows


def train_single_pinn(args, sigma, seed, value_scale, loss_weights, device, epochs):
    """Train one fixed-volatility PINN and return model plus metrics."""
    torch.manual_seed(seed)
    model = PINN(hidden_dim=args.hidden_dim, num_hidden=args.hidden_layers).to(device)
    model, metrics = train_pinn(
        model,
        K=args.K,
        T=args.T,
        r=args.r,
        sigma=sigma,
        S_max=args.S_max,
        epochs=epochs,
        device=device,
        print_every=0,
        seed=seed,
        loss_weights=loss_weights,
        value_scale=value_scale,
    )
    return model, metrics


def summarize_repeated_runs(rows):
    """Summarize repeated PINN runs by volatility."""
    summary = []
    for sigma in sorted({float(row["sigma"]) for row in rows}):
        sigma_rows = [row for row in rows if float(row["sigma"]) == sigma]
        abs_errors = np.array([float(row["abs_error"]) for row in sigma_rows], dtype=float)
        runtimes = np.array([float(row["runtime_sec"]) for row in sigma_rows], dtype=float)
        summary.append(
            {
                "sigma": sigma,
                "n_runs": len(sigma_rows),
                "mean_abs_error": float(np.mean(abs_errors)),
                "std_abs_error": float(np.std(abs_errors, ddof=1)) if len(abs_errors) > 1 else 0.0,
                "min_abs_error": float(np.min(abs_errors)),
                "max_abs_error": float(np.max(abs_errors)),
                "mean_runtime_sec": float(np.mean(runtimes)),
                "std_runtime_sec": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
            }
        )
    return summary


def resolve_output_path(path):
    """Resolve relative output paths from the thesis project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    args.output = resolve_output_path(args.output)
    args.pointwise_output = resolve_output_path(args.pointwise_output)
    args.repeat_output = resolve_output_path(args.repeat_output)
    args.repeat_summary_output = resolve_output_path(args.repeat_summary_output)
    args.effort_output = resolve_output_path(args.effort_output)
    args.figures_dir = resolve_output_path(args.figures_dir)
    args.checkpoint_dir = resolve_output_path(args.checkpoint_dir)
    curve_S_max = args.curve_S_max if args.curve_S_max is not None else args.S_max * args.curve_extension
    loss_weights = LossWeights(
        pde=args.pde_weight,
        terminal=args.terminal_weight,
        boundary_left=args.left_boundary_weight,
        boundary_right=args.right_boundary_weight,
    )
    value_scale = float(args.value_scale if args.value_scale is not None else args.K)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_curve = np.linspace(1e-8, curve_S_max, args.curve_points)
    rows = []
    pointwise_rows = []

    for sigma in args.sigmas:
        sigma = float(sigma)
        torch.manual_seed(args.seed)
        model = PINN(hidden_dim=args.hidden_dim, num_hidden=args.hidden_layers).to(device)
        checkpoint_path = pinn_checkpoint_path(args.checkpoint_dir, sigma, args.S_max, value_scale)

        if args.load_checkpoints and checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            metrics = {"runtime_sec": 0.0, "final_total_loss": np.nan}
        else:
            model, metrics = train_pinn(
                model,
                K=args.K,
                T=args.T,
                r=args.r,
                sigma=sigma,
                S_max=args.S_max,
                epochs=args.epochs,
                device=device,
                print_every=0,
                seed=args.seed,
                loss_weights=loss_weights,
                value_scale=value_scale,
            )
            if args.save_checkpoints:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

        comparison = compare_t0_curve(
            model,
            S_curve,
            args.K,
            args.T,
            args.r,
            sigma,
            args.S_max,
            device=device,
            value_scale=value_scale,
        )
        in_domain_metrics = region_error_metrics(comparison, comparison["S"] <= args.S_max)
        extrapolation_metrics = region_error_metrics(comparison, comparison["S"] > args.S_max)
        pinn_spot = price_pinn(
            model,
            args.S0,
            0.0,
            args.S_max,
            args.T,
            device=device,
            value_scale=value_scale,
        )
        analytical_spot = black_scholes_call(args.S0, args.K, args.T, args.r, sigma)
        spot_abs_error = abs(pinn_spot - analytical_spot)
        spot_rel_error = spot_abs_error / abs(analytical_spot)

        rows.append(
            {
                "sigma": sigma,
                "pinn_price_at_S0": pinn_spot,
                "analytical_price_at_S0": analytical_spot,
                "spot_abs_error": spot_abs_error,
                "spot_rel_error": spot_rel_error,
                "curve_mae": comparison["mae"],
                "curve_rmse": comparison["rmse"],
                "curve_max_abs_error": comparison["max_abs_error"],
                "in_domain_mae": in_domain_metrics["mae"],
                "in_domain_rmse": in_domain_metrics["rmse"],
                "in_domain_max_abs_error": in_domain_metrics["max_abs_error"],
                "extrapolation_mae": extrapolation_metrics["mae"],
                "extrapolation_rmse": extrapolation_metrics["rmse"],
                "extrapolation_max_abs_error": extrapolation_metrics["max_abs_error"],
                "training_S_max": args.S_max,
                "curve_S_max": curve_S_max,
                "extrapolation_width": max(0.0, curve_S_max - args.S_max),
                "value_scale": value_scale,
                "runtime_sec": metrics["runtime_sec"],
                "final_loss": metrics["final_total_loss"],
            }
        )
        pointwise_rows.extend(pointwise_error_rows(comparison, sigma, args.S_max))
        save_curve_figure(comparison, sigma, args.figures_dir, args.S_max, S0=args.S0)

    write_csv(rows, args.output)
    write_csv(pointwise_rows, args.pointwise_output)
    save_pinn_error_by_stock(pointwise_rows, args.figures_dir)

    repeated_rows = []
    if args.repeat_seeds:
        repeat_epochs = args.repeat_epochs if args.repeat_epochs is not None else args.epochs
        for sigma in args.sigmas:
            sigma = float(sigma)
            analytical_spot = black_scholes_call(args.S0, args.K, args.T, args.r, sigma)
            for seed in args.repeat_seeds:
                model, metrics = train_single_pinn(
                    args,
                    sigma,
                    seed,
                    value_scale,
                    loss_weights,
                    device,
                    repeat_epochs,
                )
                pinn_spot = price_pinn(
                    model,
                    args.S0,
                    0.0,
                    args.S_max,
                    args.T,
                    device=device,
                    value_scale=value_scale,
                )
                abs_error = abs(pinn_spot - analytical_spot)
                repeated_rows.append(
                    {
                        "sigma": sigma,
                        "seed": int(seed),
                        "epochs": int(repeat_epochs),
                        "pinn_price_at_S0": pinn_spot,
                        "analytical_price_at_S0": analytical_spot,
                        "abs_error": abs_error,
                        "rel_error": abs_error / abs(analytical_spot),
                        "runtime_sec": metrics["runtime_sec"],
                        "final_loss": metrics["final_total_loss"],
                        "value_scale": value_scale,
                    }
                )

        repeated_summary_rows = summarize_repeated_runs(repeated_rows)
        write_csv(repeated_rows, args.repeat_output)
        write_csv(repeated_summary_rows, args.repeat_summary_output)
        save_pinn_reliability_summary(repeated_summary_rows, args.figures_dir, individual_rows=repeated_rows)

    effort_rows = []
    if args.effort_epochs:
        for sigma in args.sigmas:
            sigma = float(sigma)
            analytical_spot = black_scholes_call(args.S0, args.K, args.T, args.r, sigma)
            for epochs in args.effort_epochs:
                model, metrics = train_single_pinn(
                    args,
                    sigma,
                    args.seed,
                    value_scale,
                    loss_weights,
                    device,
                    int(epochs),
                )
                pinn_spot = price_pinn(
                    model,
                    args.S0,
                    0.0,
                    args.S_max,
                    args.T,
                    device=device,
                    value_scale=value_scale,
                )
                abs_error = abs(pinn_spot - analytical_spot)
                effort_rows.append(
                    {
                        "method": f"PINN sigma={sigma:.2f}",
                        "sigma": sigma,
                        "epochs": int(epochs),
                        "pinn_price_at_S0": pinn_spot,
                        "analytical_price_at_S0": analytical_spot,
                        "abs_error": abs_error,
                        "rel_error": abs_error / abs(analytical_spot),
                        "runtime_sec": metrics["runtime_sec"],
                        "final_loss": metrics["final_total_loss"],
                        "value_scale": value_scale,
                    }
                )

        write_csv(effort_rows, args.effort_output)
        save_error_vs_runtime(effort_rows, args.figures_dir, filename="pinn_effort_error_vs_runtime.png")
        existing_results_dir = PROJECT_ROOT / "results"
        combined_efficiency_rows = (
            read_csv_rows(existing_results_dir / "mc_convergence.csv")
            + read_csv_rows(existing_results_dir / "cn_refinement.csv")
            + effort_rows
        )
        if combined_efficiency_rows:
            save_error_vs_runtime(
                combined_efficiency_rows,
                args.figures_dir,
                filename="error_vs_runtime_with_pinn.png",
            )

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote {len(pointwise_rows)} pointwise error rows to {args.pointwise_output}")
    if repeated_rows:
        print(f"Wrote {len(repeated_rows)} repeated PINN rows to {args.repeat_output}")
        print(f"Wrote repeated PINN summary to {args.repeat_summary_output}")
    if effort_rows:
        print(f"Wrote {len(effort_rows)} PINN effort rows to {args.effort_output}")
    print(f"Wrote PINN reliability figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
