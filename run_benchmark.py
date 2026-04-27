"""Run the volatility-sweep benchmark and save the results to CSV."""

import argparse
import csv
import json
import math
from pathlib import Path

from src.benchmarking import (
    CNGrid,
    MonteCarloConfig,
    OptionParams,
    run_volatility_benchmark,
    summarize_benchmark,
    volatility_grid,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_sigma_list(value):
    """Parse a comma-separated volatility list."""
    if not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",")]


def parse_args():
    """Parse command-line benchmark settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results") / "volatility_benchmark.csv")
    parser.add_argument("--summary-output", type=Path, default=Path("results") / "benchmark_summary.csv")
    parser.add_argument("--metadata-output", type=Path, default=Path("results") / "benchmark_metadata.json")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--sigma-start", type=float, default=0.10)
    parser.add_argument("--sigma-stop", type=float, default=0.45)
    parser.add_argument("--sigma-step", type=float, default=0.01)
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=110.0)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--r", type=float, default=0.035)
    parser.add_argument("--S-max", type=float, default=600.0)
    parser.add_argument("--M", type=int, default=300)
    parser.add_argument("--N", type=int, default=300)
    parser.add_argument("--mc-paths", type=int, default=100000)
    parser.add_argument("--mc-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--pinn-mode",
        choices=["none", "selected", "all"],
        default="none",
        help="'none': skip PINN; 'selected': run PINN only for --pinn-sigmas; "
             "'all': run PINN for every sigma (slow ~30 min, use --pinn-save-checkpoints for resumability)",
    )
    parser.add_argument("--pinn-sigmas", type=parse_sigma_list, default=parse_sigma_list("0.10,0.25,0.40"))
    parser.add_argument("--pinn-epochs", type=int, default=5000)
    parser.add_argument("--pinn-hidden-dim", type=int, default=64)
    parser.add_argument("--pinn-hidden-layers", type=int, default=3)
    parser.add_argument("--pinn-value-scale", type=float, default=None)
    parser.add_argument("--pinn-pde-weight", type=float, default=1.0)
    parser.add_argument("--pinn-terminal-weight", type=float, default=5.0)
    parser.add_argument("--pinn-left-boundary-weight", type=float, default=1.0)
    parser.add_argument("--pinn-right-boundary-weight", type=float, default=5.0)
    parser.add_argument("--pinn-checkpoint-dir", type=Path, default=Path("results") / "pinn_checkpoints")
    parser.add_argument("--pinn-save-checkpoints", action="store_true")
    parser.add_argument("--pinn-load-checkpoints", action="store_true")
    return parser.parse_args()


def pinn_checkpoint_path(checkpoint_dir, sigma, S_max=None, value_scale=None):
    """Return the checkpoint path for one fixed-volatility PINN."""
    suffix = f"_Smax_{S_max:.1f}" if S_max is not None else ""
    scale_suffix = f"_scale_{value_scale:.1f}" if value_scale is not None else ""
    return checkpoint_dir / f"pinn_sigma_{sigma:.4f}{suffix}{scale_suffix}.pt"


def build_pinn_pricer(args, params):
    """Return a PINN pricing callable, or None when PINN benchmarking is disabled."""
    if args.pinn_mode == "none":
        return None

    import torch

    from src.pinn import LossWeights, PINN, price_pinn, train_pinn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_scale = float(args.pinn_value_scale if args.pinn_value_scale is not None else params.K)
    loss_weights = LossWeights(
        pde=args.pinn_pde_weight,
        terminal=args.pinn_terminal_weight,
        boundary_left=args.pinn_left_boundary_weight,
        boundary_right=args.pinn_right_boundary_weight,
    )

    def pricer(sigma):
        # Current PINN architecture is not volatility-conditioned, so each sigma needs its own fit.
        torch.manual_seed(args.seed)
        model = PINN(hidden_dim=args.pinn_hidden_dim, num_hidden=args.pinn_hidden_layers).to(device)
        checkpoint_path = pinn_checkpoint_path(args.pinn_checkpoint_dir, sigma, args.S_max, value_scale)

        if args.pinn_load_checkpoints and checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            train_pinn(
                model,
                K=params.K,
                T=params.T,
                r=params.r,
                sigma=sigma,
                S_max=args.S_max,
                epochs=args.pinn_epochs,
                device=device,
                print_every=0,
                value_scale=value_scale,
                loss_weights=loss_weights,
            )
            if args.pinn_save_checkpoints:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

        return price_pinn(model, params.S0, 0.0, args.S_max, params.T, device=device, value_scale=value_scale)

    return pricer


def write_csv(rows, output_path):
    """Write benchmark rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(args, sigmas, pinn_sigmas):
    """Write benchmark settings to a JSON metadata file."""
    metadata = {
        "option_params": {"S0": args.S0, "K": args.K, "T": args.T, "r": args.r},
        "sigma_grid": [float(sigma) for sigma in sigmas],
        "cn_grid": {"S_max": args.S_max, "M": args.M, "N": args.N},
        "monte_carlo": {
            "n_paths": args.mc_paths,
            "n_repeats": args.mc_repeats,
            "seed": args.seed,
        },
        "pinn": {
            "mode": args.pinn_mode,
            "selected_sigmas": [float(s) for s in sigmas] if args.pinn_mode == "all" else ([] if pinn_sigmas is None else [float(sigma) for sigma in pinn_sigmas]),
            "epochs": args.pinn_epochs,
            "hidden_dim": args.pinn_hidden_dim,
            "hidden_layers": args.pinn_hidden_layers,
            "value_scale": args.pinn_value_scale if args.pinn_value_scale is not None else args.K,
            "loss_weights": {
                "pde": args.pinn_pde_weight,
                "terminal": args.pinn_terminal_weight,
                "boundary_left": args.pinn_left_boundary_weight,
                "boundary_right": args.pinn_right_boundary_weight,
            },
            "checkpoint_dir": str(args.pinn_checkpoint_dir),
            "save_checkpoints": args.pinn_save_checkpoints,
            "load_checkpoints": args.pinn_load_checkpoints,
        },
        "outputs": {
            "benchmark_csv": str(args.output),
            "summary_csv": str(args.summary_output),
            "metadata_json": str(args.metadata_output),
            "figures_dir": str(args.figures_dir),
        },
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def resolve_output_path(path):
    """Resolve relative output paths from the thesis project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    args.output = resolve_output_path(args.output)
    args.summary_output = resolve_output_path(args.summary_output)
    args.metadata_output = resolve_output_path(args.metadata_output)
    args.figures_dir = resolve_output_path(args.figures_dir)
    args.pinn_checkpoint_dir = resolve_output_path(args.pinn_checkpoint_dir)

    params = OptionParams(S0=args.S0, K=args.K, T=args.T, r=args.r)
    cn_grid = CNGrid(S_max=args.S_max, M=args.M, N=args.N)
    mc_config = MonteCarloConfig(n_paths=args.mc_paths, n_repeats=args.mc_repeats, seed=args.seed)
    sigmas = volatility_grid(args.sigma_start, args.sigma_stop, args.sigma_step)
    pinn_sigmas = None
    if args.pinn_mode == "selected":
        pinn_sigmas = []
        ignored_sigmas = []
        for sigma in args.pinn_sigmas:
            if any(math.isclose(sigma, grid_sigma) for grid_sigma in sigmas):
                pinn_sigmas.append(sigma)
            else:
                ignored_sigmas.append(sigma)
        if ignored_sigmas:
            print(f"Ignoring PINN sigmas not present in the main grid: {ignored_sigmas}")
    elif args.pinn_mode == "all":
        pinn_sigmas = None  # run PINN for every sigma in the grid

    rows = run_volatility_benchmark(
        params=params,
        sigmas=sigmas,
        cn_grid=cn_grid,
        mc_config=mc_config,
        pinn_pricer=build_pinn_pricer(args, params),
        pinn_sigmas=pinn_sigmas,
    )
    write_csv(rows, args.output)
    write_csv(summarize_benchmark(rows), args.summary_output)
    write_metadata(args, sigmas, pinn_sigmas)

    if args.make_figures:
        from src.plots import save_benchmark_figures

        save_benchmark_figures(rows, args.figures_dir)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote benchmark summary to {args.summary_output}")
    print(f"Wrote benchmark metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
