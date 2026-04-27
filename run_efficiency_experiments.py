"""Run MC convergence and Crank-Nicolson refinement experiments."""

import argparse
import csv
from pathlib import Path

from src.benchmarking import OptionParams
from src.experiments import run_cn_refinement, run_mc_convergence, summarize_experiment
from src.plots import save_cn_refinement, save_error_vs_runtime, save_mc_convergence

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_int_list(value):
    """Parse a comma-separated integer list."""
    if not value.strip() or value.strip().lower() in {"none", "null"}:
        return []
    return [int(item.strip()) for item in value.split(",")]


def parse_args():
    """Parse experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=110.0)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--r", type=float, default=0.035)
    parser.add_argument("--S-max", type=float, default=600.0)
    parser.add_argument(
        "--mc-path-counts",
        type=parse_int_list,
        default=parse_int_list("1000,2500,5000,10000,25000,50000,100000,250000,500000,1000000"),
    )
    parser.add_argument("--mc-repeats", type=int, default=10)
    parser.add_argument(
        "--cn-grid-sizes",
        type=parse_int_list,
        default=parse_int_list("50,75,100,150,200,250,300,400,500"),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def resolve_output_path(path):
    """Resolve relative output paths from the thesis project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv(rows, output_path):
    """Write experiment rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.results_dir = resolve_output_path(args.results_dir)
    args.figures_dir = resolve_output_path(args.figures_dir)
    params = OptionParams(S0=args.S0, K=args.K, T=args.T, r=args.r)

    mc_rows = run_mc_convergence(
        params=params,
        sigma=args.sigma,
        path_counts=args.mc_path_counts,
        n_repeats=args.mc_repeats,
        seed=args.seed,
    )
    cn_rows = run_cn_refinement(
        params=params,
        sigma=args.sigma,
        S_max=args.S_max,
        grid_sizes=args.cn_grid_sizes,
    )
    combined_rows = mc_rows + cn_rows
    summary_rows = summarize_experiment(combined_rows)

    write_csv(mc_rows, args.results_dir / "mc_convergence.csv")
    write_csv(cn_rows, args.results_dir / "cn_refinement.csv")
    write_csv(summary_rows, args.results_dir / "efficiency_summary.csv")

    save_mc_convergence(mc_rows, args.figures_dir)
    save_cn_refinement(cn_rows, args.figures_dir)
    save_error_vs_runtime(combined_rows, args.figures_dir, filename="error_vs_runtime.png")

    print(f"Wrote MC convergence rows to {args.results_dir / 'mc_convergence.csv'}")
    print(f"Wrote CN refinement rows to {args.results_dir / 'cn_refinement.csv'}")
    print(f"Wrote efficiency summary to {args.results_dir / 'efficiency_summary.csv'}")
    print(f"Wrote efficiency figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
