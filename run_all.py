"""Run all thesis experiments and generate all outputs in one command.

Usage examples:
  python run_all.py                                       # BS + MC + CN only (~2 min)
  python run_all.py --pinn-mode selected                  # add PINN at 3 sigmas (~5 min)
  python run_all.py --pinn-mode all --pinn-save-checkpoints  # full 4-way comparison (~35 min)
  python run_all.py --pinn-mode all --pinn-load-checkpoints  # reuse saved checkpoints (~2 min)
  python run_all.py --pinn-mode all --run-pinn-reliability   # benchmark + deep PINN study
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _run(cmd, description):
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pinn-mode",
        choices=["none", "selected", "all"],
        default="none",
        help="'none': skip PINN (fast); 'selected': 3 sigmas; 'all': full grid (~35 min)",
    )
    parser.add_argument(
        "--pinn-save-checkpoints",
        action="store_true",
        help="Save trained PINN models so they can be reloaded with --pinn-load-checkpoints",
    )
    parser.add_argument(
        "--pinn-load-checkpoints",
        action="store_true",
        help="Load existing PINN checkpoints instead of retraining (requires prior --pinn-save-checkpoints run)",
    )
    parser.add_argument(
        "--run-pinn-reliability",
        action="store_true",
        help="Run detailed PINN reliability study (seed variability, training effort, pointwise errors)",
    )
    parser.add_argument(
        "--skip-efficiency",
        action="store_true",
        help="Skip MC convergence and CN refinement experiments",
    )
    args = parser.parse_args()

    python = sys.executable
    step = 1
    total = 2 + (0 if args.skip_efficiency else 1) + (1 if args.run_pinn_reliability else 0)

    # Step 1: Volatility benchmark
    pinn_label = {"none": "", "selected": ", PINN@3σ", "all": ", PINN@all σ"}[args.pinn_mode]
    benchmark_cmd = [
        python, str(PROJECT_ROOT / "run_benchmark.py"),
        "--pinn-mode", args.pinn_mode,
        "--make-figures",
    ]
    if args.pinn_save_checkpoints:
        benchmark_cmd.append("--pinn-save-checkpoints")
    if args.pinn_load_checkpoints:
        benchmark_cmd.append("--pinn-load-checkpoints")
    _run(benchmark_cmd, f"Step {step}/{total}: Volatility benchmark (BS, MC, CN{pinn_label})")
    step += 1

    # Step 2: Efficiency experiments
    if not args.skip_efficiency:
        _run(
            [python, str(PROJECT_ROOT / "run_efficiency_experiments.py")],
            f"Step {step}/{total}: Efficiency experiments (MC convergence, CN refinement)",
        )
        step += 1
    else:
        print("\nSkipping efficiency experiments (--skip-efficiency)")

    # Step 3: PINN reliability
    if args.run_pinn_reliability:
        pinn_rel_cmd = [python, str(PROJECT_ROOT / "run_pinn_reliability.py")]
        if args.pinn_save_checkpoints:
            pinn_rel_cmd.append("--save-checkpoints")
        if args.pinn_load_checkpoints:
            pinn_rel_cmd.append("--load-checkpoints")
        _run(pinn_rel_cmd, f"Step {step}/{total}: PINN reliability analysis (seed variability, effort study)")
        step += 1
    else:
        print("\nSkipping PINN reliability (pass --run-pinn-reliability to include)")

    print(f"\n{'=' * 60}")
    print("  All experiments complete.")
    print(f"  Results  →  {PROJECT_ROOT / 'results'}")
    print(f"  Figures  →  {PROJECT_ROOT / 'figures'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
