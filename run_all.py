"""Run every experiment for the report, then make the figures.

    python run_all.py            full runs (10 seeds; the neural methods take a while)
    python run_all.py --quick    tiny runs to check everything works (~2 minutes)
    python run_all.py --only tabular_fsp naive_tabular_q    a subset of methods

Outputs go to results/ (figures and summary table in results/figures/).
"""

import argparse
import subprocess
import sys
import time

from experiments.methods import METHOD_NAMES


def run_step(description, command):
    print(f"\n=== {description} ===", flush=True)
    start_time = time.time()
    subprocess.run([sys.executable, "-m", *command], check=True)
    print(f"({time.time() - start_time:.0f}s)", flush=True)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--quick", action="store_true", help="tiny runs to check the pipeline")
    argument_parser.add_argument("--only", nargs="*", choices=METHOD_NAMES, default=METHOD_NAMES)
    argument_parser.add_argument("--seeds", type=int, default=10)
    argument_parser.add_argument("--game", default="kuhn_poker")
    arguments = argument_parser.parse_args()

    quick_settings = ["--seeds", "2", "--hands", "5000"] if arguments.quick else ["--seeds", str(arguments.seeds)]
    results_directory = "results/quick" if arguments.quick else "results/self_play"
    figures_directory = "results/quick/figures" if arguments.quick else "results/figures"

    if arguments.game == "kuhn_poker":
        run_step("Plan step 1: tabular Q-learning vs. fixed opponents",
                 ["experiments.exp1_q_learning_vs_fixed_opponents",
                  *(["--seeds", "2", "--hands", "5000"] if arguments.quick else ["--seeds", str(arguments.seeds)]),
                  "--output-directory", "results/quick/exp1" if arguments.quick else "results/exp1"])

    run_step("Reference: CFR and CFR+",
             ["experiments.cfr_baseline", "--game", arguments.game,
              "--iterations", "100" if arguments.quick else "1000", "--output-directory", results_directory])

    for method_name in arguments.only:
        run_step(f"Self-play: {method_name}",
                 ["experiments.self_play", "--method", method_name, "--game", arguments.game,
                  *quick_settings, "--output-directory", results_directory])

    run_step("Figures and summary table",
             ["experiments.make_figures", "--game", arguments.game,
              "--results-directory", results_directory, "--figures-directory", figures_directory])


if __name__ == "__main__":
    main()
