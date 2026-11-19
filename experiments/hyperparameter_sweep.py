"""Try several values of one hyperparameter and log the results.

The report must say how every parameter value was chosen. Each run of this
script appends one row per value to results/sweeps/sweep_log.csv, so the log
is the record of what was tried. Copy the conclusions into HYPERPARAMETERS.md
(results/ is not committed to git).

Examples (from the repository root):
    python -m experiments.hyperparameter_sweep --method tabular_fsp --parameter anticipatory_parameter --values 0.05 0.1 0.2
    python -m experiments.hyperparameter_sweep --method nfsp --parameter learning_rate --values 0.0005 0.001 0.005 --hands 300000
"""

import argparse
import ast
import csv
import datetime
import multiprocessing
from pathlib import Path

from experiments.make_figures import final_values_per_seed
from experiments.methods import DEFAULT_TRAINING_HANDS, METHOD_NAMES, PRIMARY_POLICY, hyperparameters_for
from experiments.self_play import parse_overrides, run_method


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--method", required=True, choices=METHOD_NAMES)
    argument_parser.add_argument("--parameter", required=True)
    argument_parser.add_argument("--values", nargs="+", required=True)
    argument_parser.add_argument("--game", default="kuhn_poker")
    argument_parser.add_argument("--hands", type=int, default=None)
    argument_parser.add_argument("--seeds", type=int, default=3)
    argument_parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    argument_parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                                 help="other settings to hold fixed during the sweep")
    argument_parser.add_argument("--log-file", default="results/sweeps/sweep_log.csv")
    arguments = argument_parser.parse_args()

    number_of_training_hands = arguments.hands or DEFAULT_TRAINING_HANDS[arguments.method] // 2
    evaluate_every_hands = max(1, number_of_training_hands // 50)
    # Sweep seeds start at 1000 so tuning never reuses the seeds of the final runs.
    seeds = list(range(1000, 1000 + arguments.seeds))
    log_path = Path(arguments.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()

    with log_path.open("a", newline="") as log_file:
        log_writer = csv.writer(log_file)
        if write_header:
            log_writer.writerow(["timestamp", "game", "method", "parameter", "value", "other_settings", "hands",
                                 "seeds", "final_current_exploitability", "final_average_exploitability",
                                 "primary_policy"])
        for raw_value in arguments.values:
            try:
                value = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                value = raw_value
            overrides = {**parse_overrides(arguments.set), arguments.parameter: value}
            hyperparameters = hyperparameters_for(arguments.method, overrides)
            results = run_method(arguments.method, arguments.game, number_of_training_hands, evaluate_every_hands,
                                 seeds, hyperparameters, arguments.workers)
            final_current = final_values_per_seed(results["current_exploitability_by_seed"]).mean()
            final_average = final_values_per_seed(results["average_exploitability_by_seed"]).mean()
            print(f"{arguments.parameter} = {value!r:>12}:  current {final_current:.4f}   average {final_average:.4f}"
                  f"   (judged on {PRIMARY_POLICY[arguments.method]})", flush=True)
            log_writer.writerow([datetime.datetime.now().isoformat(timespec="seconds"), arguments.game,
                                 arguments.method, arguments.parameter, repr(value), " ".join(arguments.set),
                                 number_of_training_hands, len(seeds), f"{final_current:.5f}", f"{final_average:.5f}",
                                 PRIMARY_POLICY[arguments.method]])
            log_file.flush()
    print(f"Appended results to {log_path}")


if __name__ == "__main__":
    main()
