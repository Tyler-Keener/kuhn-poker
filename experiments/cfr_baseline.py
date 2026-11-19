"""Reference baseline: OpenSpiel's CFR and CFR+ solvers (not our code).

CFR (Zinkevich et al., 2007) and CFR+ (Tammelin, 2014) walk the full game tree
every iteration, so they are plotted against their own iterations rather than
against training hands. They show what a non-learning solver reaches, not a
same-budget comparison with our RL agents.

Run from the repository root:
    python -m experiments.cfr_baseline
"""

import argparse
from pathlib import Path

import numpy as np
import pyspiel

from kuhn_rl.evaluation import cfr_exploitability_curve


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--game", default="kuhn_poker")
    argument_parser.add_argument("--iterations", type=int, default=1_000)
    argument_parser.add_argument("--output-directory", default="results/self_play")
    arguments = argument_parser.parse_args()

    game = pyspiel.load_game(arguments.game)
    iterations, cfr_exploitability = cfr_exploitability_curve(game, arguments.iterations)
    _, cfr_plus_exploitability = cfr_exploitability_curve(game, arguments.iterations, use_cfr_plus=True)

    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez(output_directory / f"{arguments.game}_cfr.npz", iterations=iterations,
             cfr_exploitability=cfr_exploitability, cfr_plus_exploitability=cfr_plus_exploitability)
    for checkpoint in [10, 100, arguments.iterations]:
        if checkpoint <= arguments.iterations:
            print(f"After {checkpoint:>5} iterations: CFR {cfr_exploitability[checkpoint - 1]:.5f}   "
                  f"CFR+ {cfr_plus_exploitability[checkpoint - 1]:.5f}")


if __name__ == "__main__":
    main()
