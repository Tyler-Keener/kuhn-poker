"""Turn saved results into the report's figures and a summary table.

Reads results/self_play/<game>_<method>.npz (from self_play.py) and
<game>_cfr.npz (from cfr_baseline.py); skips anything not run yet.

Writes to results/figures/:
  tabular_averaging.png   naive tabular Q vs. its after-the-fact average vs. tabular FSP
  neural_averaging.png    the same comparison for DQN and NFSP
  all_methods.png         each method's main curve together
  policy_flips.png        how often the greedy policy keeps changing (cycling)
  cfr_reference.png       CFR and CFR+ against their own iterations
  summary.md              final values with 95% confidence intervals and t-tests

"Final" = each seed's mean over the last 10% of checkpoints, then mean and
95% t-interval across seeds. Averaging the last 10% smooths out the jumps of
greedy policies between checkpoints.

Run from the repository root:
    python -m experiments.make_figures
"""

import argparse
from pathlib import Path

import numpy as np
import pyspiel
from scipy import stats

from experiments.methods import DISPLAY_NAMES, METHOD_NAMES, PRIMARY_POLICY
from kuhn_rl.evaluation import lowest_deterministic_exploitability, uniform_random_exploitability
from kuhn_rl.plotting import mean_and_confidence_interval, plot_curves_with_confidence


def load_results(results_directory, game_name):
    results_by_method = {}
    for method_name in METHOD_NAMES:
        path = results_directory / f"{game_name}_{method_name}.npz"
        if path.exists():
            results_by_method[method_name] = dict(np.load(path))
    return results_by_method


def final_values_per_seed(values_by_seed, fraction_of_checkpoints=0.1):
    number_of_last_checkpoints = max(1, int(round(values_by_seed.shape[1] * fraction_of_checkpoints)))
    return values_by_seed[:, -number_of_last_checkpoints:].mean(axis=1)


def describe(values):
    mean_value, lower_bound, upper_bound = mean_and_confidence_interval(np.asarray(values)[:, None])
    return f"{mean_value[0]:.4f} [{lower_bound[0]:.4f}, {upper_bound[0]:.4f}]"


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--game", default="kuhn_poker")
    argument_parser.add_argument("--results-directory", default="results/self_play")
    argument_parser.add_argument("--figures-directory", default="results/figures")
    arguments = argument_parser.parse_args()

    results_directory = Path(arguments.results_directory)
    figures_directory = Path(arguments.figures_directory)
    figures_directory.mkdir(parents=True, exist_ok=True)
    game = pyspiel.load_game(arguments.game)
    results_by_method = load_results(results_directory, arguments.game)

    reference_lines = {"Uniform random policy": uniform_random_exploitability(game)}
    deterministic_floor = lowest_deterministic_exploitability(str(game))
    if deterministic_floor is not None:
        reference_lines["Best any deterministic policy can do (1/6)"] = deterministic_floor

    def curve(method_name, which_policy):
        results = results_by_method[method_name]
        return results["training_hands"], results[f"{which_policy}_exploitability_by_seed"]

    # Figures 1 and 2: does averaging help, with and without neural networks?
    for naive_method, averaging_method, figure_name, title in [
        ("naive_tabular_q", "tabular_fsp", "tabular_averaging.png", "Tabular: naive self-play vs. fictitious self-play"),
        ("naive_dqn", "nfsp", "neural_averaging.png", "Neural: naive DQN self-play vs. NFSP"),
    ]:
        curves = {}
        if naive_method in results_by_method:
            curves[f"{DISPLAY_NAMES[naive_method]} (current policy)"] = curve(naive_method, "current")
            curves[f"{DISPLAY_NAMES[naive_method]} (after-the-fact average)"] = curve(naive_method, "average")
        if averaging_method in results_by_method:
            curves[f"{DISPLAY_NAMES[averaging_method]} (average policy)"] = curve(averaging_method, "average")
        if curves:
            plot_curves_with_confidence(curves, "Exploitability (chips/hand, log scale)", title,
                                        figures_directory / figure_name,
                                        horizontal_reference_lines=reference_lines, log_y_axis=True)

    # Figure 3: every method's main curve.
    if results_by_method:
        plot_curves_with_confidence(
            {DISPLAY_NAMES[name]: curve(name, PRIMARY_POLICY[name]) for name in results_by_method},
            "Exploitability (chips/hand, log scale)", f"All methods ({arguments.game})",
            figures_directory / "all_methods.png", horizontal_reference_lines=reference_lines, log_y_axis=True)

        # Figure 4: cycling, measured as greedy-policy flips between checkpoints.
        plot_curves_with_confidence(
            {DISPLAY_NAMES[name]: (results["training_hands"], results["policy_flips_by_seed"])
             for name, results in results_by_method.items()},
            "Info states whose greedy action changed\nsince the previous checkpoint",
            "Does the current (greedy) policy settle down?", figures_directory / "policy_flips.png")

    # Figure 5: CFR reference.
    cfr_path = results_directory / f"{arguments.game}_cfr.npz"
    if cfr_path.exists():
        cfr_results = np.load(cfr_path)
        plot_curves_with_confidence(
            {"CFR (OpenSpiel)": (cfr_results["iterations"], cfr_results["cfr_exploitability"][None, :]),
             "CFR+ (OpenSpiel)": (cfr_results["iterations"], cfr_results["cfr_plus_exploitability"][None, :])},
            "Exploitability (chips/hand)", "Reference: CFR solvers (full tree traversal each iteration)",
            figures_directory / "cfr_reference.png", x_axis_label="Solver iterations",
            log_y_axis=True, log_x_axis=True)

    # Summary table and significance tests.
    lines = [f"# Results summary ({arguments.game})", "",
             "Final = each seed's mean over the last 10% of checkpoints; "
             "shown as mean [95% confidence interval] across seeds.", "",
             "| Method | Seeds | Hands | Current policy | Average policy | Policy flips (last half) |",
             "|---|---|---|---|---|---|"]
    final_by_method_and_policy = {}
    for name, results in results_by_method.items():
        for which_policy in ["current", "average"]:
            final_by_method_and_policy[(name, which_policy)] = final_values_per_seed(
                results[f"{which_policy}_exploitability_by_seed"])
        flips = results["policy_flips_by_seed"]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {len(results['seeds'])} | {results['training_hands'][-1]:,} "
            f"| {describe(final_by_method_and_policy[(name, 'current')])} "
            f"| {describe(final_by_method_and_policy[(name, 'average')])} "
            f"| {flips[:, flips.shape[1] // 2:].mean():.2f} |")
    for label, y_value in reference_lines.items():
        lines.append(f"\n{label}: {y_value:.4f}")
    if cfr_path.exists():
        lines.append(f"\nCFR after {len(cfr_results['iterations'])} iterations: "
                     f"{cfr_results['cfr_exploitability'][-1]:.5f}; CFR+: {cfr_results['cfr_plus_exploitability'][-1]:.5f}")

    comparisons = [
        (("naive_tabular_q", "current"), ("tabular_fsp", "average")),
        (("naive_tabular_q", "average"), ("tabular_fsp", "average")),
        (("naive_dqn", "current"), ("nfsp", "average")),
        (("naive_dqn", "average"), ("nfsp", "average")),
        (("tabular_fsp", "average"), ("nfsp", "average")),
    ]
    lines += ["", "## Welch's t-tests on final exploitability", "",
              "| Comparison | Difference in means | t | p |", "|---|---|---|---|"]
    for first, second in comparisons:
        if first in final_by_method_and_policy and second in final_by_method_and_policy:
            first_values, second_values = final_by_method_and_policy[first], final_by_method_and_policy[second]
            if len(first_values) > 1 and len(second_values) > 1:
                test = stats.ttest_ind(first_values, second_values, equal_var=False)
                lines.append(f"| {DISPLAY_NAMES[first[0]]} ({first[1]}) vs. {DISPLAY_NAMES[second[0]]} ({second[1]}) "
                             f"| {first_values.mean() - second_values.mean():+.4f} | {test.statistic:.2f} | {test.pvalue:.2g} |")

    summary_path = figures_directory / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nFigures and summary saved to {figures_directory}/")


if __name__ == "__main__":
    main()
