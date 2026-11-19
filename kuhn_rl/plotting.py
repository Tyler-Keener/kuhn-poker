"""Plot mean curves across random seeds with shaded 95% confidence intervals."""

import matplotlib

matplotlib.use("Agg")  # save to file; works without a display
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def mean_and_confidence_interval(values_by_seed, confidence_level=0.95):
    """values_by_seed: array of shape (number_of_seeds, number_of_checkpoints).
    Returns mean, lower bound, upper bound, using a t-distribution because we
    only have about 10 seeds."""
    values_by_seed = np.asarray(values_by_seed, dtype=float)
    number_of_seeds = values_by_seed.shape[0]
    mean_values = values_by_seed.mean(axis=0)
    if number_of_seeds < 2:
        return mean_values, mean_values, mean_values
    standard_error = values_by_seed.std(axis=0, ddof=1) / np.sqrt(number_of_seeds)
    t_critical_value = stats.t.ppf((1 + confidence_level) / 2, df=number_of_seeds - 1)
    half_width = t_critical_value * standard_error
    return mean_values, mean_values - half_width, mean_values + half_width


def plot_curves_with_confidence(
    curves_by_label,
    y_axis_label,
    title,
    output_path,
    x_axis_label="Training hands",
    horizontal_reference_lines=None,
    log_y_axis=False,
    log_x_axis=False,
):
    """curves_by_label: {"label": (x_values, values_by_seed)} with values_by_seed
    shaped (seeds, len(x_values)). Each curve may have its own x values.
    horizontal_reference_lines: {"label": y_value} for baselines like the 1/6 floor."""
    figure, axes = plt.subplots(figsize=(8, 5))
    for label, (x_values, values_by_seed) in curves_by_label.items():
        mean_values, lower_bound, upper_bound = mean_and_confidence_interval(values_by_seed)
        (line,) = axes.plot(x_values, mean_values, label=label, linewidth=1.6)
        axes.fill_between(x_values, lower_bound, upper_bound, color=line.get_color(), alpha=0.2)

    reference_line_styles = ["--", ":", "-."]
    for line_index, (label, y_value) in enumerate((horizontal_reference_lines or {}).items()):
        axes.axhline(y_value, linestyle=reference_line_styles[line_index % len(reference_line_styles)],
                     linewidth=1.2, label=label, color="black")

    if log_y_axis:
        axes.set_yscale("log")
        # With few seeds the lower 95% bound can go below zero, which a log axis
        # cannot show; set the limits from the curves so the band is cut off instead.
        positive_values = [value for line in axes.get_lines() for value in np.asarray(line.get_ydata(), dtype=float)
                           if value > 0]
        if positive_values:
            axes.set_ylim(min(positive_values) * 0.6, max(positive_values) * 1.6)
    if log_x_axis:
        axes.set_xscale("log")
    axes.set_xlabel(x_axis_label)
    axes.set_ylabel(y_axis_label)
    axes.set_title(title)
    axes.legend(fontsize=9)
    axes.grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
