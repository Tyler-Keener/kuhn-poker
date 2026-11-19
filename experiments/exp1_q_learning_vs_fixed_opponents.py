"""Plan step 1: tabular Q-learning against fixed opponents.

Against a fixed opponent, Kuhn poker is an ordinary single-agent MDP, so the
learner should discover how to exploit each opponent. We compare its exact
expected winnings with the best possible exploiting strategy (best response).

Run from the repository root:
    python -m experiments.exp1_q_learning_vs_fixed_opponents --seeds 10
"""

import argparse
from pathlib import Path

import numpy as np
import pyspiel

from kuhn_rl.agents import (
    TabularQLearningAgent,
    always_bet_agent,
    always_pass_agent,
    uniform_random_agent,
)
from kuhn_rl.evaluation import best_response_value_against, build_openspiel_policy, expected_value_against
from kuhn_rl.game_loop import play_one_hand
from kuhn_rl.plotting import plot_curves_with_confidence

OPPONENT_FACTORIES = {
    "always_bet": always_bet_agent,
    "always_pass": always_pass_agent,
    "uniform_random": uniform_random_agent,
}


def train_against_fixed_opponent(game, opponent_name, number_of_training_hands, evaluate_every_hands, seed):
    random_generator = np.random.default_rng(seed)
    learner = TabularQLearningAgent(random_generator=random_generator)
    opponent = OPPONENT_FACTORIES[opponent_name](random_generator)
    opponent_policy = build_openspiel_policy(game, opponent)

    expected_value_at_checkpoints = []
    for hand_number in range(1, number_of_training_hands + 1):
        # Randomize seats so the learner learns to play both positions.
        learner_seat = int(random_generator.integers(2))
        agents_by_seat = [learner, opponent] if learner_seat == 0 else [opponent, learner]
        play_one_hand(game, agents_by_seat, random_generator)

        if hand_number % evaluate_every_hands == 0:
            learner_policy = build_openspiel_policy(game, learner)
            expected_value_at_checkpoints.append(
                expected_value_against(game, learner_policy, opponent_policy)
            )
    return np.array(expected_value_at_checkpoints), learner


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--hands", type=int, default=100_000)
    argument_parser.add_argument("--evaluate-every", type=int, default=1_000)
    argument_parser.add_argument("--seeds", type=int, default=10)
    argument_parser.add_argument("--output-directory", default="results/exp1")
    arguments = argument_parser.parse_args()

    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    game = pyspiel.load_game("kuhn_poker")
    training_hands_at_checkpoints = np.arange(
        arguments.evaluate_every, arguments.hands + 1, arguments.evaluate_every
    )

    for opponent_name, opponent_factory in OPPONENT_FACTORIES.items():
        opponent_policy = build_openspiel_policy(game, opponent_factory())
        optimal_exploit_value = best_response_value_against(game, opponent_policy)

        values_by_seed = []
        for seed in range(arguments.seeds):
            expected_values, trained_learner = train_against_fixed_opponent(
                game, opponent_name, arguments.hands, arguments.evaluate_every, seed
            )
            values_by_seed.append(expected_values)
        values_by_seed = np.array(values_by_seed)

        final_values = values_by_seed[:, -1]
        print(f"\nOpponent: {opponent_name}")
        print(f"  Best possible winnings (best response): {optimal_exploit_value:+.4f} chips/hand")
        print(f"  Q-learning final winnings:             {final_values.mean():+.4f} "
              f"(std {final_values.std():.4f} over {arguments.seeds} seeds)")
        print("  Greedy policy learned by the last seed (P(bet) per info state):")
        for info_state in sorted(trained_learner.q_values_by_info_state):
            bet_probability = trained_learner.action_probabilities_for(info_state)[1]
            print(f"    {info_state:>4}: {bet_probability:.2f}")

        np.savez(output_directory / f"{opponent_name}.npz",
                 training_hands=training_hands_at_checkpoints,
                 expected_value_by_seed=values_by_seed,
                 best_response_value=optimal_exploit_value)
        plot_curves_with_confidence(
            {"Tabular Q-learning": (training_hands_at_checkpoints, values_by_seed)},
            y_axis_label="Expected chips won per hand",
            title=f"Q-learning vs. {opponent_name} opponent",
            output_path=output_directory / f"{opponent_name}.png",
            horizontal_reference_lines={"Best response (optimal exploit)": optimal_exploit_value},
        )

    print(f"\nSaved plots and raw data to {output_directory}/")


if __name__ == "__main__":
    main()
