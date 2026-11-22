"""Diagnostic experiments behind Sections 6.1 and 6.3 of the report.

Part A (learning rate): tabular Q-learning against the always-bet opponent with
learning rates 0.05 and 0.01. Does the larger rate sometimes fail to reach the
best-response value?

Part B (Huber loss): a single learner against a FIXED near-equilibrium opponent
(OpenSpiel's CFR average policy after 15 iterations), so there is no self-play.
Compares tabular Q-learning, DQN with squared error, and DQN with Huber loss on
  * shortfall: best-response value minus the learner's exact expected value
  * the learned value of betting first with the Queen, against the exact value

Run from the repository root:
    python -m experiments.diagnostics
    python -m experiments.diagnostics --seeds 2 --hands 5000      (quick check)
"""

import argparse
import multiprocessing

import numpy as np
import pyspiel
import torch
from open_spiel.python.algorithms import best_response, cfr, expected_game_score
from scipy import stats

from experiments.methods import build_agent, hyperparameters_for
from kuhn_rl.agents import TabularQLearningAgent, always_bet_agent
from kuhn_rl.evaluation import (
    _info_state_details,
    best_response_value_against,
    build_openspiel_policy,
    expected_value_against,
)
from kuhn_rl.game_loop import play_one_hand

GAME_NAME = "kuhn_poker"
QUEEN_FIRST_TO_ACT = "1"  # information state: holding the Queen, no bets yet
PASS_ACTION, BET_ACTION = 0, 1
CFR_ITERATIONS_FOR_OPPONENT = 15


# ---------------------------------------------------------------------- #
# Part A: learning rate against the always-bet opponent
# ---------------------------------------------------------------------- #
def run_learning_rate_seed(job):
    learning_rate, seed, number_of_hands = job
    random_generator = np.random.default_rng(seed)
    game = pyspiel.load_game(GAME_NAME)
    learner = TabularQLearningAgent(learning_rate=learning_rate, random_generator=random_generator)
    opponent = always_bet_agent(random_generator)
    for _ in range(number_of_hands):
        learner_seat = int(random_generator.integers(2))
        agents_by_seat = [learner, opponent] if learner_seat == 0 else [opponent, learner]
        play_one_hand(game, agents_by_seat, random_generator)
    opponent_policy = build_openspiel_policy(game, opponent)
    return learning_rate, expected_value_against(game, build_openspiel_policy(game, learner), opponent_policy)


# ---------------------------------------------------------------------- #
# Part B: one learner against a fixed near-equilibrium opponent
# ---------------------------------------------------------------------- #
def near_equilibrium_opponent_table(game):
    solver = cfr.CFRSolver(game)
    for _ in range(CFR_ITERATIONS_FOR_OPPONENT):
        solver.evaluate_and_update_policy()
    opponent_policy = solver.average_policy()
    probabilities_by_info_state = {
        info_state: opponent_policy.action_probability_array[row].copy()
        for info_state, row in opponent_policy.state_lookup.items()
    }
    return opponent_policy, probabilities_by_info_state


class FixedTableOpponent:
    def __init__(self, probabilities_by_info_state, random_generator):
        self.probabilities_by_info_state = probabilities_by_info_state
        self.random_generator = random_generator

    def start_hand(self):
        pass

    def end_hand(self, final_reward):
        pass

    def act(self, info_state, legal_actions, info_state_tensor=None):
        return int(self.random_generator.choice(2, p=self.probabilities_by_info_state[info_state]))


def exact_value_of_betting_queen_first(opponent_probabilities):
    """Opponent holds the Jack or the King with equal probability. Facing our bet,
    a fold (pass) wins us the 1-chip ante; a call goes to showdown for 2 chips."""
    jack_folds = opponent_probabilities["0b"][PASS_ACTION]
    king_folds = opponent_probabilities["2b"][PASS_ACTION]
    value_against_jack = jack_folds * 1 + (1 - jack_folds) * 2
    value_against_king = king_folds * 1 + (1 - king_folds) * (-2)
    return 0.5 * (value_against_jack + value_against_king)


def learned_value_of_betting_queen_first(learner, game):
    for info_state, _, owning_player, info_state_tensor, _ in _info_state_details(str(game)):
        if info_state == QUEEN_FIRST_TO_ACT and owning_player == 0:
            if hasattr(learner, "q_values_for"):
                return float(learner.q_values_for(info_state, info_state_tensor)[BET_ACTION])
            return float(learner.q_values_by_info_state[info_state][BET_ACTION])
    raise ValueError("Queen information state not found")


def run_fixed_opponent_seed(job):
    learner_label, method_name, overrides, seed, number_of_hands = job
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    random_generator = np.random.default_rng(seed)
    game = pyspiel.load_game(GAME_NAME)
    opponent_policy, opponent_probabilities = near_equilibrium_opponent_table(game)
    learner = build_agent(method_name, game, hyperparameters_for(method_name, overrides), random_generator)
    opponent = FixedTableOpponent(opponent_probabilities, random_generator)

    # The learner always sits in seat 0 (first to act), the seat whose Queen we inspect.
    for _ in range(number_of_hands):
        play_one_hand(game, [learner, opponent], random_generator)

    combined_policy = build_openspiel_policy(game, learner)
    for info_state, row in combined_policy.state_lookup.items():
        if combined_policy.states[row].current_player() == 1:
            combined_policy.action_probability_array[row] = opponent_probabilities[info_state]
    learner_value = expected_game_score.policy_value(game.new_initial_state(), [combined_policy, combined_policy])[0]
    best_value = best_response.BestResponsePolicy(game, 0, opponent_policy).value(game.new_initial_state())
    return {
        "label": learner_label,
        "shortfall": best_value - learner_value,
        "learned_queen_bet_value": learned_value_of_betting_queen_first(learner, game),
        "exact_queen_bet_value": exact_value_of_betting_queen_first(opponent_probabilities),
    }


# ---------------------------------------------------------------------- #
def mean_with_interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return f"{values.mean():+.3f}"
    half_width = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return f"{values.mean():+.3f} [{values.mean() - half_width:+.3f}, {values.mean() + half_width:+.3f}]"


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--seeds", type=int, default=10)
    argument_parser.add_argument("--hands", type=int, default=100_000)
    argument_parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    arguments = argument_parser.parse_args()
    seeds = range(arguments.seeds)

    learning_rate_jobs = [(rate, seed, arguments.hands) for rate in (0.05, 0.01) for seed in seeds]
    fixed_opponent_jobs = [
        (label, method, overrides, seed, arguments.hands)
        for label, method, overrides in [
            ("Tabular Q-learning", "naive_tabular_q", {}),
            ("DQN, squared error", "naive_dqn", {"loss_function": "mse"}),
            ("DQN, Huber loss", "naive_dqn", {"loss_function": "huber"}),
        ]
        for seed in seeds
    ]
    with multiprocessing.get_context("spawn").Pool(arguments.workers) as worker_pool:
        learning_rate_results = worker_pool.map(run_learning_rate_seed, learning_rate_jobs)
        fixed_opponent_results = worker_pool.map(run_fixed_opponent_seed, fixed_opponent_jobs)

    game = pyspiel.load_game(GAME_NAME)
    best_value_against_always_bet = best_response_value_against(
        game, build_openspiel_policy(game, always_bet_agent()))
    print(f"\nPart A: tabular Q-learning vs. always-bet opponent "
          f"({arguments.seeds} seeds, {arguments.hands:,} hands, best response {best_value_against_always_bet:+.3f})")
    for rate in (0.05, 0.01):
        values = [value for result_rate, value in learning_rate_results if result_rate == rate]
        seeds_short = sum(value < best_value_against_always_bet - 0.01 for value in values)
        print(f"  learning rate {rate}: mean {np.mean(values):+.3f}, worst seed {min(values):+.3f}, "
              f"seeds more than 0.01 short of best response: {seeds_short} of {len(values)}")

    exact_value = fixed_opponent_results[0]["exact_queen_bet_value"]
    print(f"\nPart B: one learner vs. fixed near-equilibrium opponent (CFR after "
          f"{CFR_ITERATIONS_FOR_OPPONENT} iterations; {arguments.seeds} seeds, {arguments.hands:,} hands)")
    print(f"  Exact value of betting the Queen first: {exact_value:+.3f}")
    for label in ("Tabular Q-learning", "DQN, squared error", "DQN, Huber loss"):
        results = [result for result in fixed_opponent_results if result["label"] == label]
        print(f"  {label:20}: shortfall {mean_with_interval([r['shortfall'] for r in results])}   "
              f"learned value of betting the Queen {mean_with_interval([r['learned_queen_bet_value'] for r in results])}")
    print("\nShortfall = best-response value minus the learner's exact expected value (chips/hand); "
          "mean [95% CI] over seeds.")


if __name__ == "__main__":
    main()
