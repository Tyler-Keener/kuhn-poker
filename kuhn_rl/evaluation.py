"""Exact evaluation tools, built on OpenSpiel's exploitability and CFR code.

Exploitability = how many chips per hand a perfect opponent who knows your
strategy would win from you, averaged over both seats. Nash equilibrium = 0.
"""

import itertools
from functools import lru_cache

import numpy as np
import pyspiel
from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms import best_response, cfr, exploitability, expected_game_score


# ---------------------------------------------------------------------- #
# Turning our agents into OpenSpiel policies
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _info_state_details(game_name):
    """For every information state: (row, owning player, tensor, legal actions)."""
    game = pyspiel.load_game(game_name)
    tabular_policy = openspiel_policy.TabularPolicy(game)
    details = []
    for info_state, row_index in tabular_policy.state_lookup.items():
        state = tabular_policy.states[row_index]
        owning_player = state.current_player()
        info_state_tensor = np.asarray(state.information_state_tensor(owning_player), dtype=np.float32)
        details.append((info_state, row_index, owning_player, info_state_tensor, state.legal_actions()))
    return details


def build_openspiel_policy(game, agent_for_player_0, agent_for_player_1=None, use_average_policy=False):
    """Convert our agents into an OpenSpiel TabularPolicy.

    If only one agent is given, it supplies the strategy for both seats.
    use_average_policy=False -> each agent's current policy
    use_average_policy=True  -> each agent's average policy over training
    """
    if agent_for_player_1 is None:
        agent_for_player_1 = agent_for_player_0
    agent_by_player = {0: agent_for_player_0, 1: agent_for_player_1}

    tabular_policy = openspiel_policy.TabularPolicy(game)
    for info_state, row_index, owning_player, info_state_tensor, legal_actions in _info_state_details(str(game)):
        agent = agent_by_player[owning_player]
        if use_average_policy:
            probabilities = agent.average_action_probabilities_for(info_state, info_state_tensor, legal_actions)
        else:
            probabilities = agent.action_probabilities_for(info_state, info_state_tensor, legal_actions)
        tabular_policy.action_probability_array[row_index] = probabilities
    return tabular_policy


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #
def compute_exploitability(game, tabular_policy):
    return exploitability.exploitability(game, tabular_policy)


def greedy_action_per_info_state(tabular_policy):
    """Most likely action in each information state (used to count policy flips)."""
    return tabular_policy.action_probability_array.argmax(axis=1)


def count_policy_flips(previous_greedy_actions, current_greedy_actions):
    """Number of information states whose greedy action changed between checkpoints.
    A learner that has settled has 0 flips; a cycling learner keeps flipping."""
    if previous_greedy_actions is None:
        return 0
    return int(np.sum(previous_greedy_actions != current_greedy_actions))


@lru_cache(maxsize=None)
def lowest_deterministic_exploitability(game_name):
    """Lowest exploitability any deterministic (pure) strategy profile can reach.

    A greedy value-based agent always plays one action per information state,
    so its exploitability can never go below this number. Found by checking
    every pure profile, so only feasible for tiny games (Kuhn: 2^12 = 4096).
    In Kuhn poker the answer is exactly 1/6.
    """
    game = pyspiel.load_game(game_name)
    tabular_policy = openspiel_policy.TabularPolicy(game)
    number_of_info_states = len(tabular_policy.state_lookup)
    if number_of_info_states > 16:
        return None
    number_of_actions = game.num_distinct_actions()
    lowest_value = float("inf")
    for chosen_actions in itertools.product(range(number_of_actions), repeat=number_of_info_states):
        tabular_policy.action_probability_array[:] = 0.0
        tabular_policy.action_probability_array[np.arange(number_of_info_states), chosen_actions] = 1.0
        lowest_value = min(lowest_value, compute_exploitability(game, tabular_policy))
    return lowest_value


def uniform_random_exploitability(game):
    return compute_exploitability(game, openspiel_policy.UniformRandomPolicy(game))


def best_response_value_against(game, opponent_policy):
    """Best chips per hand any strategy can win vs. opponent_policy,
    averaged over sitting in seat 0 and seat 1 (we randomize seats in training)."""
    values_by_seat = [
        best_response.BestResponsePolicy(game, seat, opponent_policy).value(game.new_initial_state())
        for seat in range(game.num_players())
    ]
    return float(np.mean(values_by_seat))


def expected_value_against(game, learner_policy, opponent_policy):
    """Exact expected chips per hand for learner_policy vs. opponent_policy,
    averaged over both seats. No sampling noise."""
    initial_state = game.new_initial_state()
    value_as_seat_0 = expected_game_score.policy_value(initial_state, [learner_policy, opponent_policy])[0]
    value_as_seat_1 = expected_game_score.policy_value(initial_state, [opponent_policy, learner_policy])[1]
    return float((value_as_seat_0 + value_as_seat_1) / 2)


# ---------------------------------------------------------------------- #
# CFR reference curve (OpenSpiel's solver, not our code)
# ---------------------------------------------------------------------- #
def cfr_exploitability_curve(game, number_of_iterations, use_cfr_plus=False):
    """Exploitability of CFR's average policy after each iteration.

    CFR walks the whole game tree every iteration, so its x-axis is iterations,
    not training hands; the two are not the same unit of cost.
    """
    solver = cfr.CFRPlusSolver(game) if use_cfr_plus else cfr.CFRSolver(game)
    exploitability_by_iteration = []
    for _ in range(number_of_iterations):
        solver.evaluate_and_update_policy()
        exploitability_by_iteration.append(compute_exploitability(game, solver.average_policy()))
    return np.arange(1, number_of_iterations + 1), np.array(exploitability_by_iteration)
