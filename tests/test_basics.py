"""Sanity checks. Run with:  pytest   (takes about a minute)"""

import numpy as np
import pyspiel
import pytest
import torch
from open_spiel.python import policy as openspiel_policy

from experiments.methods import METHOD_NAMES, build_agent, hyperparameters_for
from kuhn_rl.agents import (
    DQNAgent,
    TabularQLearningAgent,
    always_bet_agent,
    uniform_random_agent,
)
from kuhn_rl.agents.neural_components import ReservoirBuffer
from kuhn_rl.agents.policy_helpers import ActionFrequencyTable
from kuhn_rl.evaluation import (
    best_response_value_against,
    build_openspiel_policy,
    compute_exploitability,
    count_policy_flips,
    expected_value_against,
    lowest_deterministic_exploitability,
)
from kuhn_rl.game_loop import play_one_hand

GAME = pyspiel.load_game("kuhn_poker")
STATE_SIZE = GAME.information_state_tensor_size()


def train_against_fixed_opponent(learner, opponent, number_of_hands, random_generator):
    for _ in range(number_of_hands):
        learner_seat = int(random_generator.integers(2))
        seats = [learner, opponent] if learner_seat == 0 else [opponent, learner]
        play_one_hand(GAME, seats, random_generator)


# ---------------------------------------------------------------------- #
# Game and evaluation
# ---------------------------------------------------------------------- #
def test_kuhn_has_twelve_information_states():
    assert len(openspiel_policy.TabularPolicy(GAME).state_lookup) == 12


def test_hand_rewards_are_zero_sum_and_bounded():
    random_generator = np.random.default_rng(0)
    agents = [uniform_random_agent(random_generator), uniform_random_agent(random_generator)]
    for _ in range(500):
        rewards = play_one_hand(GAME, agents, random_generator)
        assert rewards.sum() == 0
        assert np.all(np.abs(rewards) <= 2)


def test_uniform_random_matches_openspiel():
    our_uniform_policy = build_openspiel_policy(GAME, uniform_random_agent())
    expected = compute_exploitability(GAME, openspiel_policy.UniformRandomPolicy(GAME))
    assert np.isclose(compute_exploitability(GAME, our_uniform_policy), expected)


def test_deterministic_policies_cannot_beat_one_sixth():
    assert np.isclose(lowest_deterministic_exploitability(str(GAME)), 1 / 6)


def test_policy_flip_counter():
    assert count_policy_flips(None, np.array([0, 1])) == 0
    assert count_policy_flips(np.array([0, 1, 1]), np.array([0, 0, 1])) == 1


# ---------------------------------------------------------------------- #
# Building blocks
# ---------------------------------------------------------------------- #
def test_action_frequency_table_gives_empirical_average():
    table = ActionFrequencyTable(2)
    assert np.allclose(table.probabilities("0", [0, 1]), [0.5, 0.5])  # unseen -> uniform
    for action in [0, 1, 1, 1]:
        table.record("0", action)
    assert np.allclose(table.probabilities("0", [0, 1]), [0.25, 0.75])


def test_reservoir_buffer_keeps_a_uniform_sample():
    buffer = ReservoirBuffer(capacity=1_000, state_size=1, number_of_actions=2,
                             random_generator=np.random.default_rng(0))
    for item_number in range(10_000):
        buffer.add(np.array([item_number], dtype=np.float32), 0, np.ones(2, dtype=np.float32))
    # Uniform over 0..9999 has mean about 5000; a circular buffer would give about 9500.
    assert len(buffer) == 1_000
    assert 4_500 < buffer.states[:, 0].mean() < 5_500


def test_untrained_tabular_agent_is_uniform():
    untrained_policy = build_openspiel_policy(GAME, TabularQLearningAgent())
    assert np.allclose(untrained_policy.action_probability_array, 0.5)


@pytest.mark.parametrize("method_name", METHOD_NAMES)
def test_every_method_runs_and_gives_valid_policies(method_name):
    torch.manual_seed(0)
    random_generator = np.random.default_rng(0)
    hyperparameters = hyperparameters_for(method_name, {})
    agents = [build_agent(method_name, GAME, hyperparameters, random_generator) for _ in range(2)]
    for _ in range(3_000):
        play_one_hand(GAME, agents, random_generator)
    for use_average_policy in [False, True]:
        joint_policy = build_openspiel_policy(GAME, *agents, use_average_policy=use_average_policy)
        assert np.allclose(joint_policy.action_probability_array.sum(axis=1), 1.0)
        assert 0.0 <= compute_exploitability(GAME, joint_policy) <= 1.0


# ---------------------------------------------------------------------- #
# Learning (single-agent case, Plan step 1)
# ---------------------------------------------------------------------- #
def test_q_learning_exploits_always_bet_opponent():
    random_generator = np.random.default_rng(1)
    learner = TabularQLearningAgent(random_generator=random_generator)
    opponent = always_bet_agent(random_generator)
    train_against_fixed_opponent(learner, opponent, 30_000, random_generator)

    opponent_policy = build_openspiel_policy(GAME, opponent)
    learner_value = expected_value_against(GAME, build_openspiel_policy(GAME, learner), opponent_policy)
    assert learner_value > best_response_value_against(GAME, opponent_policy) - 0.05


def test_dqn_exploits_always_bet_opponent():
    torch.manual_seed(0)
    random_generator = np.random.default_rng(0)
    learner = DQNAgent(STATE_SIZE, random_generator=random_generator)
    opponent = always_bet_agent(random_generator)
    train_against_fixed_opponent(learner, opponent, 40_000, random_generator)

    opponent_policy = build_openspiel_policy(GAME, opponent)
    learner_value = expected_value_against(GAME, build_openspiel_policy(GAME, learner), opponent_policy)
    assert learner_value > best_response_value_against(GAME, opponent_policy) - 0.05
