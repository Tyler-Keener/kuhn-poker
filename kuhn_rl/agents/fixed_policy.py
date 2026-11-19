"""Opponents whose strategy never changes (used in Plan step 1).

Kuhn poker actions in OpenSpiel:  0 = pass (check/fold),  1 = bet (bet/call)
See policy_helpers.py for the agent interface every agent follows.
"""

import numpy as np

PASS_ACTION = 0
BET_ACTION = 1


class FixedPolicyAgent:
    """Plays the same action probabilities in every information state (Kuhn only)."""

    def __init__(self, name, pass_probability, bet_probability, random_generator=None):
        self.name = name
        self.action_probabilities = np.array([pass_probability, bet_probability])
        self.random_generator = random_generator or np.random.default_rng()

    def start_hand(self):
        pass  # nothing to remember between hands

    def act(self, info_state, legal_actions, info_state_tensor=None):
        return int(self.random_generator.choice(legal_actions, p=self.action_probabilities))

    def end_hand(self, final_reward):
        pass  # fixed opponents never learn

    def action_probabilities_for(self, info_state, info_state_tensor=None, legal_actions=None):
        return self.action_probabilities

    def average_action_probabilities_for(self, info_state, info_state_tensor=None, legal_actions=None):
        return self.action_probabilities


def always_bet_agent(random_generator=None):
    return FixedPolicyAgent("always_bet", 0.0, 1.0, random_generator)


def always_pass_agent(random_generator=None):
    return FixedPolicyAgent("always_pass", 1.0, 0.0, random_generator)


def uniform_random_agent(random_generator=None):
    return FixedPolicyAgent("uniform_random", 0.5, 0.5, random_generator)
