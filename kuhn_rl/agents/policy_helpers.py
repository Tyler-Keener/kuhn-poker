"""Small helpers shared by every agent.

Agent interface used throughout this project
--------------------------------------------
    start_hand()
    act(info_state, legal_actions, info_state_tensor) -> action
    end_hand(final_reward)

    action_probabilities_for(info_state, info_state_tensor, legal_actions)
        the agent's *current* policy (greedy for value-based learners)
    average_action_probabilities_for(info_state, info_state_tensor, legal_actions)
        the agent's *average* policy over training

info_state is OpenSpiel's string (e.g. "1pb"); info_state_tensor is the same
information as a vector of numbers, which the neural-network agents need.
"""

from collections import defaultdict

import numpy as np


def uniform_over_legal_actions(legal_actions, number_of_actions):
    probabilities = np.zeros(number_of_actions)
    probabilities[list(legal_actions)] = 1.0 / len(legal_actions)
    return probabilities


def greedy_probabilities(action_values, legal_actions, number_of_actions):
    """Probability 1 on the best legal action; ties are split evenly."""
    best_value = max(action_values[action] for action in legal_actions)
    best_legal_actions = [action for action in legal_actions if action_values[action] == best_value]
    probabilities = np.zeros(number_of_actions)
    probabilities[best_legal_actions] = 1.0 / len(best_legal_actions)
    return probabilities


def legal_action_mask(legal_actions, number_of_actions):
    mask = np.zeros(number_of_actions, dtype=np.float32)
    mask[list(legal_actions)] = 1.0
    return mask


class ActionFrequencyTable:
    """Counts how often each action was taken in each information state.

    Normalizing the counts gives the empirical *average* policy. Because states
    are counted every time they are reached, this is the reach-weighted average
    that fictitious play needs (Heinrich, Lanctot & Silver, 2015).
    """

    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions
        self.action_counts_by_info_state = defaultdict(lambda: np.zeros(number_of_actions))

    def record(self, info_state, action):
        self.action_counts_by_info_state[info_state][action] += 1

    def probabilities(self, info_state, legal_actions):
        legal_counts = np.zeros(self.number_of_actions)
        counts = self.action_counts_by_info_state.get(info_state)
        if counts is not None:
            legal_counts[list(legal_actions)] = counts[list(legal_actions)]
        if legal_counts.sum() == 0:
            return uniform_over_legal_actions(legal_actions, self.number_of_actions)
        return legal_counts / legal_counts.sum()
