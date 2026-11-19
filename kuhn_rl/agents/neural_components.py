"""Building blocks for the neural agents: network, replay buffer, reservoir buffer."""

import numpy as np
import torch
from torch import nn


def build_multilayer_perceptron(input_size, hidden_layer_sizes, output_size):
    layers = []
    previous_size = input_size
    for hidden_size in hidden_layer_sizes:
        layers += [nn.Linear(previous_size, hidden_size), nn.ReLU()]
        previous_size = hidden_size
    layers.append(nn.Linear(previous_size, output_size))
    return nn.Sequential(*layers)


class ReplayBuffer:
    """Fixed-size circular buffer of transitions for DQN (Mnih et al., 2015).
    Once full, the oldest transition is overwritten."""

    def __init__(self, capacity, state_size, number_of_actions, random_generator):
        self.capacity = capacity
        self.random_generator = random_generator
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_legal_action_masks = np.zeros((capacity, number_of_actions), dtype=np.float32)
        self.is_terminal = np.zeros(capacity, dtype=np.float32)
        self.next_write_index = 0
        self.number_stored = 0

    def add(self, state, action, reward, next_state, next_legal_action_mask, is_terminal):
        index = self.next_write_index
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.next_legal_action_masks[index] = next_legal_action_mask
        self.is_terminal[index] = float(is_terminal)
        self.next_write_index = (index + 1) % self.capacity
        self.number_stored = min(self.number_stored + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.random_generator.integers(0, self.number_stored, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.next_legal_action_masks[indices]),
            torch.from_numpy(self.is_terminal[indices]),
        )

    def __len__(self):
        return self.number_stored


class ReservoirBuffer:
    """Keeps a uniform random sample of *everything* ever added (reservoir sampling).

    NFSP uses this for the average-policy network: a uniform sample over all of
    the agent's past best-response actions is what lets supervised learning
    recover the *average* of past policies (Heinrich & Silver, 2016).
    A circular buffer would instead forget old behavior.
    """

    def __init__(self, capacity, state_size, number_of_actions, random_generator):
        self.capacity = capacity
        self.random_generator = random_generator
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.legal_action_masks = np.zeros((capacity, number_of_actions), dtype=np.float32)
        self.number_stored = 0
        self.number_ever_added = 0

    def add(self, state, action, legal_action_mask):
        self.number_ever_added += 1
        if self.number_stored < self.capacity:
            index = self.number_stored
            self.number_stored += 1
        else:
            # Keep the new item with probability capacity / number_ever_added.
            index = int(self.random_generator.integers(0, self.number_ever_added))
            if index >= self.capacity:
                return
        self.states[index] = state
        self.actions[index] = action
        self.legal_action_masks[index] = legal_action_mask

    def sample(self, batch_size):
        indices = self.random_generator.integers(0, self.number_stored, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.legal_action_masks[indices]),
        )

    def __len__(self):
        return self.number_stored
