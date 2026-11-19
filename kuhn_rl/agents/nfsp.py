"""Simplified Neural Fictitious Self-Play, written from scratch (Heinrich & Silver, 2016).

Each NFSP agent has two networks:
  1. a DQN that learns a best response to the opponent (dqn.py, reused as-is)
  2. an average-policy network that learns, by supervised classification,
     to imitate the agent's own past best-response actions

At the start of each hand the agent picks a mode:
  * with probability eta (anticipatory_parameter): play the DQN's epsilon-greedy
    best response, and store (state, action) in a reservoir buffer
  * otherwise: sample an action from the average-policy network

The DQN learns from every transition in both modes. The average-policy network
is trained to predict the stored actions, so its output approximates the
average of the agent's past best responses: fictitious play, with networks.

What we simplified compared with the paper: one small hidden layer, Adam instead
of plain SGD, and an epsilon schedule shared with our other agents so all
methods explore the same way.
"""

import numpy as np
import torch
import torch.nn.functional as functional

from kuhn_rl.agents.dqn import DQNAgent
from kuhn_rl.agents.neural_components import ReservoirBuffer, build_multilayer_perceptron
from kuhn_rl.agents.policy_helpers import legal_action_mask


class NFSPAgent:
    def __init__(
        self,
        state_size,
        number_of_actions=2,
        anticipatory_parameter=0.1,
        hidden_layer_sizes=(64,),
        average_policy_learning_rate=1e-3,
        average_policy_batch_size=128,
        reservoir_buffer_capacity=200_000,
        minimum_samples_before_supervised_learning=1_000,
        learn_every_decisions=16,
        random_generator=None,
        **best_response_settings,
    ):
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.anticipatory_parameter = anticipatory_parameter
        self.average_policy_batch_size = average_policy_batch_size
        self.minimum_samples_before_supervised_learning = minimum_samples_before_supervised_learning
        self.learn_every_decisions = learn_every_decisions
        self.random_generator = random_generator or np.random.default_rng()

        self.best_response_learner = DQNAgent(
            state_size=state_size,
            number_of_actions=number_of_actions,
            hidden_layer_sizes=hidden_layer_sizes,
            learn_every_decisions=learn_every_decisions,
            random_generator=self.random_generator,
            **best_response_settings,
        )
        self.average_policy_network = build_multilayer_perceptron(
            state_size, hidden_layer_sizes, number_of_actions
        )
        self.average_policy_optimizer = torch.optim.Adam(
            self.average_policy_network.parameters(), lr=average_policy_learning_rate
        )
        self.reservoir_buffer = ReservoirBuffer(
            reservoir_buffer_capacity, state_size, number_of_actions, self.random_generator
        )

        self.cached_average_policy_by_info_state = {}
        self.decisions_since_last_supervised_step = 0
        self.recent_supervised_losses = []
        self.is_playing_best_response_this_hand = False
        self.is_training = True

    # ------------------------------------------------------------------ #
    # Game-loop interface
    # ------------------------------------------------------------------ #
    def start_hand(self):
        self.best_response_learner.start_hand()
        self.is_playing_best_response_this_hand = (
            not self.is_training or self.random_generator.random() < self.anticipatory_parameter
        )

    def act(self, info_state, legal_actions, info_state_tensor):
        self.best_response_learner.observe_decision_point(info_state_tensor, legal_actions)

        if self.is_playing_best_response_this_hand:
            chosen_action = self.best_response_learner.choose_action(info_state, legal_actions, info_state_tensor)
            if self.is_training:
                self.reservoir_buffer.add(
                    info_state_tensor, chosen_action, legal_action_mask(legal_actions, self.number_of_actions)
                )
        else:
            average_policy = self.average_action_probabilities_for(info_state, info_state_tensor, legal_actions)
            chosen_action = int(self.random_generator.choice(self.number_of_actions, p=average_policy))

        self.best_response_learner.remember_action(info_state_tensor, chosen_action)
        if self.is_training:
            self._count_decision_and_maybe_learn()
        return chosen_action

    def end_hand(self, final_reward):
        self.best_response_learner.end_hand(final_reward)

    # ------------------------------------------------------------------ #
    # Supervised learning of the average policy
    # ------------------------------------------------------------------ #
    def _count_decision_and_maybe_learn(self):
        self.decisions_since_last_supervised_step += 1
        if (
            self.decisions_since_last_supervised_step >= self.learn_every_decisions
            and len(self.reservoir_buffer) >= self.minimum_samples_before_supervised_learning
        ):
            self.decisions_since_last_supervised_step = 0
            self._take_supervised_step()

    def _take_supervised_step(self):
        states, taken_actions, legal_masks = self.reservoir_buffer.sample(self.average_policy_batch_size)
        logits = self.average_policy_network(states) + (1.0 - legal_masks) * -1e9
        loss = functional.cross_entropy(logits, taken_actions)
        self.average_policy_optimizer.zero_grad()
        loss.backward()
        self.average_policy_optimizer.step()
        self.recent_supervised_losses.append(loss.item())
        self.cached_average_policy_by_info_state.clear()

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #
    def action_probabilities_for(self, info_state, info_state_tensor, legal_actions=None):
        """Current policy = the DQN's greedy best response (moves around a lot)."""
        return self.best_response_learner.action_probabilities_for(info_state, info_state_tensor, legal_actions)

    def average_action_probabilities_for(self, info_state, info_state_tensor, legal_actions=None):
        """Average policy = output of the average-policy network (what NFSP is judged on)."""
        if legal_actions is None:
            legal_actions = list(range(self.number_of_actions))
        probabilities = self.cached_average_policy_by_info_state.get(info_state)
        if probabilities is None:
            with torch.no_grad():
                logits = self.average_policy_network(torch.from_numpy(info_state_tensor)).numpy()
            mask = legal_action_mask(legal_actions, self.number_of_actions)
            logits = np.where(mask > 0, logits, -np.inf)
            exponentials = np.exp(logits - logits.max())
            probabilities = exponentials / exponentials.sum()
            self.cached_average_policy_by_info_state[info_state] = probabilities
        return probabilities
