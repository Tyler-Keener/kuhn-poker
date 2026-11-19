"""Deep Q-Network agent written from scratch in PyTorch (Mnih et al., 2015).

Same idea as tabular_q_learning.py, but Q(s, a) comes from a neural network
that reads OpenSpiel's information_state_tensor instead of a lookup table.
The two DQN stabilizers are included:
  * experience replay: learn from random past transitions, not just the latest
  * target network:    a slowly updated copy used to compute the learning target

    target = reward + gamma * (1 - terminal) * max_a' Q_target(s', a')

Optional Double DQN (van Hasselt, Guez & Silver, 2016): pick a' with the online
network, score it with the target network. Reduces overestimation.

Loss function: mean squared error, NOT the Huber loss of the original DQN.
Huber loss acts like absolute error for errors above 1, so it estimates
something between the mean and the median of the returns. Atari DQN clipped
rewards to [-1, 1], where this hardly matters. In Kuhn poker one action can
end in +1 or -2: with Huber loss, Q(Queen, bet first) came out near -0.06
instead of about -0.4, so the agent wrongly preferred betting the Queen.
Mean squared error estimates the mean, which is what Q-learning needs.
Set loss_function="huber" to reproduce the problem.

As in the tabular agent, act() is split into observe / choose / remember so
that NFSP can reuse this agent as its best-response learner.

Speed note: Kuhn poker has only 12 information states, so between gradient
steps we cache the network's output for each one. The cache is cleared after
every gradient step, so it never returns stale values; it only avoids
recomputing the same forward pass thousands of times.
"""

import numpy as np
import torch
import torch.nn.functional as functional

from kuhn_rl.agents.neural_components import ReplayBuffer, build_multilayer_perceptron
from kuhn_rl.agents.policy_helpers import ActionFrequencyTable, greedy_probabilities, legal_action_mask


class DQNAgent:
    def __init__(
        self,
        state_size,
        number_of_actions=2,
        hidden_layer_sizes=(64,),
        learning_rate=1e-3,
        discount_factor=1.0,
        batch_size=128,
        replay_buffer_capacity=100_000,
        minimum_transitions_before_learning=1_000,
        learn_every_decisions=16,
        update_target_every_gradient_steps=250,
        use_double_dqn=False,
        loss_function="mse",
        starting_epsilon=0.3,
        minimum_epsilon=0.02,
        epsilon_decay_per_hand=0.99995,
        random_generator=None,
    ):
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.minimum_transitions_before_learning = minimum_transitions_before_learning
        self.learn_every_decisions = learn_every_decisions
        self.update_target_every_gradient_steps = update_target_every_gradient_steps
        self.use_double_dqn = use_double_dqn
        if loss_function not in ("mse", "huber"):
            raise ValueError("loss_function must be 'mse' or 'huber'")
        self.loss_function = loss_function
        self.epsilon = starting_epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay_per_hand = epsilon_decay_per_hand
        self.random_generator = random_generator or np.random.default_rng()

        self.online_q_network = build_multilayer_perceptron(state_size, hidden_layer_sizes, number_of_actions)
        self.target_q_network = build_multilayer_perceptron(state_size, hidden_layer_sizes, number_of_actions)
        self.target_q_network.load_state_dict(self.online_q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.online_q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(
            replay_buffer_capacity, state_size, number_of_actions, self.random_generator
        )

        self.played_action_frequencies = ActionFrequencyTable(number_of_actions)  # measurement only
        self.cached_q_values_by_info_state = {}
        self.decisions_since_last_learning = 0
        self.gradient_steps_taken = 0
        self.recent_losses = []

        self.is_training = True
        self.pending_info_state_tensor = None
        self.pending_action = None

    # ------------------------------------------------------------------ #
    # Game-loop interface
    # ------------------------------------------------------------------ #
    def start_hand(self):
        self.pending_info_state_tensor = None
        self.pending_action = None

    def act(self, info_state, legal_actions, info_state_tensor):
        self.observe_decision_point(info_state_tensor, legal_actions)
        chosen_action = self.choose_action(info_state, legal_actions, info_state_tensor)
        self.remember_action(info_state_tensor, chosen_action)
        if self.is_training:
            self.played_action_frequencies.record(info_state, chosen_action)
        return chosen_action

    def end_hand(self, final_reward):
        if self.is_training and self.pending_info_state_tensor is not None:
            self.replay_buffer.add(
                state=self.pending_info_state_tensor,
                action=self.pending_action,
                reward=final_reward,
                next_state=np.zeros(self.state_size, dtype=np.float32),
                next_legal_action_mask=np.zeros(self.number_of_actions, dtype=np.float32),
                is_terminal=True,
            )
            self._count_decision_and_maybe_learn()
        if self.is_training:
            self.epsilon = max(self.minimum_epsilon, self.epsilon * self.epsilon_decay_per_hand)
        self.pending_info_state_tensor = None
        self.pending_action = None

    # ------------------------------------------------------------------ #
    # The three pieces of act()
    # ------------------------------------------------------------------ #
    def observe_decision_point(self, info_state_tensor, legal_actions):
        if self.is_training and self.pending_info_state_tensor is not None:
            self.replay_buffer.add(
                state=self.pending_info_state_tensor,
                action=self.pending_action,
                reward=0.0,  # no chips change hands mid-hand
                next_state=info_state_tensor,
                next_legal_action_mask=legal_action_mask(legal_actions, self.number_of_actions),
                is_terminal=False,
            )
            self._count_decision_and_maybe_learn()

    def choose_action(self, info_state, legal_actions, info_state_tensor):
        if self.is_training and self.random_generator.random() < self.epsilon:
            return int(self.random_generator.choice(legal_actions))
        greedy_policy = self.action_probabilities_for(info_state, info_state_tensor, legal_actions)
        return int(self.random_generator.choice(self.number_of_actions, p=greedy_policy))

    def remember_action(self, info_state_tensor, action):
        self.pending_info_state_tensor = info_state_tensor
        self.pending_action = action

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def _count_decision_and_maybe_learn(self):
        self.decisions_since_last_learning += 1
        if (
            self.decisions_since_last_learning >= self.learn_every_decisions
            and len(self.replay_buffer) >= self.minimum_transitions_before_learning
        ):
            self.decisions_since_last_learning = 0
            self._take_gradient_step()

    def _take_gradient_step(self):
        states, actions, rewards, next_states, next_legal_masks, is_terminal = self.replay_buffer.sample(
            self.batch_size
        )
        with torch.no_grad():
            illegal_action_penalty = (1.0 - next_legal_masks) * -1e9
            next_target_q_values = self.target_q_network(next_states) + illegal_action_penalty
            if self.use_double_dqn:
                next_online_q_values = self.online_q_network(next_states) + illegal_action_penalty
                best_next_actions = next_online_q_values.argmax(dim=1, keepdim=True)
                best_next_values = next_target_q_values.gather(1, best_next_actions).squeeze(1)
            else:
                best_next_values = next_target_q_values.max(dim=1).values
            # Terminal transitions have an all-zero mask; (1 - terminal) zeroes them out.
            target_values = rewards + self.discount_factor * (1.0 - is_terminal) * best_next_values

        predicted_values = self.online_q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        if self.loss_function == "huber":
            loss = functional.smooth_l1_loss(predicted_values, target_values)
        else:
            loss = functional.mse_loss(predicted_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.recent_losses.append(loss.item())
        self.cached_q_values_by_info_state.clear()
        self.gradient_steps_taken += 1
        if self.gradient_steps_taken % self.update_target_every_gradient_steps == 0:
            self.target_q_network.load_state_dict(self.online_q_network.state_dict())

    def q_values_for(self, info_state, info_state_tensor):
        q_values = self.cached_q_values_by_info_state.get(info_state)
        if q_values is None:
            with torch.no_grad():
                q_values = self.online_q_network(torch.from_numpy(info_state_tensor)).numpy()
            self.cached_q_values_by_info_state[info_state] = q_values
        return q_values

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #
    def action_probabilities_for(self, info_state, info_state_tensor, legal_actions=None):
        """Greedy policy (exploration off)."""
        if legal_actions is None:
            legal_actions = list(range(self.number_of_actions))
        return greedy_probabilities(
            self.q_values_for(info_state, info_state_tensor), legal_actions, self.number_of_actions
        )

    def average_action_probabilities_for(self, info_state, info_state_tensor=None, legal_actions=None):
        """Empirical frequency of the actions actually played during training."""
        if legal_actions is None:
            legal_actions = list(range(self.number_of_actions))
        return self.played_action_frequencies.probabilities(info_state, legal_actions)
