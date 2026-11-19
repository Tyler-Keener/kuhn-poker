"""Tabular Q-learning written from scratch (no OpenSpiel agents).

The Q-table maps (information state, action) -> estimated chips won from here.
An information state is OpenSpiel's string for what the agent can see:
its own card plus the betting history, e.g. "1pb" = holding the Queen,
it passed, the opponent bet. Kuhn poker has 12 of these in total.

Why the update is delayed by one decision
-----------------------------------------
After the agent acts, the *opponent* moves before the agent sees its next
information state. So we only learn where an action led the next time the
agent is asked to act (or when the hand ends). We store the previous
(info_state, action) as a "pending" transition and finish its update then:

    Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))

with reward = 0 mid-hand, reward = net chips at the end, and gamma = 1.

act() is split into three steps so that the fictitious self-play agent can
reuse this learner while sometimes choosing the action some other way:
    observe_decision_point()  finish the pending update
    choose_action()           epsilon-greedy choice
    remember_action()         store (info_state, action) as the new pending step
"""

from collections import defaultdict

import numpy as np

from kuhn_rl.agents.policy_helpers import ActionFrequencyTable, greedy_probabilities


class TabularQLearningAgent:
    def __init__(
        self,
        number_of_actions=2,
        learning_rate=0.01,
        discount_factor=1.0,
        starting_epsilon=0.3,
        minimum_epsilon=0.02,
        epsilon_decay_per_hand=0.99995,
        random_generator=None,
    ):
        self.number_of_actions = number_of_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = starting_epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay_per_hand = epsilon_decay_per_hand
        self.random_generator = random_generator or np.random.default_rng()

        # Unseen states start at Q = 0 for every action.
        self.q_values_by_info_state = defaultdict(lambda: np.zeros(self.number_of_actions))

        # Not part of the algorithm: a record of the actions this agent actually
        # took, so we can measure the time-averaged policy of naive self-play.
        self.played_action_frequencies = ActionFrequencyTable(number_of_actions)

        self.is_training = True
        self.pending_info_state = None
        self.pending_action = None

    # ------------------------------------------------------------------ #
    # Game-loop interface
    # ------------------------------------------------------------------ #
    def start_hand(self):
        self.pending_info_state = None
        self.pending_action = None

    def act(self, info_state, legal_actions, info_state_tensor=None):
        self.observe_decision_point(info_state, legal_actions)
        chosen_action = self.choose_action(info_state, legal_actions)
        self.remember_action(info_state, chosen_action)
        if self.is_training:
            self.played_action_frequencies.record(info_state, chosen_action)
        return chosen_action

    def end_hand(self, final_reward):
        if self.is_training and self.pending_info_state is not None:
            self._update_q_value(
                info_state=self.pending_info_state,
                action=self.pending_action,
                reward=final_reward,
                next_info_state=None,  # terminal: nothing left to win or lose
                next_legal_actions=None,
            )
        if self.is_training:
            self.epsilon = max(self.minimum_epsilon, self.epsilon * self.epsilon_decay_per_hand)
        self.pending_info_state = None
        self.pending_action = None

    # ------------------------------------------------------------------ #
    # The three pieces of act()
    # ------------------------------------------------------------------ #
    def observe_decision_point(self, info_state, legal_actions):
        if self.is_training and self.pending_info_state is not None:
            # Mid-hand step: no chips change hands yet, so reward is 0.
            self._update_q_value(
                info_state=self.pending_info_state,
                action=self.pending_action,
                reward=0.0,
                next_info_state=info_state,
                next_legal_actions=legal_actions,
            )

    def choose_action(self, info_state, legal_actions):
        if self.is_training and self.random_generator.random() < self.epsilon:
            return int(self.random_generator.choice(legal_actions))
        greedy_policy = greedy_probabilities(
            self.q_values_by_info_state[info_state], legal_actions, self.number_of_actions
        )
        return int(self.random_generator.choice(self.number_of_actions, p=greedy_policy))

    def remember_action(self, info_state, action):
        self.pending_info_state = info_state
        self.pending_action = action

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def _update_q_value(self, info_state, action, reward, next_info_state, next_legal_actions):
        if next_info_state is None:
            best_future_value = 0.0
        else:
            next_q_values = self.q_values_by_info_state[next_info_state]
            best_future_value = max(next_q_values[a] for a in next_legal_actions)

        target_value = reward + self.discount_factor * best_future_value
        current_value = self.q_values_by_info_state[info_state][action]
        temporal_difference_error = target_value - current_value
        self.q_values_by_info_state[info_state][action] += self.learning_rate * temporal_difference_error

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #
    def action_probabilities_for(self, info_state, info_state_tensor=None, legal_actions=None):
        """Greedy policy (exploration off). Unvisited states come out uniform."""
        if legal_actions is None:
            legal_actions = list(range(self.number_of_actions))
        return greedy_probabilities(
            self.q_values_by_info_state[info_state], legal_actions, self.number_of_actions
        )

    def average_action_probabilities_for(self, info_state, info_state_tensor=None, legal_actions=None):
        """Empirical frequency of the actions actually played during training."""
        if legal_actions is None:
            legal_actions = list(range(self.number_of_actions))
        return self.played_action_frequencies.probabilities(info_state, legal_actions)
