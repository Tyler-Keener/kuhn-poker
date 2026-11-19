"""Tabular fictitious self-play: the table-based version of NFSP.

This is the "simple fictitious play with tabular averaging" fallback from the
proposal, built so that it matches NFSP (Heinrich & Silver, 2016) piece for piece:

    NFSP                                  this agent
    ------------------------------------  ------------------------------------
    DQN network  -> best response         tabular Q-learning -> best response
    average-policy network (supervised)   table of action counts
    reservoir buffer of (state, action)   the counts themselves (same average)
    anticipatory parameter eta            same

At the start of each hand the agent picks a mode:
  * with probability eta:      play its epsilon-greedy best response, and record
                               the action it took into the average-policy table
  * with probability 1 - eta:  play its average policy

The Q-learner learns from every hand in both modes. Because the opponent also
mostly plays its *average* policy, the Q-learner is learning a best response to
the opponent's average past play, which is exactly fictitious play
(Brown, 1951; Robinson, 1951). The average policy is what we evaluate.

Comparing this agent with naive tabular Q-learning isolates the effect of
averaging, with no neural-network approximation involved.
"""

import numpy as np

from kuhn_rl.agents.policy_helpers import ActionFrequencyTable
from kuhn_rl.agents.tabular_q_learning import TabularQLearningAgent


class TabularFictitiousSelfPlayAgent:
    def __init__(
        self,
        number_of_actions=2,
        anticipatory_parameter=0.1,
        random_generator=None,
        **best_response_settings,
    ):
        self.number_of_actions = number_of_actions
        self.anticipatory_parameter = anticipatory_parameter
        self.random_generator = random_generator or np.random.default_rng()
        self.best_response_learner = TabularQLearningAgent(
            number_of_actions=number_of_actions,
            random_generator=self.random_generator,
            **best_response_settings,
        )
        self.average_policy_table = ActionFrequencyTable(number_of_actions)
        self.is_playing_best_response_this_hand = False
        self.is_training = True

    def start_hand(self):
        self.best_response_learner.start_hand()
        self.is_playing_best_response_this_hand = (
            not self.is_training or self.random_generator.random() < self.anticipatory_parameter
        )

    def act(self, info_state, legal_actions, info_state_tensor=None):
        self.best_response_learner.observe_decision_point(info_state, legal_actions)

        if self.is_playing_best_response_this_hand:
            chosen_action = self.best_response_learner.choose_action(info_state, legal_actions)
            self.average_policy_table.record(info_state, chosen_action)
        else:
            average_policy = self.average_policy_table.probabilities(info_state, legal_actions)
            chosen_action = int(self.random_generator.choice(self.number_of_actions, p=average_policy))

        # Q-learning is off-policy, so it can learn from average-policy moves too.
        self.best_response_learner.remember_action(info_state, chosen_action)
        return chosen_action

    def end_hand(self, final_reward):
        self.best_response_learner.end_hand(final_reward)

    def action_probabilities_for(self, info_state, info_state_tensor=None, legal_actions=None):
        """Current policy = the greedy best response (moves around a lot)."""
        return self.best_response_learner.action_probabilities_for(info_state, info_state_tensor, legal_actions)

    def average_action_probabilities_for(self, info_state, info_state_tensor=None, legal_actions=None):
        """Average policy = the output of fictitious play."""
        if legal_actions is None:
            legal_actions = list(range(self.number_of_actions))
        return self.average_policy_table.probabilities(info_state, legal_actions)
