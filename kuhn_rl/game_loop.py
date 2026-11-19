"""Plays one hand of a game using OpenSpiel only as the rules engine."""

import numpy as np


def play_one_hand(game, agents_by_seat, random_generator):
    """Play a single hand and return each seat's net chips.

    game:             an OpenSpiel game, e.g. pyspiel.load_game("kuhn_poker")
    agents_by_seat:   [agent_in_seat_0, agent_in_seat_1]
    random_generator: numpy Generator used for the card deal
    """
    for agent in agents_by_seat:
        agent.start_hand()

    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            # The deal: OpenSpiel lists each possible card with its probability.
            possible_outcomes, outcome_probabilities = zip(*state.chance_outcomes())
            dealt_outcome = random_generator.choice(possible_outcomes, p=outcome_probabilities)
            state.apply_action(int(dealt_outcome))
        else:
            acting_seat = state.current_player()
            info_state = state.information_state_string(acting_seat)
            info_state_tensor = np.asarray(state.information_state_tensor(acting_seat), dtype=np.float32)
            chosen_action = agents_by_seat[acting_seat].act(info_state, state.legal_actions(), info_state_tensor)
            state.apply_action(chosen_action)

    final_rewards_by_seat = state.returns()
    for seat, agent in enumerate(agents_by_seat):
        agent.end_hand(final_rewards_by_seat[seat])
    return np.array(final_rewards_by_seat)
