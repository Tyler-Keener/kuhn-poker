"""The four self-play methods compared in the main experiment, and their settings.

                      no averaging (naive)     averaging (fictitious play)
    tabular           naive_tabular_q          tabular_fsp
    neural network    naive_dqn                nfsp

This 2 x 2 design separates the effect of averaging (columns) from the effect
of function approximation (rows).

Every hyperparameter lives in DEFAULT_HYPERPARAMETERS so it is saved with each
run and can be overridden from the command line with  --set name=value.
See HYPERPARAMETERS.md for how each value was chosen.
"""

import copy

import torch

from kuhn_rl.agents import DQNAgent, NFSPAgent, TabularFictitiousSelfPlayAgent, TabularQLearningAgent

SHARED_EXPLORATION = {
    "starting_epsilon": 0.3,
    "minimum_epsilon": 0.02,
    "epsilon_decay_per_hand": 0.99995,
}

SHARED_DQN_SETTINGS = {
    "hidden_layer_sizes": (64,),
    "learning_rate": 1e-3,
    "batch_size": 128,
    "replay_buffer_capacity": 100_000,
    "minimum_transitions_before_learning": 1_000,
    "learn_every_decisions": 16,
    "update_target_every_gradient_steps": 250,
    "use_double_dqn": False,
}

DEFAULT_HYPERPARAMETERS = {
    "naive_tabular_q": {"learning_rate": 0.01, **SHARED_EXPLORATION},
    "tabular_fsp": {"learning_rate": 0.01, "anticipatory_parameter": 0.1, **SHARED_EXPLORATION},
    "naive_dqn": {**SHARED_DQN_SETTINGS, **SHARED_EXPLORATION},
    "nfsp": {
        **SHARED_DQN_SETTINGS,
        **SHARED_EXPLORATION,
        "anticipatory_parameter": 0.1,
        "average_policy_learning_rate": 1e-3,
        "average_policy_batch_size": 128,
        "reservoir_buffer_capacity": 200_000,
        "minimum_samples_before_supervised_learning": 1_000,
    },
}

DEFAULT_TRAINING_HANDS = {
    "naive_tabular_q": 500_000,
    "tabular_fsp": 500_000,
    "naive_dqn": 2_000_000,
    "nfsp": 2_000_000,
}

# Which policy each method is judged on (the other is still recorded).
PRIMARY_POLICY = {
    "naive_tabular_q": "current",
    "tabular_fsp": "average",
    "naive_dqn": "current",
    "nfsp": "average",
}

DISPLAY_NAMES = {
    "naive_tabular_q": "Naive self-play, tabular Q",
    "tabular_fsp": "Fictitious self-play, tabular",
    "naive_dqn": "Naive self-play, DQN",
    "nfsp": "NFSP (neural fictitious self-play)",
}

METHOD_NAMES = list(DEFAULT_HYPERPARAMETERS)


def hyperparameters_for(method_name, overrides=None):
    settings = copy.deepcopy(DEFAULT_HYPERPARAMETERS[method_name])
    for name, value in (overrides or {}).items():
        if name not in settings:
            raise ValueError(f"{method_name} has no setting called {name!r}. Options: {sorted(settings)}")
        settings[name] = value
    return settings


def build_agent(method_name, game, hyperparameters, random_generator):
    number_of_actions = game.num_distinct_actions()
    state_size = game.information_state_tensor_size()
    settings = dict(hyperparameters)
    if "hidden_layer_sizes" in settings:
        settings["hidden_layer_sizes"] = tuple(settings["hidden_layer_sizes"])

    if method_name == "naive_tabular_q":
        return TabularQLearningAgent(number_of_actions=number_of_actions, random_generator=random_generator, **settings)
    if method_name == "tabular_fsp":
        return TabularFictitiousSelfPlayAgent(
            number_of_actions=number_of_actions, random_generator=random_generator, **settings
        )
    if method_name == "naive_dqn":
        return DQNAgent(state_size, number_of_actions=number_of_actions, random_generator=random_generator, **settings)
    if method_name == "nfsp":
        return NFSPAgent(state_size, number_of_actions=number_of_actions, random_generator=random_generator, **settings)
    raise ValueError(f"Unknown method {method_name!r}. Options: {METHOD_NAMES}")


def seed_everything(seed):
    """Make a run repeatable: torch weights and our numpy generator both use the seed."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # we run several seeds in parallel instead
