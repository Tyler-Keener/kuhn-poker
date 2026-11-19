"""Main experiment (Plan steps 2-4): train one self-play method over many seeds.

Two agents of the same method train against each other: agent A always sits
in seat 0 and agent B in seat 1, so together they form one strategy profile.
At every checkpoint we record, exactly (no sampling noise):

  * current_exploitability   exploitability of both agents' current policies
                             (greedy for value-based learners)
  * average_exploitability   exploitability of both agents' average policies
                             (for naive learners: the empirical frequency of
                             the actions they played, measured after the fact)
  * policy_flips             information states whose greedy action changed
                             since the previous checkpoint (a cycling signal)

Seeds run in parallel, one process each.

Run from the repository root, for example:
    python -m experiments.self_play --method tabular_fsp --seeds 10
    python -m experiments.self_play --method nfsp --seeds 10 --hands 2000000
    python -m experiments.self_play --method naive_dqn --seeds 3 --set learning_rate=0.0005
"""

import argparse
import ast
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pyspiel

from experiments.methods import (
    DEFAULT_TRAINING_HANDS,
    DISPLAY_NAMES,
    METHOD_NAMES,
    PRIMARY_POLICY,
    build_agent,
    hyperparameters_for,
    seed_everything,
)
from kuhn_rl.evaluation import (
    build_openspiel_policy,
    compute_exploitability,
    count_policy_flips,
    greedy_action_per_info_state,
)
from kuhn_rl.game_loop import play_one_hand


def run_one_seed(job):
    method_name, game_name, number_of_training_hands, evaluate_every_hands, seed, hyperparameters = job
    seed_everything(seed)
    random_generator = np.random.default_rng(seed)
    game = pyspiel.load_game(game_name)
    agent_in_seat_0 = build_agent(method_name, game, hyperparameters, random_generator)
    agent_in_seat_1 = build_agent(method_name, game, hyperparameters, random_generator)

    current_exploitability, average_exploitability, policy_flips = [], [], []
    previous_greedy_actions = None
    start_time = time.time()
    for hand_number in range(1, number_of_training_hands + 1):
        play_one_hand(game, [agent_in_seat_0, agent_in_seat_1], random_generator)

        if hand_number % evaluate_every_hands == 0:
            current_policy = build_openspiel_policy(game, agent_in_seat_0, agent_in_seat_1)
            average_policy = build_openspiel_policy(
                game, agent_in_seat_0, agent_in_seat_1, use_average_policy=True
            )
            current_exploitability.append(compute_exploitability(game, current_policy))
            average_exploitability.append(compute_exploitability(game, average_policy))
            greedy_actions = greedy_action_per_info_state(current_policy)
            policy_flips.append(count_policy_flips(previous_greedy_actions, greedy_actions))
            previous_greedy_actions = greedy_actions

    return {
        "seed": seed,
        "current_exploitability": np.array(current_exploitability),
        "average_exploitability": np.array(average_exploitability),
        "policy_flips": np.array(policy_flips),
        "seconds": time.time() - start_time,
    }


def run_method(method_name, game_name, number_of_training_hands, evaluate_every_hands, seeds,
               hyperparameters, number_of_workers):
    jobs = [
        (method_name, game_name, number_of_training_hands, evaluate_every_hands, seed, hyperparameters)
        for seed in seeds
    ]
    if number_of_workers <= 1:
        results = [run_one_seed(job) for job in jobs]
    else:
        with multiprocessing.get_context("spawn").Pool(number_of_workers) as worker_pool:
            results = worker_pool.map(run_one_seed, jobs)
    return {
        "training_hands": np.arange(evaluate_every_hands, number_of_training_hands + 1, evaluate_every_hands),
        "seeds": np.array([result["seed"] for result in results]),
        "current_exploitability_by_seed": np.stack([result["current_exploitability"] for result in results]),
        "average_exploitability_by_seed": np.stack([result["average_exploitability"] for result in results]),
        "policy_flips_by_seed": np.stack([result["policy_flips"] for result in results]),
        "seconds_by_seed": np.array([result["seconds"] for result in results]),
    }


def parse_overrides(setting_strings):
    """Turn ["learning_rate=0.001", "hidden_layer_sizes=(128,)"] into a dict."""
    overrides = {}
    for setting in setting_strings or []:
        name, raw_value = setting.split("=", 1)
        try:
            overrides[name] = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            overrides[name] = raw_value
    return overrides


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--method", required=True, choices=METHOD_NAMES)
    argument_parser.add_argument("--game", default="kuhn_poker", help="kuhn_poker or leduc_poker")
    argument_parser.add_argument("--hands", type=int, default=None, help="training hands (default depends on method)")
    argument_parser.add_argument("--evaluate-every", type=int, default=None, help="default: hands / 100")
    argument_parser.add_argument("--seeds", type=int, default=10)
    argument_parser.add_argument("--first-seed", type=int, default=0)
    argument_parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    argument_parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                                 help="override hyperparameters, e.g. --set learning_rate=0.0005")
    argument_parser.add_argument("--output-directory", default="results/self_play")
    arguments = argument_parser.parse_args()

    number_of_training_hands = arguments.hands or DEFAULT_TRAINING_HANDS[arguments.method]
    evaluate_every_hands = arguments.evaluate_every or max(1, number_of_training_hands // 100)
    hyperparameters = hyperparameters_for(arguments.method, parse_overrides(arguments.set))
    seeds = list(range(arguments.first_seed, arguments.first_seed + arguments.seeds))

    print(f"{DISPLAY_NAMES[arguments.method]} on {arguments.game}: {number_of_training_hands:,} hands, "
          f"{len(seeds)} seeds, {arguments.workers} workers")
    start_time = time.time()
    results = run_method(arguments.method, arguments.game, number_of_training_hands, evaluate_every_hands,
                         seeds, hyperparameters, arguments.workers)

    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = output_directory / f"{arguments.game}_{arguments.method}"
    np.savez(f"{output_stem}.npz", **results)
    Path(f"{output_stem}_config.json").write_text(json.dumps({
        "method": arguments.method,
        "game": arguments.game,
        "training_hands": number_of_training_hands,
        "evaluate_every_hands": evaluate_every_hands,
        "seeds": seeds,
        "hyperparameters": hyperparameters,
    }, indent=2))

    primary_key = f"{PRIMARY_POLICY[arguments.method]}_exploitability_by_seed"
    final_values = results[primary_key][:, -1]
    print(f"Final exploitability ({PRIMARY_POLICY[arguments.method]} policy): "
          f"{final_values.mean():.4f} +/- {final_values.std(ddof=1) if len(seeds) > 1 else 0:.4f} (std over seeds)")
    print(f"Final exploitability (current policy): {results['current_exploitability_by_seed'][:, -1].mean():.4f}")
    print(f"Final exploitability (average policy): {results['average_exploitability_by_seed'][:, -1].mean():.4f}")
    print(f"Mean policy flips per checkpoint, last half of training: "
          f"{results['policy_flips_by_seed'][:, results['policy_flips_by_seed'].shape[1] // 2:].mean():.2f}")
    print(f"Saved {output_stem}.npz and config ({time.time() - start_time:.0f}s total)")


if __name__ == "__main__":
    main()
