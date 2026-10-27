"""Train and evaluate a masked PPO policy on the Cruise Control simulator.

Run from the repository root with the project virtual environment:

    python scripts/train_ppo_cruise_control.py --timesteps 10000
    python scripts/train_ppo_cruise_control.py --timesteps 50000 --seed 11

The policy selects among the enabled ``controlFunc`` paths. This trains against the
generated simulator and does not prove the quantified dL stability theorem.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
AST2 = ROOT / "AST_parser_2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AST2) not in sys.path:
    sys.path.insert(0, str(AST2))

from generator_core import generate_gym_environment_file  # pyright: ignore[reportMissingImports]


class EpisodeStatsCallback(BaseCallback):
    """Collect episode reward and length from the Monitor wrapper."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(int(episode["l"]))
        return True


def build_env_class(model_path: Path, generated_path: Path):
    model_text = model_path.read_text(encoding="utf-8")
    if not generate_gym_environment_file(
        model_text, "CruiseControlPPO", str(generated_path)
    ):
        raise RuntimeError(f"Could not generate environment from {model_path}")

    module_name = "generated_cruise_control_ppo_env"
    spec = importlib.util.spec_from_file_location(module_name, generated_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated environment: {generated_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.KeYmaeraGymEnv


def plot_training_curve(callback: EpisodeStatsCallback, output_path: Path) -> None:
    if not callback.episode_rewards:
        print("No training episodes completed; skipping training curve.")
        return

    window = max(1, min(20, len(callback.episode_rewards) // 5))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for axis, values, label in (
        (axes[0], callback.episode_rewards, "Episode reward"),
        (axes[1], callback.episode_lengths, "Episode length (steps)"),
    ):
        axis.plot(values, alpha=0.35, label=label)
        if len(values) >= window:
            average = np.convolve(values, np.ones(window) / window, mode="valid")
            axis.plot(
                range(window - 1, len(values)),
                average,
                label=f"Moving average (window={window})",
            )
        axis.set_ylabel(label)
        axis.legend()
    axes[1].set_xlabel("Completed episode")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved training curve: {output_path}")


def plot_episode_trajectory(
    result: dict,
    output_path: Path,
    policy_name: str = "PPO",
) -> None:
    """Plot a rollout's states, controller modes/actions, and per-step reward."""
    states = result["state_history"]
    steps = np.arange(len(states))
    action_steps = np.arange(1, len(result["chosen_actions"]) + 1)
    reached = result["reached_near_zero_speed"]
    label = "REACHED |v| <= 1" if reached else "DID NOT REACH |v| <= 1"
    transitions = {
        transition["step"]: transition
        for transition in result["mode_transitions"]
    }

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    axes[0].plot(steps, [state.get("v", np.nan) for state in states], label="Velocity v")
    axes[0].axhspan(-1.0, 1.0, color="tab:green", alpha=0.15, label="Near-zero zone")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Velocity")
    axes[0].set_title(
        f"{policy_name}, seed {result['seed']}: {label}; "
        f"reason={result['termination_reason']}"
    )
    axes[0].legend(loc="best")

    axes[1].plot(steps, [state.get("x", np.nan) for state in states], color="tab:blue")
    axes[1].set_ylabel("Position error x")
    axes[1].set_title("Position error")

    axes[2].step(
        steps,
        [state.get("mode", np.nan) for state in states],
        where="post",
        color="tab:purple",
    )
    axes[2].set_ylabel("Mode")
    axes[2].set_yticks(range(6))
    axes[2].set_title("Controller mode")
    for transition_step in transitions:
        axes[2].axvline(transition_step, color="black", alpha=0.18, linewidth=0.8)

    axes[3].step(
        action_steps,
        result["chosen_actions"],
        where="post",
        color="tab:orange",
    )
    axes[3].set_ylabel("Path ID")
    axes[3].set_title("Selected controlFunc path")

    axes[4].plot(
        action_steps,
        result["reward_history"],
        color="tab:green",
        label="Step reward",
    )
    axes[4].plot(
        action_steps,
        np.cumsum(result["reward_history"]),
        color="tab:red",
        alpha=0.8,
        label="Cumulative reward",
    )
    axes[4].set_xlabel("Control step")
    axes[4].set_ylabel("Reward")
    axes[4].legend(loc="best")
    axes[4].set_title(
        f"Steps={result['steps']}, in-zone={result['steps_inside_zone']} "
        f"({result['fraction_steps_inside_zone']:.1%}), "
        f"longest run={result['longest_consecutive_inside_steps']} steps/"
        f"{result['longest_consecutive_inside_simulation_time']:.3f}s, "
        f"finished inside={result['final_inside_zone']}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved cruise-control episode plot: {output_path}")


def plot_representative_episodes(
    results: list[dict],
    output_dir: Path,
    max_per_category: int = 2,
) -> None:
    """Plot representative PPO episodes for each near-zero reachability category."""
    reached = [result for result in results if result["reached_near_zero_speed"]]
    not_reached = [
        result for result in results if not result["reached_near_zero_speed"]
    ]
    categories = (
        (
            "success",
            sorted(
                reached,
                key=lambda result: (
                    result["longest_consecutive_inside_simulation_time"],
                    result["longest_consecutive_inside_steps"],
                ),
                reverse=True,
            ),
        ),
        (
            "failure",
            sorted(
                not_reached,
                key=lambda result: result["total_reward"],
            ),
        ),
    )
    for category, episodes in categories:
        if not episodes:
            print(f"No PPO episode in category '{category}' to plot.")
            continue
        for index, result in enumerate(episodes[:max_per_category], start=1):
            output_path = output_dir / (
                "cruise_control_"
                f"{category}_{index}_seed_{result['seed']}.png"
            )
            plot_episode_trajectory(result, output_path)


def plot_evaluation_summary(
    policy_results: list[dict],
    baseline_results: list[dict],
    output_path: Path,
) -> None:
    """Compare PPO and first-legal baseline on shared seeds and zone metrics."""
    if not policy_results or len(policy_results) != len(baseline_results):
        raise ValueError("Policy and baseline evaluations must be non-empty and aligned.")
    if [result["seed"] for result in policy_results] != [
        result["seed"] for result in baseline_results
    ]:
        raise ValueError("Policy and baseline evaluation seeds must match.")

    indices = np.arange(1, len(policy_results) + 1)
    width = 0.38
    colors = {"PPO": "tab:blue", "First-legal baseline": "tab:orange"}

    def grouped_bars(axis, policy_values, baseline_values, title, ylabel):
        axis.bar(
            indices - width / 2, policy_values, width,
            label="PPO", color=colors["PPO"],
        )
        axis.bar(
            indices + width / 2, baseline_values, width,
            label="First-legal baseline", color=colors["First-legal baseline"],
        )
        axis.set_title(title)
        axis.set_xlabel("Evaluation episode (paired seeds)")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize="small")

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    grouped_bars(
        axes[0, 0],
        [result["total_reward"] for result in policy_results],
        [result["total_reward"] for result in baseline_results],
        "Episode reward",
        "Reward",
    )
    grouped_bars(
        axes[0, 1],
        [result["fraction_steps_inside_zone"] for result in policy_results],
        [result["fraction_steps_inside_zone"] for result in baseline_results],
        "Fraction of steps with |v| <= 1",
        "Fraction",
    )
    grouped_bars(
        axes[1, 0],
        [
            result["longest_consecutive_inside_simulation_time"]
            for result in policy_results
        ],
        [
            result["longest_consecutive_inside_simulation_time"]
            for result in baseline_results
        ],
        "Longest consecutive near-zero run",
        "Simulation time (s)",
    )
    grouped_bars(
        axes[1, 1],
        [int(result["final_inside_zone"]) for result in policy_results],
        [int(result["final_inside_zone"]) for result in baseline_results],
        "Finished inside |v| <= 1",
        "Inside zone (0/1)",
    )
    axes[1, 1].set_ylim(-0.05, 1.15)

    for label, results in (
        ("PPO", policy_results),
        ("First-legal baseline", baseline_results),
    ):
        path_counts = Counter(
            action for result in results for action in result["chosen_actions"]
        )
        path_ids = sorted(path_counts)
        axes[2, 0].plot(
            path_ids,
            [path_counts[path_id] for path_id in path_ids],
            marker="o",
            label=label,
        )
    axes[2, 0].set_title("Controller path selections")
    axes[2, 0].set_xlabel("Path ID")
    axes[2, 0].set_ylabel("Selections across episodes")
    axes[2, 0].legend()

    grouped_bars(
        axes[2, 1],
        [len(result["mode_transitions"]) for result in policy_results],
        [len(result["mode_transitions"]) for result in baseline_results],
        "Mode transitions per episode",
        "Transitions",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved cruise-control evaluation summary: {output_path}")


def _near_zero_zone_metrics(
    velocity_samples: list[float],
    simulation_times: list[float],
) -> dict[str, float | int | bool]:
    """Measure in-zone step occupancy and sampled consecutive simulation time."""
    if len(velocity_samples) != len(simulation_times) or not velocity_samples:
        raise ValueError("Velocity and simulation-time samples must be non-empty and aligned.")

    steps_inside_zone = 0
    inside_zone_simulation_time = 0.0
    longest_consecutive_inside_steps = 0
    longest_consecutive_inside_simulation_time = 0.0
    consecutive_inside_steps = 0
    consecutive_inside_simulation_time = 0.0
    previous_inside = abs(velocity_samples[0]) <= 1.0

    for index in range(1, len(velocity_samples)):
        inside = abs(velocity_samples[index]) <= 1.0
        if inside:
            steps_inside_zone += 1
            consecutive_inside_steps = (
                consecutive_inside_steps + 1 if previous_inside else 1
            )
            if previous_inside:
                elapsed = max(0.0, simulation_times[index] - simulation_times[index - 1])
                inside_zone_simulation_time += elapsed
                consecutive_inside_simulation_time += elapsed
            else:
                consecutive_inside_simulation_time = 0.0
            longest_consecutive_inside_steps = max(
                longest_consecutive_inside_steps, consecutive_inside_steps
            )
            longest_consecutive_inside_simulation_time = max(
                longest_consecutive_inside_simulation_time,
                consecutive_inside_simulation_time,
            )
        else:
            consecutive_inside_steps = 0
            consecutive_inside_simulation_time = 0.0
        previous_inside = inside

    episode_steps = len(velocity_samples) - 1
    return {
        "steps_inside_zone": steps_inside_zone,
        "fraction_steps_inside_zone": (
            steps_inside_zone / episode_steps if episode_steps else 0.0
        ),
        "inside_zone_simulation_time": inside_zone_simulation_time,
        "longest_consecutive_inside_steps": longest_consecutive_inside_steps,
        "longest_consecutive_inside_simulation_time": (
            longest_consecutive_inside_simulation_time
        ),
        "final_inside_zone": abs(velocity_samples[-1]) <= 1.0,
    }


def evaluate_policy(env, model, episodes: int, seed: int) -> list[dict]:
    results = []
    policy_name = "First-legal baseline" if model is None else "PPO policy"
    max_episode_steps = int(env.unwrapped.max_episode_steps)
    with tqdm(
        total=episodes * max_episode_steps,
        desc=f"{policy_name} evaluation",
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        for episode_index in range(episodes):
            observation, info = env.reset(seed=seed + episode_index)
            state_history = [dict(info["state"])]
            velocity_samples = [float(info["state"]["v"])]
            simulation_time_samples = [
                float(info.get("simulation_elapsed_time", 0.0))
            ]
            chosen_actions = []
            reward_history = []
            mode_transitions = []
            total_reward = 0.0
            terminated = truncated = False
            reached_near_zero_speed = abs(float(info["state"]["v"])) <= 1.0
            goal_step = 0 if reached_near_zero_speed else None
            goal_simulation_time = (
                float(info.get("simulation_elapsed_time", 0.0))
                if reached_near_zero_speed else None
            )
            progress.set_description(
                f"{policy_name} evaluation ({episode_index + 1}/{episodes})"
            )

            while not (terminated or truncated):
                mask = env.unwrapped.action_masks()
                if not mask.any():
                    raise RuntimeError(
                        "Cruise Control returned an empty action mask during evaluation."
                    )
                if model is None:
                    action = int(np.flatnonzero(mask)[0])
                else:
                    action, _ = model.predict(
                        observation, deterministic=True, action_masks=mask
                    )
                previous_mode = float(info["state"]["mode"])
                observation, reward, terminated, truncated, info = env.step(action)
                progress.update(1)
                total_reward += float(reward)
                reward_history.append(float(reward))
                chosen_actions.append(int(info["control_action"]))
                current_state = dict(info["state"])
                state_history.append(current_state)
                velocity_samples.append(float(current_state["v"]))
                simulation_time_samples.append(
                    float(info.get("simulation_elapsed_time", 0.0))
                )
                if float(current_state["mode"]) != previous_mode:
                    mode_transitions.append({
                        "step": len(chosen_actions),
                        "simulation_time": float(
                            info.get("simulation_elapsed_time", 0.0)
                        ),
                        "from_mode": previous_mode,
                        "to_mode": float(current_state["mode"]),
                        "path_id": int(info["control_action"]),
                    })
                if goal_step is None and abs(float(current_state["v"])) <= 1.0:
                    reached_near_zero_speed = True
                    goal_step = len(chosen_actions)
                    goal_simulation_time = float(
                        info.get("simulation_elapsed_time", 0.0)
                    )

            zone_metrics = _near_zero_zone_metrics(
                velocity_samples, simulation_time_samples
            )
            results.append({
                "episode": episode_index + 1,
                "seed": seed + episode_index,
                "total_reward": total_reward,
                "steps": len(chosen_actions),
                "reached_near_zero_speed": reached_near_zero_speed,
                "goal_step": goal_step,
                "goal_simulation_time": goal_simulation_time,
                **zone_metrics,
                "mode_transitions": mode_transitions,
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": info.get("termination_reason"),
                "chosen_actions": chosen_actions,
                "state_history": state_history,
                "simulation_time_samples": simulation_time_samples,
                "reward_history": reward_history,
                "initial_state": state_history[0],
                "final_state": state_history[-1],
            })
            progress.set_postfix(
                episode_reward=f"{total_reward:.1f}",
                near_zero=reached_near_zero_speed,
            )
    return results


def save_evaluation(results: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "episode",
            "seed",
            "total_reward",
            "steps",
            "reached_near_zero_speed",
            "goal_step",
            "goal_simulation_time",
            "steps_inside_zone",
            "fraction_steps_inside_zone",
            "inside_zone_simulation_time",
            "longest_consecutive_inside_steps",
            "longest_consecutive_inside_simulation_time",
            "final_inside_zone",
            "mode_transitions",
            "terminated",
            "truncated",
            "termination_reason",
            "chosen_actions",
            "initial_state",
            "final_state",
        ], extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = dict(result)
            for field in (
                "mode_transitions",
                "chosen_actions",
                "initial_state",
                "final_state",
            ):
                row[field] = json.dumps(row[field], sort_keys=True)
            writer.writerow(row)
    print(f"Saved evaluation episodes: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "test" / "input" / "withBound" / "CruiseControlStability.txt",
    )
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the saved policy against the first-legal baseline.",
    )
    parser.add_argument(
        "--plots-per-category",
        type=int,
        default=2,
        help="Maximum representative PPO episodes to plot per reachability category.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Maximum simulator steps per training and evaluation episode.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "logs" / "cruise_control")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--generated-env",
        type=Path,
        default=ROOT / "generated_cruise_control_env.py",
        help="Persistent path for the generated Gym environment module.",
    )
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("--timesteps must be at least 1")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be at least 1")
    if args.plots_per_category < 1:
        parser.error("--plots-per-category must be at least 1")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be at least 1")
    if not args.model.is_file():
        raise FileNotFoundError(f"Cruise Control model not found: {args.model}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.generated_env.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "ppo_cruise_control.zip"
    curve_path = args.out_dir / "cruise_control_training_curve.png"
    evaluation_path = args.out_dir / "cruise_control_evaluation.csv"

    env_class = build_env_class(args.model, args.generated_env)
    print(f"Saved generated environment: {args.generated_env}")

    def make_env():
        env = env_class(
            render_mode=None,
            config_override={
                "max_episode_steps": args.max_episode_steps,
                "initial_state_buffer_config": {"enabled": False},
            },
        )
        return Monitor(env)

    if args.eval_only:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Saved policy not found for evaluation: {model_path}"
            )
        model = MaskablePPO.load(str(model_path))
        print(
            f"Loaded saved policy for evaluation: {model_path}; "
            f"episode limit={args.max_episode_steps} steps."
        )
    else:
        train_env = DummyVecEnv([make_env])
        try:
            action_count = int(train_env.get_attr("action_space")[0].n)
            if action_count <= 1:
                raise RuntimeError(
                    "Generated Cruise Control environment has no meaningful "
                    f"controller choices (action space size: {action_count})."
                )
            print(
                f"Training MaskablePPO with {action_count} controlFunc paths "
                f"for {args.timesteps} timesteps; episode limit="
                f"{args.max_episode_steps} steps; seed={args.seed}."
            )

            rollout_steps = min(512, max(64, args.timesteps))
            model = MaskablePPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                ent_coef=0.01,
                n_steps=rollout_steps,
                batch_size=min(64, rollout_steps),
                seed=args.seed,
            )
            callback = EpisodeStatsCallback()
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback,
                progress_bar=True,
            )
            model.save(str(model_path))
            print(f"Saved policy: {model_path}")
            plot_training_curve(callback, curve_path)
        finally:
            train_env.close()

    eval_env = make_env()
    try:
        results = evaluate_policy(
            eval_env, model, args.eval_episodes, args.seed + 1000
        )
        baseline_results = evaluate_policy(
            eval_env, None, args.eval_episodes, args.seed + 1000
        )
        save_evaluation(results, evaluation_path)
        save_evaluation(
            baseline_results,
            args.out_dir / "cruise_control_first_legal_baseline.csv",
        )
    finally:
        eval_env.close()

    plot_evaluation_summary(
        results,
        baseline_results,
        args.out_dir / "cruise_control_evaluation_summary.png",
    )
    plot_representative_episodes(
        results, args.out_dir, max_per_category=args.plots_per_category
    )

    rewards = [result["total_reward"] for result in results]
    lengths = [result["steps"] for result in results]
    successes = sum(result["reached_near_zero_speed"] for result in results)
    action_counts = Counter(
        action for result in results for action in result["chosen_actions"]
    )
    print(f"Mean evaluation reward: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    baseline_rewards = [result["total_reward"] for result in baseline_results]
    print(
        "First-legal-action baseline: "
        f"{np.mean(baseline_rewards):.3f} +/- {np.std(baseline_rewards):.3f} "
        f"(policy delta {np.mean(rewards) - np.mean(baseline_rewards):+.3f})"
    )
    print(f"Mean evaluation length: {np.mean(lengths):.1f} steps")
    print(
        f"Reached |v| <= 1: {successes}/{len(results)} "
        f"({100.0 * successes / len(results):.1f}%)"
    )
    for label, evaluation_results in (
        ("PPO", results),
        ("First-legal baseline", baseline_results),
    ):
        print(
            f"{label} near-zero occupancy: "
            f"{np.mean([r['steps_inside_zone'] for r in evaluation_results]):.1f} "
            f"steps/episode "
            f"({100.0 * np.mean([r['fraction_steps_inside_zone'] for r in evaluation_results]):.2f}%); "
            f"mean total sampled in-zone time="
            f"{np.mean([r['inside_zone_simulation_time'] for r in evaluation_results]):.3f}s; "
            f"mean longest consecutive run="
            f"{np.mean([r['longest_consecutive_inside_steps'] for r in evaluation_results]):.1f} "
            f"steps / "
            f"{np.mean([r['longest_consecutive_inside_simulation_time'] for r in evaluation_results]):.3f}s; "
            f"finished inside="
            f"{sum(r['final_inside_zone'] for r in evaluation_results)}/{len(evaluation_results)}"
        )
    print(f"Selected controller paths: {dict(sorted(action_counts.items()))}")
    for result in results:
        transitions = [
            f"step {item['step']} (t={item['simulation_time']:.3f}): "
            f"{item['from_mode']:g}->{item['to_mode']:g} via path {item['path_id']}"
            for item in result["mode_transitions"]
        ]
        print(
            f"Episode {result['episode']}: reward={result['total_reward']:.3f}, "
            f"steps={result['steps']}, final_v={result['final_state']['v']:.4f}, "
            f"goal_step={result['goal_step']}, "
            f"inside_steps={result['steps_inside_zone']}, "
            f"longest_inside={result['longest_consecutive_inside_steps']} steps/"
            f"{result['longest_consecutive_inside_simulation_time']:.3f}s, "
            f"final_inside={result['final_inside_zone']}, "
            f"terminated={result['terminated']}, truncated={result['truncated']}, "
            f"reason={result['termination_reason']}, "
            f"mode_transitions={transitions}"
        )


if __name__ == "__main__":
    main()
