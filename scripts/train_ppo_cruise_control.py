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
        ])
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
