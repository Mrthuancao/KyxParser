"""Train and evaluate PPO on the cruise-control model centered at velocity 25.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe scripts\\train_ppo_cruise_control_around25.py --timesteps 10000

The target band defaults to the model's ``nearTargetSpeed`` predicate:
``abs(v - 25) <= 1``. The generated simulator run is empirical, not a proof of
the quantified dL stability theorem.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_ppo_cruise_control import (  # pyright: ignore[reportMissingImports]
    EpisodeStatsCallback,
    MaskablePPO,
    build_env_class,
    evaluate_policy,
    plot_evaluation_summary,
    plot_representative_episodes,
    plot_training_curve,
    save_evaluation,
)


def _cruise_safe_on_state(env, state: dict) -> bool:
    """Evaluate the model's zero-argument safety predicate on one saved state."""
    raw_env = env.unwrapped
    predicate = raw_env.user_defined_functions.get("cruiseSafe")
    if predicate is None:
        raise RuntimeError(
            "Generated around-25 environment has no cruiseSafe predicate."
        )
    previous_state = raw_env.current_state
    raw_env.current_state = state
    try:
        return bool(predicate())
    finally:
        raw_env.current_state = previous_state


def _add_safety_metrics(env, results: list[dict]) -> None:
    for result in results:
        safety_samples = [
            _cruise_safe_on_state(env, state)
            for state in result["state_history"]
        ]
        result["cruise_safe_throughout"] = all(safety_samples)
        result["cruise_safe_failure_samples"] = sum(
            not is_safe for is_safe in safety_samples
        )


def _save_safety_report(
    policy_results: list[dict],
    baseline_results: list[dict],
    output_dir: Path,
) -> None:
    rows = []
    for policy_name, results in (
        ("PPO", policy_results),
        ("First-legal baseline", baseline_results),
    ):
        for result in results:
            rows.append({
                "policy": policy_name,
                "episode": result["episode"],
                "seed": result["seed"],
                "target_band_reached": result["reached_target_band"],
                "fraction_steps_in_target_band": (
                    result["fraction_steps_inside_zone"]
                ),
                "longest_consecutive_target_band_seconds": (
                    result["longest_consecutive_inside_simulation_time"]
                ),
                "finished_in_target_band": result["final_inside_zone"],
                "cruise_safe_throughout": result["cruise_safe_throughout"],
                "cruise_safe_failure_samples": (
                    result["cruise_safe_failure_samples"]
                ),
                "termination_reason": result["termination_reason"],
            })

    csv_path = output_dir / "cruise_control_around25_safety.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved around-25 safety metrics: {csv_path}")

    labels = ["PPO", "First-legal baseline"]
    safety_rates = [
        np.mean([result["cruise_safe_throughout"] for result in results])
        for results in (policy_results, baseline_results)
    ]
    target_occupancy = [
        np.mean([result["fraction_steps_inside_zone"] for result in results])
        for results in (policy_results, baseline_results)
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        x - width / 2,
        safety_rates,
        width,
        label="Cruise safety invariant held throughout",
    )
    axis.bar(
        x + width / 2,
        target_occupancy,
        width,
        label="Fraction of steps in target band",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_title("Around-25 target tracking and safety")
    axis.legend(loc="best")
    fig.tight_layout()
    plot_path = output_dir / "cruise_control_around25_safety.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved around-25 safety plot: {plot_path}")

    for label, results in zip(labels, (policy_results, baseline_results)):
        safe_count = sum(result["cruise_safe_throughout"] for result in results)
        failure_samples = sum(
            result["cruise_safe_failure_samples"] for result in results
        )
        print(
            f"{label} cruiseSafe held throughout: "
            f"{safe_count}/{len(results)} episodes; "
            f"unsafe sampled states={failure_samples}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            ROOT / "test" / "input" / "withBound"
            / "CruiseControlStabilityAround25.txt"
        ),
    )
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--target-speed-min", type=float, default=24.0)
    parser.add_argument("--target-speed-max", type=float, default=26.0)
    parser.add_argument("--plots-per-category", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "logs" / "cruise_control_around25",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "cruise_control_around25",
    )
    parser.add_argument(
        "--generated-env",
        type=Path,
        default=ROOT / "generated_cruise_control_around25_env.py",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate a previously trained policy without training.",
    )
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be positive")
    if args.plots_per_category < 1:
        parser.error("--plots-per-category must be positive")
    if args.target_speed_min > args.target_speed_max:
        parser.error("--target-speed-min must not exceed --target-speed-max")
    if not args.model.is_file():
        raise FileNotFoundError(f"Around-25 model not found: {args.model}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.generated_env.parent.mkdir(parents=True, exist_ok=True)
    policy_path = args.model_dir / "ppo_cruise_control_around25.zip"
    env_class = build_env_class(args.model, args.generated_env)
    print(f"Generated environment: {args.generated_env}")

    def make_env():
        return Monitor(env_class(
            render_mode=None,
            config_override={
                "max_episode_steps": args.max_episode_steps,
                "initial_state_buffer_config": {"enabled": False},
            },
        ))

    # Inspect representative reset masks so a degenerate action interface is
    # visible before spending time training.
    probe_env = make_env()
    legal_action_counts = []
    try:
        for episode_index in range(10):
            probe_env.reset(seed=args.seed + episode_index)
            legal_action_counts.append(
                int(probe_env.unwrapped.action_masks().sum())
            )
        action_count = int(probe_env.action_space.n)
    finally:
        probe_env.close()
    print(
        f"Controller action space: Discrete({action_count}); "
        f"legal choices on sampled resets: min={min(legal_action_counts)}, "
        f"max={max(legal_action_counts)}, mean={np.mean(legal_action_counts):.2f}"
    )
    if max(legal_action_counts) <= 1:
        print(
            "WARNING: sampled around-25 resets expose at most one legal "
            "controller path. PPO can run, but has little or no controller "
            "choice to learn in this initial-speed range."
        )

    if args.eval_only:
        if not policy_path.is_file():
            raise FileNotFoundError(f"Saved PPO policy not found: {policy_path}")
        model = MaskablePPO.load(str(policy_path))
        print(f"Loaded policy: {policy_path}")
    else:
        train_env = DummyVecEnv([make_env])
        try:
            model = MaskablePPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                ent_coef=0.01,
                seed=args.seed,
            )
            callback = EpisodeStatsCallback()
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback,
                progress_bar=True,
            )
            model.save(str(policy_path))
            print(f"Saved PPO policy: {policy_path}")
            plot_training_curve(
                callback, args.out_dir / "cruise_control_around25_training_curve.png"
            )
        finally:
            train_env.close()

    eval_env = make_env()
    try:
        policy_results = evaluate_policy(
            eval_env,
            model,
            args.eval_episodes,
            args.seed + 1000,
            args.target_speed_min,
            args.target_speed_max,
        )
        baseline_results = evaluate_policy(
            eval_env,
            None,
            args.eval_episodes,
            args.seed + 1000,
            args.target_speed_min,
            args.target_speed_max,
        )
        _add_safety_metrics(eval_env, policy_results)
        _add_safety_metrics(eval_env, baseline_results)
        save_evaluation(
            policy_results,
            args.out_dir / "cruise_control_around25_evaluation.csv",
        )
        save_evaluation(
            baseline_results,
            args.out_dir / "cruise_control_around25_first_legal_baseline.csv",
        )
    finally:
        eval_env.close()

    plot_evaluation_summary(
        policy_results,
        baseline_results,
        args.out_dir / "cruise_control_around25_evaluation_summary.png",
        args.target_speed_min,
        args.target_speed_max,
    )
    plot_representative_episodes(
        policy_results,
        args.out_dir,
        args.target_speed_min,
        args.target_speed_max,
        max_per_category=args.plots_per_category,
    )
    _save_safety_report(policy_results, baseline_results, args.out_dir)

    print(
        "Note: these sampled simulator results are not a proof of the "
        "quantified stability theorem."
    )


if __name__ == "__main__":
    main()
