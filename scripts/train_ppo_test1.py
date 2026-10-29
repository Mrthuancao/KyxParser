"""Train and evaluate a masked PPO controller on the Test 1 robot model.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe scripts\\train_ppo_test1.py --timesteps 10000

The evaluation reports post-action ``ensure`` separately from the ``loopinv``
safety invariant. Test 1's initial condition already implies ``ensure``, so
that predicate alone is not a useful measure of learned safety.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
AST2 = ROOT / "AST_parser_2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AST2) not in sys.path:
    sys.path.insert(0, str(AST2))

from generator_core import generate_gym_environment_file  # pyright: ignore[reportMissingImports]
from ast_config import BinaryFormulaOp

# Zero-based ID of Test 1's third controlFunc alternative (a := A).
ACCELERATION_PATH_ID = 2
BASELINE_RADIUS = 512.0
BASELINE_ARC_LENGTH = 70.0


class EpisodeStatsCallback(BaseCallback):
    """Collect completed episode rewards and lengths from Monitor."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

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
        model_text, "KeYmaeraTest1", str(generated_path)
    ):
        raise RuntimeError(f"Could not generate an environment from {model_path}")

    spec = importlib.util.spec_from_file_location(
        "generated_test1_ppo_env", generated_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated environment: {generated_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.KeYmaeraGymEnv


def configure_initial_state_sampling(env) -> None:
    """Sample bounded positions and obstacle coordinates with a valid premise."""
    sampled_direction = {"angle": 0.0}
    env.initial_state_samplers.update({
        "v": lambda: 0.0,
        "a": lambda: 0.0,
        "dx": lambda: _sample_dx(env, sampled_direction),
        "dy": lambda: float(np.sin(sampled_direction["angle"])),
        "w": lambda: 0.0,
        "r": lambda: 10.0,
        "t": lambda: 0.0,
    })


def _sample_dx(env, sampled_direction: dict[str, float]) -> float:
    angle = float(env.np_random.uniform(-np.pi, np.pi))
    sampled_direction["angle"] = angle
    return float(np.cos(angle))


def _predicate(env, name: str, state: dict) -> bool:
    if name == "isWellformedDir":
        norm_squared = state["dx"] ** 2 + state["dy"] ** 2
        return math.isclose(norm_squared, 1.0, rel_tol=0.0, abs_tol=1e-6)
    if name == "loopinv":
        return all(
            _predicate(env, predicate_name, state)
            for predicate_name in (
                "isPhysicallyValid",
                "isWellformedDir",
                "isSafe",
                "isEfficient",
            )
        )
    function = env.unwrapped.user_defined_functions.get(name)
    if function is None:
        raise RuntimeError(f"Generated Test 1 environment has no '{name}' predicate")
    args = {
        "isPhysicallyValid": ("v", "r"),
        "isWellformedDir": ("dx", "dy"),
        "isSafe": ("x", "y", "v", "xo", "yo"),
        "isEfficient": ("v",),
        "ensure": ("x", "y", "xo", "yo"),
        "loopinv": ("x", "y", "v", "dx", "dy", "xo", "yo", "r"),
    }.get(name)
    if args is None:
        raise ValueError(f"No argument mapping is defined for '{name}'")
    try:
        return bool(function(*(state[key] for key in args)))
    except KeyError as exc:
        raise RuntimeError(
            f"Cannot evaluate Test 1 predicate '{name}': state is missing {exc}"
        ) from exc


def _remaining_safety_distance(env, state: dict) -> float:
    stop_dist = env.unwrapped.user_defined_functions.get("stopDist")
    if stop_dist is None:
        raise RuntimeError("Generated Test 1 environment has no 'stopDist' function")
    axis_clearance = max(
        abs(state["x"] - state["xo"]),
        abs(state["y"] - state["yo"]),
    )
    return float(axis_clearance - stop_dist(state["v"]))


def _configure_comparison_initial_state_sampling(env) -> tuple[float, float]:
    raw_env = env.unwrapped
    start_x, start_y = 10.0, 50.0
    theta = BASELINE_ARC_LENGTH / BASELINE_RADIUS
    obstacle_x = start_x + BASELINE_RADIUS * math.sin(theta)
    obstacle_y = start_y + BASELINE_RADIUS * (1.0 - math.cos(theta))
    raw_env.initial_state_samplers.update({
        "x": lambda: start_x,
        "y": lambda: start_y,
        "v": lambda: 0.0,
        "a": lambda: 0.0,
        "dx": lambda: 1.0,
        "dy": lambda: 0.0,
        "w": lambda: 0.0,
        "r": lambda: BASELINE_RADIUS,
        "xo": lambda: obstacle_x,
        "yo": lambda: obstacle_y,
        "t": lambda: 0.0,
    })
    return obstacle_x, obstacle_y


def _configure_forced_acceleration_baseline(
    env, obstacle: tuple[float, float]
) -> None:
    raw_env = env.unwrapped
    obstacle_x, obstacle_y = obstacle
    raw_env._isWellformedDir_impl = lambda dx, dy: math.isclose(
        dx * dx + dy * dy, 1.0, rel_tol=0.0, abs_tol=1e-6
    )
    raw_env.user_defined_functions["isWellformedDir"] = (
        raw_env._isWellformedDir_impl
    )

    def bypass_admissible_separation_guard(condition):
        if isinstance(condition, BinaryFormulaOp) and condition.op == "|":
            return True
        return None

    raw_env._controller_test_handler = bypass_admissible_separation_guard

    def sample_nondeterministic_assignment(var_name: str) -> float:
        if var_name == "r":
            return BASELINE_RADIUS
        if var_name == "w":
            return _curvature_rate_for_velocity(
                float(raw_env.current_state["v"]), BASELINE_RADIUS
            )
        if var_name == "xo":
            return obstacle_x
        if var_name == "yo":
            return obstacle_y
        raise ValueError(
            f"Unexpected Test 1 baseline nondeterministic variable: {var_name}"
        )

    raw_env._nondeterministic_assignment_handler = (
        sample_nondeterministic_assignment
    )


def _curvature_rate_for_velocity(velocity: float, radius: float) -> float:
    quotient = velocity / radius
    candidates = (
        quotient,
        math.nextafter(quotient, math.inf),
        math.nextafter(quotient, -math.inf),
    )
    for candidate in candidates:
        if radius * candidate == velocity:
            return candidate
    raise ArithmeticError(
        f"Could not represent w satisfying r*w=v for v={velocity}, r={radius}."
    )


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
            average = np.convolve(
                values, np.ones(window) / window, mode="valid"
            )
            axis.plot(
                range(window - 1, len(values)),
                average,
                label=f"Moving average (window={window})",
            )
        axis.set_ylabel(label)
        axis.legend()
    axes[1].set_xlabel("Completed episode")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved training curve: {output_path}")


def collect_episodes(
    env,
    model,
    count: int,
    seed: int,
    forced_action: int | None = None,
) -> list[dict]:
    episodes = []
    max_steps = int(env.unwrapped.max_episode_steps)
    description = (
        "PPO evaluation" if forced_action is None
        else "Acceleration baseline"
    )
    with tqdm(
        total=count,
        desc=description,
        unit="episode",
        dynamic_ncols=True,
    ) as episode_progress:
        for episode_index in range(count):
            observation, info = env.reset(seed=seed + episode_index)
            states = [dict(info["state"])]
            rewards: list[float] = []
            actions: list[dict] = []
            terminated = truncated = False
            total_reward = 0.0
            episode_steps = 0
            step_progress = tqdm(
                total=max_steps,
                desc=f"Episode {episode_index + 1}/{count}",
                unit="step",
                dynamic_ncols=True,
                leave=False,
                position=1,
                disable=forced_action is None,
            )
            try:
                while not (terminated or truncated):
                    if forced_action is None:
                        mask = env.unwrapped.action_masks()
                        if not mask.any():
                            raise RuntimeError(
                                "Test 1 returned an empty legal-action mask "
                                "during evaluation."
                            )
                        action, _ = model.predict(
                            observation,
                            deterministic=True,
                            action_masks=mask,
                        )
                    else:
                        action = int(forced_action)
                    observation, reward, terminated, truncated, info = env.step(action)
                    next_state = dict(info["state"])
                    path_id = info.get("control_action")
                    actions.append({
                        "path_id": path_id,
                        "policy_action": int(np.asarray(action).reshape(-1)[0]),
                        "control_flow_info": info.get("control_flow_info"),
                    })
                    states.append(next_state)
                    rewards.append(float(reward))
                    total_reward += float(reward)
                    episode_steps += 1
                    step_progress.update(1)
                    step_progress.set_postfix(
                        terminated=terminated,
                        truncated=truncated,
                        reason=info.get("termination_reason") or "running",
                    )
                    if episode_steps > max_steps:
                        raise RuntimeError(
                            "Test 1 exceeded max_episode_steps without "
                            "terminating or truncating."
                        )
            finally:
                step_progress.close()

            invariant_values = [
                _predicate(env, "loopinv", state) for state in states
            ]
            safety_distance_history = [
                _remaining_safety_distance(env, state) for state in states
            ]
            initial_ensure = _predicate(env, "ensure", states[0])
            success = any(
                _predicate(env, "ensure", state) for state in states[1:]
            )
            termination_reason = info.get("termination_reason")
            if termination_reason is None:
                termination_reason = (
                    "time_limit_truncation"
                    if truncated else "stopped_without_reason"
                )
            episodes.append({
                "episode": episode_index + 1,
                "seed": seed + episode_index,
                "state_history": states,
                "remaining_safety_distance_history": safety_distance_history,
                "reward_history": rewards,
                "chosen_actions": actions,
                "success": success,
                "initial_ensure": initial_ensure,
                "initial_invariant": invariant_values[0],
                "invariant_violated": not all(invariant_values),
                "generated_safety_termination": termination_reason in {
                    "model_loop_invariant_violation",
                    "model_safety_postcondition_violation",
                },
                "total_reward": total_reward,
                "steps": len(actions),
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": termination_reason,
                "final_state": states[-1],
            })
            episode_progress.update(1)
            episode_progress.set_postfix(
                steps=len(actions),
                reason=termination_reason,
            )
    return episodes


def plot_episode(episode: dict, output_path: Path) -> None:
    states = episode["state_history"]
    steps = np.arange(len(states))
    actions = episode["chosen_actions"]
    trajectory_path = output_path.with_name(
        f"{output_path.stem}_trajectory{output_path.suffix}"
    )
    trajectory_fig, trajectory_axis = plt.subplots(figsize=(7, 7))
    trajectory_axis.plot(
        [state["x"] for state in states],
        [state["y"] for state in states],
        marker=".",
        label="Robot trajectory",
    )
    trajectory_axis.scatter(
        [states[0]["xo"]], [states[0]["yo"]],
        marker="x", s=70, color="tab:red", label="Obstacle",
    )
    trajectory_axis.set_xlabel("x")
    trajectory_axis.set_ylabel("y")
    trajectory_axis.set_title(
        f"Episode {episode['episode']}: planar robot path"
    )
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.legend()
    trajectory_fig.tight_layout()
    trajectory_fig.savefig(trajectory_path, dpi=160)
    plt.close(trajectory_fig)
    print(f"Saved Test 1 trajectory plot: {trajectory_path}")

    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(steps, [state["v"] for state in states], label="v")
    axes[0].plot(steps, [state["a"] for state in states], label="a")
    axes[0].set_ylabel("Velocity / acceleration")
    axes[0].legend()

    axes[1].plot(
        steps,
        episode["remaining_safety_distance_history"],
        color="tab:red",
        label="Remaining safety distance",
    )
    axes[1].axhline(
        0.0, color="black", linestyle="--", linewidth=1,
        label="Safety boundary",
    )
    axes[1].set_ylabel("Distance margin")
    axes[1].set_title("max(|x-xo|, |y-yo|) - stopDist(v)")
    axes[1].legend()

    axes[2].plot(steps, [state["dx"] for state in states], label="dx")
    axes[2].plot(steps, [state["dy"] for state in states], label="dy")
    axes[2].set_ylabel("Direction")
    axes[2].legend()

    axes[3].plot(steps, [state["w"] for state in states], label="w")
    axes[3].plot(steps, [state["r"] for state in states], label="r")
    axes[3].set_ylabel("Turn rate / radius")
    axes[3].legend()

    axes[4].step(
        np.arange(1, len(actions) + 1),
        [action["path_id"] for action in actions],
        where="post",
    )
    axes[4].set_xlabel("Control step")
    axes[4].set_ylabel("Path ID")
    axes[4].set_title("Selected controlFunc branch")
    fig.suptitle(
        f"Episode {episode['episode']}: post-action ensure={episode['success']}, "
        f"loopinv violation={episode['invariant_violated']}, "
        f"generated safety termination="
        f"{episode['generated_safety_termination']}, "
        f"reason={episode['termination_reason']}"
    )
    fig.tight_layout()
    signals_path = output_path.with_name(
        f"{output_path.stem}_signals{output_path.suffix}"
    )
    fig.savefig(signals_path, dpi=160)
    plt.close(fig)
    print(f"Saved Test 1 state plot: {signals_path}")


def save_evaluation(episodes: list[dict], output_dir: Path) -> None:
    if not episodes:
        raise ValueError("Evaluation requires at least one episode.")

    rewards = [episode["total_reward"] for episode in episodes]
    successes = [episode["success"] for episode in episodes]
    invariant_passes = [
        not episode["invariant_violated"] for episode in episodes
    ]
    action_counts = Counter(
        action["path_id"]
        for episode in episodes
        for action in episode["chosen_actions"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = [
        "tab:green" if safe else "tab:red"
        for safe in invariant_passes
    ]
    indices = np.arange(1, len(episodes) + 1)
    axes[0, 0].bar(indices, rewards, color=colors)
    axes[0, 0].axhline(
        np.mean(rewards), color="black", linestyle="--",
        label=f"Mean reward={np.mean(rewards):.2f}",
    )
    axes[0, 0].set_title("Evaluation reward (green=loop invariant held)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()

    axes[0, 1].plot(
        indices, np.cumsum(successes) / indices, marker="o",
        label="post-action ensure predicate",
    )
    axes[0, 1].plot(
        indices, np.cumsum(invariant_passes) / indices, marker="s",
        label="loop invariant throughout",
    )
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_title("Cumulative success and safety rates")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].legend()

    axes[1, 0].bar(indices, [episode["steps"] for episode in episodes])
    axes[1, 0].set_title("Evaluation episode length")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Control steps")

    path_ids = sorted(action_counts)
    axes[1, 1].bar(
        [str(path_id) for path_id in path_ids],
        [action_counts[path_id] for path_id in path_ids],
    )
    axes[1, 1].set_title("Controller path selections")
    axes[1, 1].set_xlabel("Zero-based path ID")
    axes[1, 1].set_ylabel("Selections")
    fig.tight_layout()
    plot_path = output_dir / "test1_evaluation_summary.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved Test 1 evaluation plot: {plot_path}")

    csv_path = output_dir / "test1_evaluation.csv"
    fields = (
        "episode",
        "seed",
        "initial_state",
        "initial_ensure",
        "success",
        "initial_invariant",
        "invariant_violated",
        "generated_safety_termination",
        "minimum_remaining_safety_distance", "termination_reason", "terminated",
        "truncated", "steps", "total_reward", "chosen_actions", "final_state",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            row = {
                key: episode[key]
                for key in fields
                if key not in {
                    "initial_state",
                    "minimum_remaining_safety_distance",
                }
            }
            row["minimum_remaining_safety_distance"] = min(
                episode["remaining_safety_distance_history"]
            )
            row["initial_state"] = json.dumps(
                episode["state_history"][0], sort_keys=True
            )
            row["chosen_actions"] = json.dumps(row["chosen_actions"])
            row["final_state"] = json.dumps(row["final_state"], sort_keys=True)
            writer.writerow(row)
    print(f"Saved Test 1 evaluation data: {csv_path}")

    reasons = Counter(episode["termination_reason"] for episode in episodes)
    print(f"Mean reward: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    print(
        f"Post-action ensure predicate: {sum(successes)}/{len(episodes)} "
        f"({100 * np.mean(successes):.1f}%)"
    )
    print(
        f"loop invariant held throughout: {sum(invariant_passes)}/{len(episodes)} "
        f"({100 * np.mean(invariant_passes):.1f}%)"
    )
    print(f"Controller path counts: {dict(action_counts)}")
    print(f"Termination reasons: {dict(reasons)}")


def save_policy_comparison(
    policy_episodes: list[dict],
    baseline_episodes: list[dict],
    output_dir: Path,
) -> None:
    if not policy_episodes or not baseline_episodes:
        raise ValueError("Policy comparison requires episodes for both policies.")

    groups = (
        ("PPO", policy_episodes),
        ("Always acceleration (guard bypass)", baseline_episodes),
    )
    metrics = (
        ("loop invariant held", lambda e: not e["invariant_violated"]),
        ("post-action ensure", lambda e: e["success"]),
        (
            "generated safety termination",
            lambda e: e["generated_safety_termination"],
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for axis, (metric_name, metric) in zip(axes, metrics):
        rates = [
            np.mean([metric(episode) for episode in episodes])
            for _, episodes in groups
        ]
        axis.bar([name for name, _ in groups], rates, color=["tab:blue", "tab:red"])
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Episode rate")
        axis.set_title(metric_name)
        axis.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Test 1: PPO vs. forced-acceleration monitor stress test")
    fig.tight_layout()
    plot_path = output_dir / "test1_policy_vs_acceleration_baseline.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved PPO/baseline comparison plot: {plot_path}")

    csv_path = output_dir / "test1_policy_vs_acceleration_baseline.csv"
    fields = (
        "policy",
        "episode",
        "seed",
        "initial_state",
        "success",
        "invariant_violated",
        "generated_safety_termination",
        "minimum_remaining_safety_distance",
        "steps", "total_reward", "termination_reason", "chosen_actions",
        "final_state",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for policy_name, episodes in groups:
            for episode in episodes:
                writer.writerow({
                    "policy": policy_name,
                    "episode": episode["episode"],
                    "seed": episode["seed"],
                    "initial_state": json.dumps(
                        episode["state_history"][0], sort_keys=True
                    ),
                    "success": episode["success"],
                    "invariant_violated": episode["invariant_violated"],
                    "generated_safety_termination": (
                        episode["generated_safety_termination"]
                    ),
                    "minimum_remaining_safety_distance": min(
                        episode["remaining_safety_distance_history"]
                    ),
                    "steps": episode["steps"],
                    "total_reward": episode["total_reward"],
                    "termination_reason": episode["termination_reason"],
                    "chosen_actions": json.dumps(episode["chosen_actions"]),
                    "final_state": json.dumps(episode["final_state"], sort_keys=True),
                })
    print(f"Saved PPO/baseline comparison data: {csv_path}")

    for policy_name, episodes in groups:
        invariant_failures = sum(
            episode["invariant_violated"] for episode in episodes
        )
        safety_terminations = sum(
            episode["generated_safety_termination"] for episode in episodes
        )
        termination_reasons = Counter(
            episode["termination_reason"] for episode in episodes
        )
        print(
            f"{policy_name}: loop invariant violated in "
            f"{invariant_failures}/{len(episodes)} episodes; "
            f"generated safety terminations="
            f"{safety_terminations}/{len(episodes)}; "
            f"mean steps={np.mean([episode['steps'] for episode in episodes]):.1f}; "
            f"termination reasons={dict(termination_reasons)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "test" / "input" / "withBound" / "test1.txt",
    )
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ode-solver-step-size", type=float, default=0.02)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "logs" / "test1"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models" / "test1"
    )
    parser.add_argument(
        "--generated-env",
        type=Path,
        default=ROOT / "generated_test1_ppo_env.py",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate an existing policy instead of training a new one.",
    )
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be positive")
    if args.ode_solver_step_size <= 0:
        parser.error("--ode-solver-step-size must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Test 1 model not found: {args.model}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.generated_env.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "ppo_test1.zip"
    EnvClass = build_env_class(args.model, args.generated_env)
    print(f"Generated Test 1 environment: {args.generated_env}")

    def make_env():
        env = EnvClass(
            render_mode=None,
            config_override={
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
                "invalid_action_mode": "mask",
                "ode_solver_step_size": args.ode_solver_step_size,
                "initial_state_buffer_config": {"enabled": False},
            },
        )
        configure_initial_state_sampling(env)
        return Monitor(env)

    if args.eval_only:
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved Test 1 PPO model not found: {model_path}")
        model = MaskablePPO.load(str(model_path))
        print(f"Loaded policy for evaluation: {model_path}")
    else:
        train_env = DummyVecEnv([make_env])
        try:
            action_count = int(train_env.get_attr("action_space")[0].n)
            if action_count <= 1:
                raise RuntimeError(
                    "Test 1 has no meaningful discrete controller choices."
                )
            callback = EpisodeStatsCallback()
            model = MaskablePPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                ent_coef=0.01,
                seed=args.seed,
            )
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback,
                progress_bar=True,
            )
            model.save(str(model_path))
            print(f"Saved PPO policy: {model_path}")
            plot_training_curve(
                callback, args.out_dir / "test1_training_curve.png"
            )
        finally:
            train_env.close()

    eval_env = make_env()
    try:
        episodes = collect_episodes(
            eval_env, model, args.eval_episodes, args.seed + 1000
        )
        obstacle = _configure_comparison_initial_state_sampling(eval_env)
        action_count = int(eval_env.unwrapped.action_space.n)
        if ACCELERATION_PATH_ID >= action_count:
            raise RuntimeError(
                f"Test 1 has {action_count} controller paths, but the "
                f"acceleration baseline expects path {ACCELERATION_PATH_ID}."
            )
        _configure_forced_acceleration_baseline(eval_env, obstacle)
        acceleration_baseline_episodes = collect_episodes(
            eval_env,
            None,
            args.eval_episodes,
            args.seed + 1000,
            forced_action=ACCELERATION_PATH_ID,
        )
    finally:
        eval_env.close()

    for policy_name, policy_episodes in (
        ("ppo", episodes),
        ("acceleration_baseline", acceleration_baseline_episodes),
    ):
        for category, selected in (
            (
                "invariant_pass",
                [e for e in policy_episodes if not e["invariant_violated"]],
            ),
            (
                "invariant_violation",
                [e for e in policy_episodes if e["invariant_violated"]],
            ),
        ):
            for index, episode in enumerate(selected[:5], start=1):
                plot_episode(
                    episode,
                    args.out_dir
                    / f"test1_{policy_name}_{category}_{index}_seed_{episode['seed']}.png",
                )
    save_evaluation(episodes, args.out_dir)
    save_policy_comparison(
        episodes, acceleration_baseline_episodes, args.out_dir
    )


if __name__ == "__main__":
    main()
