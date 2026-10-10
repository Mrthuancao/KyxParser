"""Train MaskablePPO on the fixed-authority train-approach Gymnasium task.

The policy chooses between the formal controller's two guarded branches:
0 = brake, 1 = accelerate. Training and masked validation exclude branches
whose guards fail; a separate unmasked evaluation checks reliance on masking.
Success means stopping in
[m - delta, m] at speed no greater than d.

Run from the repository root:
    python scripts/train_ppo_train_essential.py
    python scripts/train_ppo_train_essential.py --timesteps 50000
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

proj_root = Path(__file__).resolve().parents[1]
ast2 = proj_root / "AST_parser_2"
if str(ast2) not in sys.path:
    sys.path.insert(0, str(ast2))
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from generator_core import generate_gym_environment_file


class EpisodeStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode:
                self.episode_rewards.append(episode["r"])
                self.episode_lengths.append(episode["l"])
        return True


def build_env_class(model_path: Path, generated_path: Path):
    source = model_path.read_text(encoding="utf-8")
    if not generate_gym_environment_file(source, "KeYmaeraTrainEssential", str(generated_path)):
        raise RuntimeError(f"Environment generation failed for {model_path}")

    spec = importlib.util.spec_from_file_location("generated_train_essential", generated_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generated environment module: {generated_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KeYmaeraGymEnv


def is_successful_arrival(
    state: dict,
    movement_authority: float,
    arrival_zone_width: float,
    arrival_speed: float,
) -> bool:
    position = state.get("z", float("-inf"))
    velocity = state.get("v", float("inf"))
    return (
        movement_authority - arrival_zone_width <= position <= movement_authority
        and 0.0 <= velocity <= arrival_speed
    )


def evaluate_policy(
    model,
    env,
    n_episodes: int,
    movement_authority: float,
    arrival_zone_width: float,
    arrival_speed: float,
    use_action_mask: bool,
) -> list:
    """Run complete deterministic episodes and retain trajectories/reasons for plots."""
    episodes = []
    try:
        for episode_index in range(n_episodes):
            observation, reset_info = env.reset(seed=episode_index)
            state_history = [dict(reset_info.get("state", {}))]
            reward_history = []
            rejected_actions = 0
            terminated = truncated = False

            while not (terminated or truncated):
                predict_kwargs = {}
                if use_action_mask:
                    predict_kwargs["action_masks"] = env.action_masks()
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                    **predict_kwargs,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                reward_history.append(float(reward))
                state_history.append(dict(info.get("state", {})))
                rejected_actions += int(
                    "rejected_action" in info
                    or info.get("control_flow_info") == "test_condition_failed"
                )

            final_state = state_history[-1] if state_history else {}
            success = is_successful_arrival(
                final_state,
                movement_authority,
                arrival_zone_width,
                arrival_speed,
            )
            if terminated and success:
                outcome = "success"
            elif truncated:
                outcome = "timeout"
            else:
                outcome = "failure"

            episodes.append({
                "state_history": state_history,
                "reward_history": reward_history,
                "total_reward": float(sum(reward_history)),
                "steps": len(reward_history),
                "terminated": terminated,
                "truncated": truncated,
                "success": bool(terminated and success),
                "outcome": outcome,
                "termination_reason": (
                    info.get("control_flow_info", "goal_or_safety_condition")
                    if terminated else "time_limit" if truncated else "unknown"
                ),
                "rejected_actions": rejected_actions,
            })
    finally:
        env.close()
    return episodes


def print_evaluation_metrics(mode: str, episodes: list):
    returns = [episode["total_reward"] for episode in episodes]
    lengths = [episode["steps"] for episode in episodes]
    successes = sum(episode["success"] for episode in episodes)
    failures = sum(episode["outcome"] == "failure" for episode in episodes)
    timeouts = sum(episode["outcome"] == "timeout" for episode in episodes)
    rejected_actions = sum(episode["rejected_actions"] for episode in episodes)
    print(f"\n=== {mode.title()} evaluation ({len(episodes)} episodes) ===")
    print(f"Mean return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"Mean length: {np.mean(lengths):.1f} steps")
    print(f"Successful safe arrivals: {successes}/{len(episodes)}")
    print(f"Failed terminations: {failures}/{len(episodes)}")
    print(f"Time-limit truncations: {timeouts}/{len(episodes)}")
    print(f"Rejected branch requests: {rejected_actions}")
    if mode == "unmasked":
        reasons = {}
        for episode in episodes:
            if episode["outcome"] == "failure":
                reason = episode["termination_reason"]
                reasons[reason] = reasons.get(reason, 0) + 1
        if reasons:
            print(f"Unmasked failure reasons: {reasons}")


def plot_episode(
    episode: dict,
    mode: str,
    index: int,
    out_dir: Path,
    movement_authority: float,
    arrival_zone_width: float,
    arrival_speed: float,
):
    states = episode["state_history"]
    rewards = episode["reward_history"]
    positions = [state.get("z", np.nan) for state in states]
    velocities = [state.get("v", np.nan) for state in states]
    accelerations = [state.get("a", np.nan) for state in states]

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(positions, color="tab:blue", label="Position z")
    axes[0].axhline(
        movement_authority, color="tab:red", linestyle="--", label="Authority m"
    )
    axes[0].axhline(
        movement_authority - arrival_zone_width,
        color="tab:green",
        linestyle=":",
        label="Arrival zone start",
    )
    axes[0].set_ylabel("Position")
    axes[0].legend(loc="best")
    axes[1].plot(velocities, color="tab:purple")
    axes[1].axhline(
        arrival_speed, color="tab:green", linestyle=":", label="Arrival speed limit"
    )
    axes[1].set_ylabel("Velocity")
    axes[1].legend(loc="best")
    axes[2].step(range(len(accelerations)), accelerations, where="post", color="tab:orange")
    axes[2].set_ylabel("Acceleration")
    axes[3].plot(rewards, color="tab:cyan")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Reward")
    fig.suptitle(
        f"{mode.title()} {episode['outcome']} episode {index + 1} "
        f"(return={episode['total_reward']:.1f}, "
        f"rejected={episode['rejected_actions']}, "
        f"reason={episode['termination_reason']})"
    )
    fig.tight_layout()
    path = out_dir / f"train_essential_fixed_authority_{mode}_{episode['outcome']}_{index + 1}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {mode} {episode['outcome']} episode plot: {path}")


def plot_sample_episodes(
    episodes: list,
    mode: str,
    max_per_outcome: int,
    out_dir: Path,
    movement_authority: float,
    arrival_zone_width: float,
    arrival_speed: float,
):
    outcomes = {
        "success": [episode for episode in episodes if episode["outcome"] == "success"],
        "failure": [episode for episode in episodes if episode["outcome"] == "failure"],
        "timeout": [episode for episode in episodes if episode["outcome"] == "timeout"],
    }
    for outcome, selected in outcomes.items():
        for index, episode in enumerate(selected[:max_per_outcome]):
            plot_episode(
                episode,
                mode,
                index,
                out_dir,
                movement_authority,
                arrival_zone_width,
                arrival_speed,
            )
    if not outcomes["success"]:
        print(f"No {mode} successful episodes to plot.")
    if not outcomes["failure"]:
        print(f"No {mode} failed episodes to plot.")
    if not outcomes["timeout"]:
        print(f"No {mode} timed-out episodes to plot.")


def plot_validation_summary(episodes_by_mode: dict, out_path: Path):
    modes = list(episodes_by_mode)
    outcomes = ("success", "failure", "timeout")
    rates = {
        mode: [
            sum(episode["success"] for episode in episodes) / len(episodes),
            sum(episode["outcome"] == "failure" for episode in episodes) / len(episodes),
            sum(episode["outcome"] == "timeout" for episode in episodes) / len(episodes),
        ]
        for mode, episodes in episodes_by_mode.items()
    }

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    positions = np.arange(len(outcomes))
    width = 0.35
    for index, mode in enumerate(modes):
        offset = (index - (len(modes) - 1) / 2) * width
        axes[0].bar(
            positions + offset,
            rates[mode],
            width,
            label=mode.title(),
        )
    axes[0].set_xticks(positions, [label.title() for label in outcomes])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Episode fraction")
    axes[0].set_title("Validation outcomes: masked vs unmasked policy")
    axes[0].legend()

    for mode, episodes in episodes_by_mode.items():
        returns = [episode["total_reward"] for episode in episodes]
        axes[1].bar(
            mode.title(),
            np.mean(returns),
            yerr=np.std(returns),
            capsize=5,
            label=mode.title(),
        )
    axes[1].set_ylabel("Mean episode return")
    axes[1].set_title("Validation return (mean ± standard deviation)")

    intervention_rates = []
    for mode in modes:
        episodes = episodes_by_mode[mode]
        total_steps = sum(episode["steps"] for episode in episodes)
        total_rejections = sum(episode["rejected_actions"] for episode in episodes)
        intervention_rates.append(
            100.0 * total_rejections / total_steps if total_steps else 0.0
        )
    axes[2].bar([mode.title() for mode in modes], intervention_rates, color="tab:orange")
    axes[2].set_ylabel("Interventions per 100 steps")
    axes[2].set_title("Illegal branch requests during unmasked evaluation")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved masked/unmasked validation plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dl-path",
        type=Path,
        default=proj_root / "test" / "input" / "withBound" / "train_essential_fixed_authority.kyx",
    )
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--plot-episodes", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=proj_root / "logs" / "train_essential_fixed_authority",
    )
    parser.add_argument("--model-dir", type=Path, default=proj_root / "models")
    args = parser.parse_args()

    if (args.timesteps < 2 or args.eval_episodes <= 0 or args.max_episode_steps <= 0
            or args.plot_episodes <= 0):
        parser.error(
            "timesteps must be at least 2; episode counts, plot counts, and lengths "
            "must be positive"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    generated_path = args.model_dir / "generated_train_essential_fixed_authority_env.py"
    env_class = build_env_class(args.dl_path, generated_path)

    movement_authority = 10.0
    arrival_speed = 0.2
    arrival_zone_width = 0.5

    def make_env():
        env = env_class(
            config_override={
                "max_episode_steps": args.max_episode_steps,
                "ode_solver_step_size": 0.02,
                "initial_state_buffer_config": {"enabled": False},
            }
        )
        if getattr(env.action_space, "n", None) != 2:
            raise RuntimeError(
                f"Expected two discrete controller choices (brake/accelerate), got "
                f"{env.action_space}"
            )
        if not env.differential_equations or not env.config["problem"]["loop_invariants"]:
            raise RuntimeError("Generated environment is missing train dynamics or loop invariant")
        return env

    train_env = DummyVecEnv([lambda: Monitor(make_env())])
    rollout_steps = min(2048, args.timesteps)
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        ent_coef=0.01,
        n_steps=rollout_steps,
        batch_size=min(64, rollout_steps),
    )
    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model_path = args.model_dir / "ppo_train_essential_fixed_authority_masked.zip"
    model.save(str(model_path))
    print(f"Saved MaskablePPO model: {model_path}")

    if callback.episode_rewards:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(callback.episode_rewards, alpha=0.35, label="Episode reward")
        axes[0].set_ylabel("Total reward")
        axes[0].legend()
        axes[1].plot(callback.episode_lengths, alpha=0.6, color="tab:purple")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        fig.suptitle("MaskablePPO training — fixed-authority train approach")
        fig.tight_layout()
        plot_path = args.out_dir / "train_essential_fixed_authority_training.png"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved training plot: {plot_path}")
    else:
        print("No episodes completed during training; skipping training plot.")

    masked_episodes = evaluate_policy(
        model, make_env(), args.eval_episodes,
        movement_authority, arrival_zone_width, arrival_speed,
        use_action_mask=True,
    )
    unmasked_episodes = evaluate_policy(
        model, make_env(), args.eval_episodes,
        movement_authority, arrival_zone_width, arrival_speed,
        use_action_mask=False,
    )
    all_episodes = {
        "masked": masked_episodes,
        "unmasked": unmasked_episodes,
    }
    for mode, episodes in all_episodes.items():
        print_evaluation_metrics(mode, episodes)
        plot_sample_episodes(
            episodes,
            mode,
            args.plot_episodes,
            args.out_dir,
            movement_authority,
            arrival_zone_width,
            arrival_speed,
        )
    plot_validation_summary(
        all_episodes,
        args.out_dir / "train_essential_fixed_authority_validation.png",
    )

    train_env.close()


if __name__ == "__main__":
    main()
