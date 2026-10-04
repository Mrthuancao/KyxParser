"""Train PPO on the fixed-authority train-approach Gymnasium task.

The policy chooses between the formal controller's two guarded branches:
0 = brake, 1 = accelerate. An unavailable branch fails its formal test and ends
the episode with a penalty. Success means stopping in [m - delta, m] at speed
no greater than d.

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
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
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
        self.rejected_actions = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self.rejected_actions += int("rejected_action" in info)
            episode = info.get("episode")
            if episode:
                self.episode_rewards.append(episode["r"])
                self.episode_lengths.append(episode["l"])
        return True


class GuardValidActionRecovery(gym.Wrapper):
    """Penalize a rejected branch choice, then execute the guard-valid branch.

    The generated KeYmaera environment itself remains strict: a failed test
    terminates its step. This training wrapper avoids turning that invalid
    choice into an uninformative terminal transition for PPO.
    """

    def __init__(self, env, invalid_action_penalty: float):
        super().__init__(env)
        self.invalid_action_penalty = invalid_action_penalty

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated and info.get("control_flow_info") == "test_condition_failed"):
            return observation, reward, terminated, truncated, info

        requested_action = int(np.asarray(action).item())
        valid_action = 1 - requested_action
        observation, valid_reward, terminated, truncated, info = self.env.step(valid_action)
        info["rejected_action"] = requested_action
        info["executed_action"] = valid_action
        return (
            observation,
            float(valid_reward) - self.invalid_action_penalty,
            terminated,
            truncated,
            info,
        )


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
    parser.add_argument("--invalid-action-penalty", type=float, default=10.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=proj_root / "logs" / "train_essential_fixed_authority",
    )
    parser.add_argument("--model-dir", type=Path, default=proj_root / "models")
    args = parser.parse_args()

    if (args.timesteps < 2 or args.eval_episodes <= 0 or args.max_episode_steps <= 0
            or args.invalid_action_penalty <= 0):
        parser.error("timesteps must be at least 2; episode counts, lengths, and penalty must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    generated_path = args.model_dir / "generated_train_essential_fixed_authority_env.py"
    env_class = build_env_class(args.dl_path, generated_path)

    # Use a feasible, non-degenerate parameter set instead of the generator's
    # generic defaults for bare symbolic constants.
    movement_authority = 10.0
    arrival_speed = 0.2
    arrival_zone_width = 0.5
    constants = [
        {"name": "ep", "value": "0.1"},
        {"name": "b", "value": "1.0"},
        {"name": "A", "value": "1.0"},
        {"name": "m", "value": str(movement_authority)},
        {"name": "d", "value": str(arrival_speed)},
        {"name": "delta", "value": str(arrival_zone_width)},
    ]

    def make_env():
        env = env_class(
            config_override={
                "constants": constants,
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
        return Monitor(GuardValidActionRecovery(env, args.invalid_action_penalty))

    train_env = DummyVecEnv([make_env])
    rollout_steps = min(2048, args.timesteps)
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        ent_coef=0.01,
        n_steps=rollout_steps,
        batch_size=min(64, rollout_steps),
    )
    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model_path = args.model_dir / "ppo_train_essential_fixed_authority.zip"
    model.save(str(model_path))
    print(f"Saved PPO model: {model_path}")

    if callback.episode_rewards:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(callback.episode_rewards, alpha=0.35, label="Episode reward")
        axes[0].set_ylabel("Total reward")
        axes[0].legend()
        axes[1].plot(callback.episode_lengths, alpha=0.6, color="tab:purple")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        fig.suptitle("PPO training — fixed-authority train approach")
        fig.tight_layout()
        plot_path = args.out_dir / "train_essential_fixed_authority_training.png"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved training plot: {plot_path}")
    else:
        print("No episodes completed during training; skipping training plot.")

    eval_env = make_env()
    episode_returns = []
    episode_lengths = []
    safe_truncations = 0
    successful_arrivals = 0
    failed_terminations = 0
    rejected_actions = 0
    for episode_index in range(args.eval_episodes):
        obs, _ = eval_env.reset(seed=episode_index)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            rejected_actions += int("rejected_action" in info)
            total_reward += float(reward)
            steps += 1
        final_state = info.get("state", {})
        success = (
            final_state.get("z", float("-inf")) >= movement_authority - arrival_zone_width
            and final_state.get("z", float("inf")) <= movement_authority
            and 0.0 <= final_state.get("v", float("inf")) <= arrival_speed
        )
        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        safe_truncations += int(truncated and not terminated)
        successful_arrivals += int(terminated and success)
        failed_terminations += int(terminated and not success)

    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Mean return: {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} steps")
    print(f"Successful safe arrivals: {successful_arrivals}/{args.eval_episodes}")
    print(f"Safe time-limit truncations: {safe_truncations}/{args.eval_episodes}")
    print(f"Failed terminations (branch-test failure or safety violation): "
          f"{failed_terminations}/{args.eval_episodes}")
    print(f"Rejected branch requests recovered: {rejected_actions}")
    print(f"Rejected requests during training: {callback.rejected_actions}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
