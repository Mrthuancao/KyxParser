"""Train a PPO agent on the generated `train_safety` Gym environment and report results.

Pipeline: parse train_safety.kyx -> generate Gym env -> wrap with Monitor -> train PPO
(stable-baselines3) -> plot training reward curve -> roll out one evaluation episode with the
trained policy -> plot episode trajectory (state variables + per-step reward) -> print metrics.

Run from project root (d:/Reinforcement learning/kyx_parser) in the .venv:

python scripts/train_ppo_train_safety.py
python scripts/train_ppo_train_safety.py --timesteps 50000
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure AST_parser_2 is importable
proj_root = Path(__file__).resolve().parents[1]
ast2 = proj_root / 'AST_parser_2'
if str(ast2) not in sys.path:
    sys.path.insert(0, str(ast2))
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from generator_core import generate_gym_environment_file


def dynamic_import_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_env_class(dl_path: Path, env_name: str, out_path: Path):
    with open(dl_path, 'r', encoding='utf-8') as f:
        dl_content = f.read()
    ok = generate_gym_environment_file(dl_content, env_name, str(out_path))
    if not ok:
        raise RuntimeError("Environment generation failed")
    module = dynamic_import_module(out_path, f"gen_{dl_path.stem}")
    return getattr(module, 'KeYmaeraGymEnv')


class EpisodeStatsCallback(BaseCallback):
    """Collects per-episode reward/length reported by the Monitor wrapper during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            ep_info = info.get('episode')
            if ep_info is not None:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
        return True


def moving_average(values, window):
    if len(values) < window:
        return np.array([], dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_training_curve(callback: EpisodeStatsCallback, out_path: Path):
    rewards = callback.episode_rewards
    if not rewards:
        print("No episodes completed during training; skipping training curve plot.")
        return
    window = max(1, min(20, len(rewards) // 5))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards, alpha=0.3, label='Episode reward')
    ma = moving_average(rewards, window)
    if len(ma) > 0:
        offset = len(rewards) - len(ma)
        ax.plot(range(offset, len(rewards)), ma, color='tab:red', label=f'Moving average (w={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total episode reward')
    ax.set_title('PPO training reward per episode')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved training curve to {out_path}")


def run_eval_episode(env, model, out_path: Path):
    obs, info = env.reset()
    state_history = [dict(info.get('state', {}))]
    reward_history = []
    terminated = truncated = False
    total_reward = 0.0
    step = 0
    max_steps = getattr(env, 'max_episode_steps', 200)

    while not (terminated or truncated) and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_history.append(reward)
        state_history.append(dict(info.get('state', {})))
        total_reward += reward
        step += 1

    var_names = list(state_history[0].keys()) if state_history else []
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name in var_names:
        axes[0].plot([s.get(name, np.nan) for s in state_history], label=name)
    axes[0].set_ylabel('State value')
    axes[0].set_title('Evaluation episode: state trajectories')
    axes[0].legend(loc='upper right', fontsize='small', ncol=2)

    axes[1].plot(reward_history, color='tab:green')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Evaluation episode: per-step reward')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved episode trajectory plot to {out_path}")

    print(f"Evaluation episode: steps={step}, total_reward={total_reward:.3f}, "
          f"terminated={terminated}, truncated={truncated}")
    return {'steps': step, 'total_reward': total_reward, 'terminated': terminated, 'truncated': truncated}


def evaluate_policy_metrics(env, model, n_episodes: int):
    """Runs n_episodes with the trained policy and reports aggregate reward/length metrics."""
    rewards = []
    lengths = []
    safety_violations = 0
    max_steps = getattr(env, 'max_episode_steps', 200)

    for _ in range(n_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        step = 0
        while not (terminated or truncated) and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
        rewards.append(total_reward)
        lengths.append(step)
        if terminated:
            safety_violations += 1

    print("\n=== Evaluation metrics over {} episodes ===".format(n_episodes))
    print(f"Mean reward: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    print(f"Mean episode length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    print(f"Safety violations (terminated early): {safety_violations}/{n_episodes}")
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_length': float(np.mean(lengths)),
        'safety_violations': safety_violations,
        'n_episodes': n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dl-path', type=Path, default=proj_root / 'test' / 'input' / 'withBound' / 'train_safety.kyx')
    parser.add_argument('--env-name', type=str, default='KeYmaeraTrainSafety')
    parser.add_argument('--timesteps', type=int, default=20000)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--out-dir', type=Path, default=proj_root / 'logs' / 'plots')
    parser.add_argument('--model-dir', type=Path, default=proj_root / 'models')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    env_out = proj_root / f"generated_{args.dl_path.stem.lower()}_env.py"

    EnvClass = build_env_class(args.dl_path, args.env_name, env_out)

    def make_env():
        env = EnvClass(render_mode=None, config_override={
            'initial_state_buffer_config': {
                'enabled': True,
                'strategy': 'z3_diverse',
                'capacity': 100,
                'max_populate_attempts': 1000
            }
        })
        return Monitor(env)

    train_env = DummyVecEnv([make_env])

    action_space_n = getattr(train_env.get_attr('action_space')[0], 'n', None)
    if action_space_n == 1:
        print("WARNING: action space is Discrete(1) — the generated model exposes no "
              "controllable decision variable, so the PPO agent has nothing meaningful to learn.")

    model = PPO('MlpPolicy', train_env, verbose=1)

    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model_path = args.model_dir / f"ppo_{args.dl_path.stem.lower()}.zip"
    model.save(str(model_path))
    print(f"Saved trained model to {model_path}")

    plot_training_curve(callback, args.out_dir / f"{args.dl_path.stem.lower()}_training_curve.png")

    eval_env = make_env()
    run_eval_episode(eval_env, model, args.out_dir / f"{args.dl_path.stem.lower()}_episode.png")
    evaluate_policy_metrics(eval_env, model, args.eval_episodes)


if __name__ == '__main__':
    main()
