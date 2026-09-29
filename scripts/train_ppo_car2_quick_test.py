"""Quick test: train PPO for a short run without large buffer population overhead."""
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


def main():
    parser = argparse.ArgumentParser(description="Quick test of car2 training with shape= reward")
    parser.add_argument('--dl-path', type=Path, default=proj_root / 'test' / 'input' / 'withBound' / 'car2.txt')
    parser.add_argument('--env-name', type=str, default='KeYmaeraCar2')
    parser.add_argument('--timesteps', type=int, default=5000)
    parser.add_argument('--out-dir', type=Path, default=proj_root / 'logs' / 'plots')
    parser.add_argument('--model-dir', type=Path, default=proj_root / 'models')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    env_out = proj_root / f"generated_{args.dl_path.stem.lower()}_env.py"

    EnvClass = build_env_class(args.dl_path, args.env_name, env_out)

    car2_constants = [
        {'name': 'ep', 'value': '0.2'},
        {'name': 'b', 'value': '1.0'},
        {'name': 'A', 'value': '1.0'},
        {'name': 'Vmax', 'value': '5.0'},
        {'name': 'GDelta', 'value': '40.0'}
    ]

    ode_failure_penalty = -50.0

    def make_env():
        env = EnvClass(render_mode=None, config_override={
            'constants': car2_constants,
            'ode_failure_penalty': ode_failure_penalty,
            'ode_solver_step_size': 0.02,
            'initial_state_buffer_config': {
                'enabled': False,  # Disable buffer for quick test
            }
        })
        return Monitor(env)

    train_env = DummyVecEnv([make_env])

    print("Starting quick training test with shape= reward shaping...")
    model = PPO('MlpPolicy', train_env, verbose=1, ent_coef=0.01)

    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    print("\n=== Quick Test Results ===")
    if callback.episode_rewards:
        print(f"Episodes completed: {len(callback.episode_rewards)}")
        print(f"Mean reward: {np.mean(callback.episode_rewards):.3f}")
        print(f"Max reward: {np.max(callback.episode_rewards):.3f}")
        print(f"Min reward: {np.min(callback.episode_rewards):.3f}")
        print("\nFirst 10 episode rewards:")
        for i, r in enumerate(callback.episode_rewards[:10]):
            print(f"  Episode {i}: {r:.3f}")
    else:
        print("No episodes completed!")

    train_env.close()


if __name__ == '__main__':
    main()
