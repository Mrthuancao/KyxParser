"""Train a PPO agent on the generated `car2` Gym environment and report results.

car2 is a 1-D waypoint-reaching robot: `xr` (position), `vr` (velocity), `ar` (acceleration),
driving toward a goal at `xg` within a `GDelta`-wide tolerance zone. There is no `y` axis in
this model.

Pipeline: parse car2.txt -> generate Gym env -> wrap with Monitor -> train PPO
(stable-baselines3) -> plot training curve (reward + episode length) -> roll out multiple evaluation
episodes and plot a sample of successful/failed trajectories -> run N validation episodes and plot 
per-episode success/failure -> print metrics.

Run from project root (d:/Reinforcement learning/kyx_parser) in the .venv:

python scripts/train_ppo_car2.py
python scripts/train_ppo_car2.py --timesteps 50000

Note: the car2 model's ODE integration is significantly slower than train_safety's (~10-15
steps/sec), so 10k timesteps takes roughly 10-15 minutes.
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable

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
    """Training graph: per-episode reward and episode length, each with a moving average."""
    rewards = callback.episode_rewards
    lengths = callback.episode_lengths
    if not rewards:
        print("No episodes completed during training; skipping training curve plot.")
        return
    window = max(1, min(20, len(rewards) // 5))
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(rewards, alpha=0.3, label='Episode reward')
    ma = moving_average(rewards, window)
    if len(ma) > 0:
        offset = len(rewards) - len(ma)
        axes[0].plot(range(offset, len(rewards)), ma, color='tab:red', label=f'Moving average (w={window})')
    axes[0].set_ylabel('Total episode reward')
    axes[0].set_title('Training: reward per episode (car2)')
    axes[0].legend()

    axes[1].plot(lengths, alpha=0.3, color='tab:purple', label='Episode length')
    ma_len = moving_average(lengths, window)
    if len(ma_len) > 0:
        offset = len(lengths) - len(ma_len)
        axes[1].plot(range(offset, len(lengths)), ma_len, color='tab:orange', label=f'Moving average (w={window})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode length (steps)')
    axes[1].set_title('Training: episode length per episode')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved training curve to {out_path}")


def reached_waypoint_from_states(
    state_history: list,
    ensure_condition: Callable[[float, float, float], bool],
) -> bool:
    """Apply the generated model predicate to each recorded state, including pre-crash states."""
    for s in state_history:
        xr, vr, xg = s.get('xr'), s.get('vr'), s.get('xg')
        if xr is None or vr is None or xg is None:
            continue
        if ensure_condition(xr, vr, xg):
            return True
    return False


def plot_single_episode(episode_data: dict, out_path: Path, gdelta: float):
    """Plots a single episode's trajectory: position, velocity, acceleration, reward."""
    state_history = episode_data['state_history']
    reward_history = episode_data['reward_history']
    success = episode_data['success']
    total_reward = episode_data['total_reward']
    steps = episode_data['steps']

    xr_series = [s.get('xr', np.nan) for s in state_history]
    vr_series = [s.get('vr', np.nan) for s in state_history]
    ar_series = [s.get('ar', np.nan) for s in state_history]
    xg_final = state_history[-1].get('xg', np.nan)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(xr_series, label='xr (position)', color='tab:blue')
    axes[0].axhline(xg_final, color='tab:red', linestyle='--', label='xg (goal)')
    axes[0].axhspan(xg_final - gdelta, xg_final + gdelta, color='tab:green', alpha=0.15, label='goal zone (±GDelta)')
    axes[0].set_ylabel('Position')
    axes[0].set_title(f"Episode: robot position vs goal ({'REACHED' if success else 'NOT REACHED'})")
    axes[0].legend(loc='best', fontsize='small')

    axes[1].plot(vr_series, color='tab:purple')
    axes[1].set_ylabel('Velocity (vr)')
    axes[1].set_title('Robot velocity')

    axes[2].plot(ar_series, color='tab:orange')
    axes[2].set_ylabel('Acceleration (ar)')
    axes[2].set_title('Robot acceleration (action)')

    axes[3].plot(reward_history, color='tab:green')
    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('Reward')
    axes[3].set_title('Per-step reward')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved episode plot to {out_path} (steps={steps}, total_reward={total_reward:.3f})")


def collect_episodes(env, model, n_episodes: int, max_steps: int = 200):
    """Collects episode data (state history, rewards, success flag) for n episodes."""
    episodes = []
    ensure_condition = env.unwrapped.user_defined_functions.get('ensure')
    if ensure_condition is None:
        raise RuntimeError("Generated environment does not provide the model's 'ensure' predicate")
    for _ in range(n_episodes):
        obs, info = env.reset()
        state_history = [dict(info.get('state', {}))]
        reward_history = []
        terminated = truncated = False
        total_reward = 0.0
        step = 0

        while not (terminated or truncated) and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_history.append(reward)
            state_history.append(dict(info.get('state', {})))
            total_reward += reward
            step += 1

        success = reached_waypoint_from_states(state_history, ensure_condition)
        episodes.append({
            'state_history': state_history,
            'reward_history': reward_history,
            'success': success,
            'total_reward': total_reward,
            'steps': step,
            'terminated': terminated,
            'truncated': truncated
        })

    return episodes


def evaluate_policy_metrics(episodes: list, out_path: Path):
    """Plots validation graph and prints metrics from collected episodes."""
    rewards = [e['total_reward'] for e in episodes]
    lengths = [e['steps'] for e in episodes]
    successes = [e['success'] for e in episodes]
    safety_violations = sum(1 for e in episodes if e['terminated'] and not e['success'])
    n_episodes = len(episodes)
    success_rate = float(np.mean(successes)) if successes else 0.0

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    colors = ['tab:green' if s else 'tab:red' for s in successes]
    axes[0].bar(range(n_episodes), rewards, color=colors)
    axes[0].axhline(np.mean(rewards), color='black', linestyle='--', label=f'Mean reward = {np.mean(rewards):.2f}')
    axes[0].set_xlabel('Validation episode')
    axes[0].set_ylabel('Total reward')
    axes[0].set_title('Validation: reward per episode (green=reached waypoint, red=failed)')
    axes[0].legend()

    cumulative_success_rate = np.cumsum(successes) / np.arange(1, n_episodes + 1)
    axes[1].plot(cumulative_success_rate, color='tab:blue')
    axes[1].axhline(success_rate, color='black', linestyle='--', label=f'Final success rate = {success_rate * 100:.1f}%')
    axes[1].set_xlabel('Validation episode')
    axes[1].set_ylabel('Cumulative success rate')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title('Validation: waypoint success rate')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved validation graph to {out_path}")

    print("\n=== Validation metrics over {} episodes ===".format(n_episodes))
    print(f"Mean reward: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    print(f"Mean episode length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    print(f"Reached waypoint: {sum(successes)}/{n_episodes} ({success_rate * 100:.1f}%)")
    print(f"Safety violations (terminated early): {safety_violations}/{n_episodes}")
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_length': float(np.mean(lengths)),
        'success_rate': success_rate,
        'safety_violations': safety_violations,
        'n_episodes': n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dl-path', type=Path, default=proj_root / 'test' / 'input' / 'withBound' / 'car2.txt')
    parser.add_argument('--env-name', type=str, default='KeYmaeraCar2')
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--out-dir', type=Path, default=proj_root / 'logs' / 'plots')
    parser.add_argument('--model-dir', type=Path, default=proj_root / 'models')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    env_out = proj_root / f"generated_{args.dl_path.stem.lower()}_env.py"
    
    # AUTO-DELETE old generated env to force fresh regeneration from model file
    if env_out.exists():
        env_out.unlink()
        print(f"Deleted stale generated env: {env_out}")

    EnvClass = build_env_class(args.dl_path, args.env_name, env_out)

    # Use the reward and safety terms defined by the model without adding a separate solver-failure
    # penalty on top.
    ode_failure_penalty = 0

    def make_env(buffer_capacity: int):
        # An untrained/exploring policy often ends episodes almost immediately (it violates the
        # ODE domain), which drains the initial-state buffer far faster than a well-behaved
        # policy would. Once drained, refilling on-the-fly via rejection sampling is extremely
        # slow (or fails outright), so size the buffer to cover a full training run up front.
        env = EnvClass(render_mode=None, config_override={
            'ode_failure_penalty': ode_failure_penalty,
            # Default max_step (0.001) forces >=200 forced RK45 sub-steps per control cycle
            # (ep=0.2), each doing an expensive SymPy substitution -> ~16 steps/sec. Raising it
            # cuts forced sub-steps ~20x (measured ~230+ steps/sec) with no accuracy loss beyond
            # what rtol/atol already bound; solve_ivp still adapts smaller internally if needed.
            'ode_solver_step_size': 0.02,
            'initial_state_buffer_config': {
                'enabled': True,
                'strategy': 'z3_diverse',
                'capacity': buffer_capacity,
                'max_populate_attempts': buffer_capacity * 2
            }
        })
        return Monitor(env)

    train_buffer_capacity = min(max(args.timesteps, 2000), 30000)
    train_env = DummyVecEnv([lambda: make_env(train_buffer_capacity)])

    action_space_n = getattr(train_env.get_attr('action_space')[0], 'n', None)
    if action_space_n == 1:
        print("WARNING: action space is Discrete(1) — the generated model exposes no "
              "controllable decision variable, so the PPO agent has nothing meaningful to learn.")

    # ent_coef adds an exploration bonus so the policy doesn't prematurely collapse to a narrow
    # "just survive" behavior before ever discovering the sparse success reward.
    model = PPO('MlpPolicy', train_env, verbose=1, ent_coef=0.01)

    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model_path = args.model_dir / f"ppo_{args.dl_path.stem.lower()}.zip"
    model.save(str(model_path))
    print(f"Saved trained model to {model_path}")

    plot_training_curve(callback, args.out_dir / f"{args.dl_path.stem.lower()}_training_curve.png")

    eval_buffer_capacity = min(max((args.eval_episodes + 1) * 210, 2000), 30000)
    eval_env = make_env(eval_buffer_capacity)
    max_steps = getattr(eval_env.unwrapped, 'max_episode_steps', 200)
    gdelta = float(eval_env.unwrapped.current_state['GDelta'])

    # Collect episodes for evaluation
    episodes = collect_episodes(eval_env, model, args.eval_episodes, max_steps)
    
    # Separate into successes and failures
    successes = [e for e in episodes if e['success']]
    failures = [e for e in episodes if not e['success']]
    
    # Plot sample successful episodes (up to 5)
    for idx, ep in enumerate(successes[:5]):
        plot_single_episode(ep, args.out_dir / f"{args.dl_path.stem.lower()}_success_{idx}.png", gdelta)
    
    # Plot sample failed episodes (up to 5)
    for idx, ep in enumerate(failures[:5]):
        plot_single_episode(ep, args.out_dir / f"{args.dl_path.stem.lower()}_failure_{idx}.png", gdelta)
    
    # Plot validation metrics
    evaluate_policy_metrics(episodes, args.out_dir / f"{args.dl_path.stem.lower()}_validation.png")

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()