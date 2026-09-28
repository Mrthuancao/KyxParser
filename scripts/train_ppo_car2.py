"""Train a PPO agent on the generated `car2` Gym environment and report results.

car2 is a 1-D waypoint-reaching robot: `xr` (position), `vr` (velocity), `ar` (acceleration),
driving toward a goal at `xg` within a `GDelta`-wide tolerance zone. There is no `y` axis in
this model.

Pipeline: parse car2.txt -> generate Gym env -> wrap with Monitor -> train PPO
(stable-baselines3) -> plot training curve (reward + episode length) -> roll out one evaluation
episode and plot car state trajectories (position/velocity/acceleration vs goal) -> run N
validation episodes and plot per-episode success/failure -> print metrics.

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


def reached_waypoint_from_rewards(reward_history: list, success_threshold: float) -> bool:
    """The model's success reward is a large positive spike (vs. 0/negative otherwise), so any
    step near that value unambiguously means the real `ensure(xr,vr,xg)` condition fired --
    more reliable than inspecting only the final state, which can look deceptively "successful"
    right at a crash."""
    return any(r >= success_threshold for r in reward_history)


def run_eval_episode(env, model, out_path: Path, gdelta: float, success_threshold: float):
    """Runs one episode and plots the car's position (vs goal), velocity, acceleration, reward."""
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

    xr_series = [s.get('xr', np.nan) for s in state_history]
    vr_series = [s.get('vr', np.nan) for s in state_history]
    ar_series = [s.get('ar', np.nan) for s in state_history]
    xg_final = state_history[-1].get('xg', np.nan)
    success = reached_waypoint_from_rewards(reward_history, success_threshold)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(xr_series, label='xr (position)', color='tab:blue')
    axes[0].axhline(xg_final, color='tab:red', linestyle='--', label='xg (goal)')
    axes[0].axhspan(xg_final - gdelta, xg_final + gdelta, color='tab:green', alpha=0.15, label='goal zone (±GDelta)')
    axes[0].set_ylabel('Position')
    axes[0].set_title(f"Evaluation episode: robot position vs goal ({'REACHED' if success else 'NOT REACHED'})")
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
    print(f"Saved car state trajectory plot to {out_path}")

    print(f"Evaluation episode: steps={step}, total_reward={total_reward:.3f}, "
          f"terminated={terminated}, truncated={truncated}, reached_waypoint={success}")
    return {'steps': step, 'total_reward': total_reward, 'terminated': terminated,
            'truncated': truncated, 'reached_waypoint': success}


def evaluate_policy_metrics(env, model, n_episodes: int, gdelta: float, success_threshold: float, out_path: Path):
    """Runs n_episodes with the trained policy, plots a validation graph, and prints metrics."""
    rewards = []
    lengths = []
    successes = []
    safety_violations = 0
    max_steps = getattr(env, 'max_episode_steps', 200)

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_rewards = []
        terminated = truncated = False
        total_reward = 0.0
        step = 0
        while not (terminated or truncated) and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            total_reward += reward
            step += 1
        rewards.append(total_reward)
        lengths.append(step)
        if terminated:
            safety_violations += 1
        successes.append(reached_waypoint_from_rewards(episode_rewards, success_threshold))

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

    EnvClass = build_env_class(args.dl_path, args.env_name, env_out)

    # Default constants (each generator-picked as the lower bound of its @const range) leave
    # `bounds()` unsatisfiable (Vmax too small relative to A/b/ep/GDelta), so the initial-state
    # buffer can never populate. Override to a consistent, feasible set.
    car2_constants = [
        {'name': 'ep', 'value': '0.2'},
        {'name': 'b', 'value': '1.0'},
        {'name': 'A', 'value': '1.0'},
        {'name': 'Vmax', 'value': '5.0'},
        {'name': 'GDelta', 'value': '40.0'}
    ]
    gdelta = 40.0

    # The generator's reward is on the model's declared scale (+1000/-1000/0, from the file's own
    # @progress/@safety annotations). That magnitude is too large for PPO's default value-function
    # learning rate: value_loss swings between ~100 and ~30,000 across updates, and ep_rew_mean
    # oscillates/regresses instead of climbing steadily. Rescale to +10/-10/0 for training stability;
    # the *relative* shape of the reward (and hence the optimal policy) is unchanged.
    reward_scale = 0.01
    raw_reward_expr = EnvClass().config['reward_function_body']
    scaled_reward_expr = f"({raw_reward_expr}) * {reward_scale}"
    success_threshold = 500.0 * reward_scale  # half of the model's +1000 success bonus, rescaled

    def make_env(buffer_capacity: int):
        # An untrained/exploring policy often ends episodes almost immediately (it violates the
        # ODE domain), which drains the initial-state buffer far faster than a well-behaved
        # policy would. Once drained, refilling on-the-fly via rejection sampling is extremely
        # slow (or fails outright), so size the buffer to cover a full training run up front.
        env = EnvClass(render_mode=None, config_override={
            'constants': car2_constants,
            'reward_function_body': scaled_reward_expr,
            # The ODE-solver/domain-violation penalty is a separate hardcoded constant (-100.0)
            # from `reward_function_body`, so it must be rescaled to match or it dominates the
            # signal (this is the most common failure mode -- frequent "Differential domain
            # constraint violated" crashes).
            'ode_failure_penalty': -100.0 * reward_scale,
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
    run_eval_episode(eval_env, model, args.out_dir / f"{args.dl_path.stem.lower()}_episode.png", gdelta, success_threshold)
    evaluate_policy_metrics(eval_env, model, args.eval_episodes, gdelta, success_threshold,
                             args.out_dir / f"{args.dl_path.stem.lower()}_validation.png")

    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
