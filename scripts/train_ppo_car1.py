"""Train PPO on the generated Car 1 passive-safety environment.

The policy chooses an opaque ID for a complete valid path through `controlFunc`,
plus discretized acceleration and curve-radius values. The generated environment
interprets the path; this wrapper supplies Car 1's control values and derives
dependent curve geometry (`w`, `xc`, and `yc`).

Run from the project root:

    .venv\\Scripts\\python.exe scripts\\train_ppo_car1.py
    .venv\\Scripts\\python.exe scripts\\train_ppo_car1.py --timesteps 50000
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

proj_root = Path(__file__).resolve().parents[1]
ast2 = proj_root / 'AST_parser_2'
if str(ast2) not in sys.path:
    sys.path.insert(0, str(ast2))
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from generator_core import generate_gym_environment_file


class HierarchicalCar1Env(gym.Wrapper):
    """Expose a complete controller-path ID and bounded control parameters to PPO."""

    acceleration_bins = 11
    radius_bins = 20
    radius_limit = 10.0

    def __init__(self, env):
        super().__init__(env)
        self.radius_values = np.concatenate((
            np.linspace(-self.radius_limit, -0.5, self.radius_bins // 2),
            np.linspace(0.5, self.radius_limit, self.radius_bins // 2),
        ))
        self.choice_path_count = int(env.action_space.n)
        self.action_space = spaces.MultiDiscrete(
            [self.choice_path_count, self.acceleration_bins, self.radius_bins]
        )
        self._selected_acceleration = 0.0
        self._selected_radius = 1.0
        self._full_step_duration = float(env.time_step)
        self._generated_control_executor = env._execute_control_flow

        # The generated executor owns AST traversal; this wrapper supplies only
        # Car 1's action parameters for nondeterministic assignments.
        env._nondeterministic_assignment_handler = self._assign_nondeterministic
        env._assignment_handler = self._handle_assignment
        env._execute_control_flow = self._execute_control_flow

    def action_masks(self):
        """Mask illegal controller paths; all parameter bins remain selectable."""
        path_mask = np.asarray(self.env.action_masks(), dtype=bool)
        return np.concatenate((
            path_mask,
            np.ones(self.acceleration_bins, dtype=bool),
            np.ones(self.radius_bins, dtype=bool),
        ))

    def _execute_control_flow(self, action):
        action_values = np.asarray(action, dtype=np.int64).reshape(-1)
        expected_size = 3
        if action_values.size != expected_size or not self.action_space.contains(action_values):
            raise ValueError(
                f"Expected a Car 1 action with {expected_size} valid components; "
                f"received {action!r}"
            )

        env = self.env
        state = env.current_state
        self._full_step_duration = float(state.get('ep', env.time_step))
        env.time_step = self._full_step_duration

        constants = {
            name: float(env.current_state[name])
            for name in ('b', 'A')
        }
        braking = constants['b']
        acceleration_limit = constants['A']
        self._selected_acceleration = float(np.linspace(
            -braking,
            acceleration_limit,
            self.acceleration_bins,
        )[action_values[1]])
        self._selected_radius = float(self.radius_values[action_values[2]])

        self._sample_obstacle_velocity()
        status = self._generated_control_executor(int(action_values[0]))
        if not status['terminated']:
            self._debug_print(
                f"Car 1 selected acceleration={self._selected_acceleration:.3f}, "
                f"radius={self._selected_radius:.3f}"
            )
        return status

    def _debug_print(self, message):
        self.env._debug_print(message)

    def _sample_obstacle_velocity(self):
        env = self.env
        obstacle_speed_limit = float(env.current_state['V'])
        speed = obstacle_speed_limit * np.sqrt(env.np_random.uniform())
        angle = env.np_random.uniform(0.0, 2.0 * np.pi)
        env.current_state['vxo'] = float(speed * np.cos(angle))
        env.current_state['vyo'] = float(speed * np.sin(angle))

    def _sample_environment_measurement(self, name):
        sampler = self.env.initial_state_samplers.get(name)
        if sampler is not None:
            return float(sampler())
        config = self.env.config.get('initial_state_sampling_config', {}).get(name, {})
        if 'min' in config and 'max' in config:
            low = float(self.env._evaluate_expression(config['min'], {}))
            high = float(self.env._evaluate_expression(config['max'], {}))
            return float(self.env.np_random.uniform(low, high))
        return float(self.env.current_state.get(name, 0.0))

    def _assign_nondeterministic(self, name):
        state = self.env.current_state
        if name == 'a':
            value = self._selected_acceleration
        elif name == 'r':
            value = self._selected_radius
        elif name == 'w':
            value = state['v'] / self._selected_radius
        elif name == 'xc':
            value = state['x'] - self._selected_radius * state['dy']
        elif name == 'yc':
            value = state['y'] + self._selected_radius * state['dx']
        elif name in ('xo', 'yo'):
            value = self._sample_environment_measurement(name)
        elif name in ('vxo', 'vyo'):
            value = state[name]
        else:
            sampler = self.env.initial_state_samplers.get(name)
            value = sampler() if sampler is not None else state.get(name, 0.0)
        return float(value)

    def _handle_assignment(self, name, value):
        if name == 'a' and value < 0:
            velocity = float(self.env.current_state['v'])
            if velocity <= 0:
                return 0.0
            if velocity / abs(float(value)) < self._full_step_duration:
                self.env.time_step = velocity / abs(float(value))
        return float(value)


class EpisodeStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []
        self.lengths = []

    def _on_step(self):
        for info in self.locals.get('infos', []):
            episode = info.get('episode')
            if episode:
                self.rewards.append(episode['r'])
                self.lengths.append(episode['l'])
        return True


def load_environment_class(model_path, generated_path):
    model_text = model_path.read_text(encoding='utf-8')
    if not generate_gym_environment_file(model_text, 'KeYmaeraCar1', str(generated_path)):
        raise RuntimeError(f"Failed to generate a Gym environment from {model_path}")
    spec = importlib.util.spec_from_file_location('generated_car1_env', generated_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generated environment {generated_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KeYmaeraGymEnv


def plot_training(callback, path):
    if not callback.rewards:
        print("No full episodes completed during training; skipping reward plot.")
        return
    window = max(1, min(20, len(callback.rewards) // 5))
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(callback.rewards, alpha=0.35, label='Episode reward')
    if len(callback.rewards) >= window:
        average = np.convolve(callback.rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(
            range(window - 1, len(callback.rewards)),
            average,
            label=f'Moving average (w={window})',
        )
    axes[0].set_ylabel('Total episode reward')
    axes[0].set_title('Training: reward per episode (Car 1)')
    axes[0].legend()
    axes[1].plot(callback.lengths, alpha=0.35, color='tab:purple', label='Episode length')
    if len(callback.lengths) >= window:
        average_length = np.convolve(
            callback.lengths,
            np.ones(window) / window,
            mode='valid',
        )
        axes[1].plot(
            range(window - 1, len(callback.lengths)),
            average_length,
            color='tab:orange',
            label=f'Moving average (w={window})',
        )
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Length (steps)')
    axes[1].set_title('Training: episode length (Car 1)')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved training plot to {path}")


def _car1_safety_predicates(env, state):
    functions = env.unwrapped.user_defined_functions
    loopinv = functions['loopinv']
    ensure = functions['ensure']
    return {
        'loopinv': bool(loopinv(*(
            state[name] for name in (
                'x', 'y', 'v', 'w', 'r', 'dx', 'dy', 'xo', 'yo', 'xc', 'yc',
            )
        ))),
        'ensure': bool(ensure(*(
            state[name] for name in ('v', 'x', 'y', 'xo', 'yo')
        ))),
    }


def _episode_outcome(episode):
    if episode['control_flow_info']:
        return f"control_flow_failure:{episode['control_flow_info']}"
    if episode['loopinv_violations'] or episode['ensure_violations']:
        return 'safety_violation'
    if episode['terminated']:
        return 'other_termination'
    if episode['truncated']:
        return 'safe_horizon'
    return 'incomplete'


def plot_single_episode(episode, output_path):
    states = episode['state_history']
    steps = np.arange(len(states))
    robot_x = [state['x'] for state in states]
    robot_y = [state['y'] for state in states]
    obstacle_x = [state['xo'] for state in states]
    obstacle_y = [state['yo'] for state in states]
    loopinv = episode['loopinv_history']
    ensure = episode['ensure_history']
    title_outcome = episode['outcome'].replace('_', ' ')

    fig, axes = plt.subplots(4, 1, figsize=(10, 13))
    axes[0].plot(robot_x, robot_y, marker='.', label='Robot trajectory')
    axes[0].plot(obstacle_x, obstacle_y, '--', marker='.', label='Obstacle trajectory')
    axes[0].scatter(robot_x[0], robot_y[0], marker='o', color='tab:green', label='Start')
    axes[0].scatter(robot_x[-1], robot_y[-1], marker='x', color='black', label='End')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f"Episode {episode['index']}: {title_outcome}")
    axes[0].axis('equal')
    axes[0].legend(loc='best', fontsize='small')

    axes[1].plot(steps, [state['v'] for state in states], label='Speed v')
    axes[1].plot(steps, [state['a'] for state in states], label='Acceleration a')
    axes[1].plot(steps, [state['w'] for state in states], label='Angular velocity w')
    axes[1].set_ylabel('Control / state')
    axes[1].legend(loc='best', fontsize='small')

    axes[2].step(steps, loopinv, where='post', label='Loop invariant')
    axes[2].step(steps, ensure, where='post', label='Ensure safety predicate')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0, 1], labels=['False', 'True'])
    axes[2].set_ylabel('Predicate')
    axes[2].legend(loc='best', fontsize='small')

    axes[3].plot(np.arange(1, len(episode['reward_history']) + 1),
                 episode['reward_history'], color='tab:green')
    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('Reward')
    axes[3].set_title(
        f"Reward per step; total={episode['total_reward']:.2f}, "
        f"length={episode['steps']}"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved episode plot to {output_path}")


def _wilson_interval(successes, trials, z=1.96):
    if trials == 0:
        return 0.0, 0.0
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    margin = (
        z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_policy(env, model, episode_count, output_path, seed=10000):
    episodes = []
    max_steps = env.unwrapped.max_episode_steps

    for episode_index in range(episode_count):
        observation, info = env.reset(seed=seed + episode_index)
        states = [dict(info['state'])]
        initial_predicates = _car1_safety_predicates(env, states[0])
        loopinv_history = [initial_predicates['loopinv']]
        ensure_history = [initial_predicates['ensure']]
        rewards = []
        total_reward = 0.0
        terminated = truncated = False
        control_flow_info = ''

        while not (terminated or truncated) and len(rewards) < max_steps:
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=env.get_wrapper_attr('action_masks')(),
            )
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            rewards.append(float(reward))
            states.append(dict(info['state']))
            predicates = _car1_safety_predicates(env, states[-1])
            loopinv_history.append(predicates['loopinv'])
            ensure_history.append(predicates['ensure'])
            control_flow_info = info.get('control_flow_info', control_flow_info)

        episode = {
            'index': episode_index,
            'state_history': states,
            'reward_history': rewards,
            'loopinv_history': loopinv_history,
            'ensure_history': ensure_history,
            'loopinv_violations': sum(not value for value in loopinv_history),
            'ensure_violations': sum(not value for value in ensure_history),
            'total_reward': total_reward,
            'steps': len(rewards),
            'terminated': bool(terminated),
            'truncated': bool(truncated),
            'control_flow_info': control_flow_info,
        }
        episode['outcome'] = _episode_outcome(episode)
        episodes.append(episode)

    safe_episodes = [episode['outcome'] == 'safe_horizon' for episode in episodes]
    rewards = [episode['total_reward'] for episode in episodes]
    lengths = [episode['steps'] for episode in episodes]
    safe_count = sum(safe_episodes)
    safe_rate = safe_count / episode_count
    confidence_low, confidence_high = _wilson_interval(safe_count, episode_count)
    loopinv_violations = sum(episode['loopinv_violations'] for episode in episodes)
    ensure_violations = sum(episode['ensure_violations'] for episode in episodes)
    predicate_checks = sum(len(episode['state_history']) for episode in episodes)
    safety_violations = sum(
        bool(episode['loopinv_violations'] or episode['ensure_violations'])
        for episode in episodes
    )
    control_flow_failures = sum(bool(episode['control_flow_info']) for episode in episodes)
    other_terminations = sum(
        episode['terminated']
        and not episode['control_flow_info']
        and not (episode['loopinv_violations'] or episode['ensure_violations'])
        for episode in episodes
    )
    time_limit_truncations = sum(
        episode['outcome'] == 'safe_horizon' for episode in episodes
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    colors = ['tab:green' if safe else 'tab:red' for safe in safe_episodes]
    axes[0].bar(range(episode_count), rewards, color=colors)
    axes[0].axhline(
        np.mean(rewards),
        color='black',
        linestyle='--',
        label=f'Mean reward = {np.mean(rewards):.2f}',
    )
    axes[0].set_xlabel('Validation episode')
    axes[0].set_ylabel('Total reward')
    axes[0].set_title('Validation: reward (green=safe horizon, red=other outcome)')
    axes[0].legend()

    cumulative_safe_rate = np.cumsum(safe_episodes) / np.arange(1, episode_count + 1)
    axes[1].plot(cumulative_safe_rate, color='tab:blue')
    axes[1].axhline(
        safe_rate,
        color='black',
        linestyle='--',
        label=f'Safe-horizon rate = {safe_rate * 100:.1f}%',
    )
    axes[1].set_xlabel('Validation episode')
    axes[1].set_ylabel('Cumulative rate')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title('Validation: episodes reaching the time limit without safety failure')
    axes[1].legend()

    axes[2].bar(
        ['Loop-invariant', 'Ensure'],
        [loopinv_violations, ensure_violations],
        color=['tab:orange', 'tab:red'],
    )
    axes[2].set_ylabel('False predicate checks')
    axes[2].set_title(f'Safety predicate violations ({predicate_checks} state checks)')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved validation graph to {output_path}")

    safe_samples = [episode for episode in episodes if episode['outcome'] == 'safe_horizon']
    failure_samples = [episode for episode in episodes if episode['outcome'] != 'safe_horizon']
    for sample_index, episode in enumerate(safe_samples[:3]):
        plot_single_episode(
            episode,
            output_path.with_name(f'{output_path.stem}_safe_{sample_index}.png'),
        )
    for sample_index, episode in enumerate(failure_samples[:3]):
        plot_single_episode(
            episode,
            output_path.with_name(f'{output_path.stem}_failure_{sample_index}.png'),
        )

    print(f"\n=== Car 1 evaluation over {episode_count} episodes ===")
    print(f"Mean reward: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    print(f"Mean episode length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    print(
        f"Safe through time limit: {safe_count}/{episode_count} "
        f"({safe_rate * 100:.1f}%, 95% Wilson CI "
        f"{confidence_low * 100:.1f}-{confidence_high * 100:.1f}%)"
    )
    print(f"Episodes with safety-predicate violations: {safety_violations}/{episode_count}")
    print(f"Loop-invariant false state checks: {loopinv_violations}/{predicate_checks}")
    print(f"Ensure false state checks: {ensure_violations}/{predicate_checks}")
    print(f"Control-flow failures: {control_flow_failures}/{episode_count}")
    print(f"Other terminations: {other_terminations}/{episode_count}")
    print(f"Safe time-limit truncations: {time_limit_truncations}/{episode_count}")
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_length': float(np.mean(lengths)),
        'safe_horizon_rate': safe_rate,
        'safe_horizon_ci_95': (confidence_low, confidence_high),
        'safety_violation_episodes': safety_violations,
        'loopinv_violations': loopinv_violations,
        'ensure_violations': ensure_violations,
        'predicate_checks': predicate_checks,
        'control_flow_failures': control_flow_failures,
        'other_terminations': other_terminations,
        'time_limit_truncations': time_limit_truncations,
        'episodes': episodes,
        'n_episodes': episode_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dl-path',
        type=Path,
        default=proj_root / 'test' / 'input' / 'withBound' / 'car1.txt',
    )
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, default=proj_root / 'logs' / 'plots')
    parser.add_argument('--model-dir', type=Path, default=proj_root / 'models')
    args = parser.parse_args()
    if args.timesteps < 2:
        parser.error('--timesteps must be at least 2')
    if args.eval_episodes < 1:
        parser.error('--eval-episodes must be positive')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    generated_path = proj_root / 'generated_car1_env.py'
    EnvClass = load_environment_class(args.dl_path, generated_path)

    def make_env():
        base_env = EnvClass(
            render_mode=None,
            config_override={
                'ode_solver_step_size': 0.02,
                'initial_state_buffer_config': {'enabled': False},
            },
        )
        return Monitor(HierarchicalCar1Env(base_env))

    train_env = DummyVecEnv([make_env])
    train_env.seed(args.seed)
    print(f"Car 1 hierarchical action space: {train_env.action_space}")
    n_steps = min(max(args.timesteps, 32), 512)
    model = MaskablePPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        n_steps=n_steps,
        batch_size=min(64, n_steps),
        ent_coef=0.01,
        seed=args.seed,
    )
    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model_path = args.model_dir / 'ppo_car1.zip'
    model.save(str(model_path))
    print(f"Saved PPO model to {model_path}")
    plot_training(callback, args.out_dir / 'car1_training_curve.png')
    train_env.close()

    evaluation_env = make_env()
    evaluate_policy(
        evaluation_env,
        model,
        args.eval_episodes,
        args.out_dir / 'car1_validation.png',
    )
    evaluation_env.close()


if __name__ == '__main__':
    main()
