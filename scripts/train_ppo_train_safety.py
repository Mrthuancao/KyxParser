"""Train a mask-aware PPO agent on the generated `train_safety` Gym environment.

Pipeline: parse train_safety.kyx -> generate Gym env -> wrap with Monitor -> train
MaskablePPO -> plot the training reward curve -> evaluate with live progress bars ->
plot the evaluation trajectory and print metrics.

Run from project root (d:/Reinforcement learning/kyx_parser) in the .venv:

python scripts/train_ppo_train_safety.py
python scripts/train_ppo_train_safety.py --timesteps 50000
"""
import argparse
import importlib.util
from collections import Counter, deque
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

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


def run_eval_episode(env, model, out_path: Path, seed: int = 999):
    obs, info = env.reset(seed=seed)
    state_history = [dict(info.get('state', {}))]
    reward_history = []
    terminated = truncated = False
    total_reward = 0.0
    step = 0
    max_steps = getattr(env.unwrapped, 'max_episode_steps', 200)

    with tqdm(total=max_steps, desc="Evaluation episode", unit="step") as progress:
        while not (terminated or truncated) and step < max_steps:
            action, _ = model.predict(
                obs, deterministic=True, action_masks=env.unwrapped.action_masks()
            )
            obs, reward, terminated, truncated, info = env.step(action)
            reward_history.append(reward)
            state_history.append(dict(info.get('state', {})))
            total_reward += reward
            step += 1
            progress.update(1)
            progress.set_postfix(reward=f"{total_reward:.1f}")

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

    final_state = state_history[-1] if state_history else {}
    termination_reason = info.get('termination_reason')
    print(
        f"Evaluation episode: steps={step}, total_reward={total_reward:.3f}, "
        f"terminated={terminated}, truncated={truncated}, "
        f"reason={termination_reason}, final_state={final_state}"
    )
    return {
        'steps': step,
        'total_reward': total_reward,
        'terminated': terminated,
        'truncated': truncated,
        'termination_reason': termination_reason,
        'final_state': final_state,
    }


def _stopping_margin(state: dict) -> float:
    speed = float(state['v'])
    permitted_speed = float(state['d'])
    braking_force = float(state['b'])
    if braking_force <= 0:
        raise ValueError(f"Braking force must be positive, got {braking_force}.")
    return float(
        state['m'] - state['z']
        - (speed ** 2 - permitted_speed ** 2) / (2.0 * braking_force)
    )


def _control_branch_description(action: int) -> str:
    return {
        0: 'v <= vdes; a = -b (brake)',
        1: 'v <= vdes; a = 0 (coast)',
        2: 'v <= vdes; a = A (accelerate)',
        3: 'v > vdes; a = -b (brake)',
        4: 'v > vdes; a = 0 (coast)',
    }.get(action, 'unknown controller branch')


def evaluate_policy_metrics(
    env, model, n_episodes: int, eval_seed: int = 1000,
    trace_window_steps: int = 20,
):
    """Evaluate episodes and retain termination details for every rollout."""
    rewards = []
    lengths = []
    safety_violations = 0
    episode_results = []
    max_steps = getattr(env.unwrapped, 'max_episode_steps', 200)
    if trace_window_steps < 1:
        raise ValueError("trace_window_steps must be at least 1.")

    with tqdm(
        total=n_episodes * max_steps,
        desc="Policy evaluation",
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        for episode_index in range(n_episodes):
            episode_seed = eval_seed + episode_index
            obs, reset_info = env.reset(seed=episode_seed)
            terminated = truncated = False
            total_reward = 0.0
            step = 0
            info = reset_info
            recent_steps = deque(maxlen=trace_window_steps)
            progress.set_description(
                f"Policy evaluation ({episode_index + 1}/{n_episodes})"
            )
            while not (terminated or truncated) and step < max_steps:
                pre_state = dict(env.unwrapped.current_state)
                action_mask = np.asarray(
                    env.unwrapped.action_masks(), dtype=bool
                )
                action, _ = model.predict(
                    obs, deterministic=True, action_masks=action_mask
                )
                selected_action = int(np.asarray(action).reshape(-1)[0])
                control_paths = env.unwrapped.control_choice_paths
                selected_path = (
                    list(control_paths[selected_action])
                    if 0 <= selected_action < len(control_paths) else None
                )
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                post_state = dict(info.get('state', {}))
                recent_steps.append({
                    'step': step,
                    'pre_state': pre_state,
                    'pre_stopping_margin': _stopping_margin(pre_state),
                    'legal_action_mask': action_mask.tolist(),
                    'selected_action': selected_action,
                    'selected_path': selected_path,
                    'selected_branch': _control_branch_description(
                        selected_action
                    ),
                    'post_state': post_state,
                    'post_stopping_margin': _stopping_margin(post_state),
                    'acceleration_after_step': post_state.get('a'),
                    'reward': float(reward),
                    'terminated': bool(terminated),
                    'truncated': bool(truncated),
                    'termination_reason': info.get('termination_reason'),
                })
                progress.update(1)
                progress.set_postfix(
                    episode_reward=f"{total_reward:.1f}",
                    terminated=terminated,
                )
            rewards.append(total_reward)
            lengths.append(step)
            termination_reason = info.get(
                'termination_reason',
                'unknown_termination' if terminated else (
                    'time_limit_truncation' if truncated else 'step_limit_reached'
                ),
            )
            final_state = dict(info.get('state', {}))
            safety_failure = termination_reason in {
                'model_loop_invariant_violation',
                'model_safety_postcondition_violation',
            }
            if safety_failure:
                safety_violations += 1
            episode_results.append({
                'episode': episode_index + 1,
                'seed': episode_seed,
                'steps': step,
                'total_reward': float(total_reward),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'termination_reason': termination_reason,
                'control_flow_info': info.get('control_flow_info'),
                'ode_failure_reason': info.get('ode_failure_reason'),
                'final_state': final_state,
                'safety_failure': safety_failure,
                'failure_trace': (
                    list(recent_steps) if safety_failure else []
                ),
            })

    print("\n=== Evaluation metrics over {} episodes ===".format(n_episodes))
    print(f"Mean reward: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    print(f"Mean episode length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    print(f"Safety predicate/invariant failures: {safety_violations}/{n_episodes}")
    print(
        "Termination reasons: "
        f"{dict(Counter(str(result['termination_reason']) for result in episode_results))}"
    )
    for result in episode_results:
        print(
            f"Episode {result['episode']} (seed={result['seed']}): "
            f"steps={result['steps']}, reward={result['total_reward']:.3f}, "
            f"terminated={result['terminated']}, truncated={result['truncated']}, "
            f"reason={result['termination_reason']}, "
            f"control_flow={result['control_flow_info']}, "
            f"ode_failure={result['ode_failure_reason']}, "
            f"final_state={result['final_state']}"
        )
        if result['safety_failure']:
            print(
                f"  Last {len(result['failure_trace'])} steps before "
                "invariant violation:"
            )
            for trace_step in result['failure_trace']:
                print(
                    f"    step={trace_step['step']} "
                    f"margin={trace_step['pre_stopping_margin']:.6f} "
                    f"mask={trace_step['legal_action_mask']} "
                    f"action={trace_step['selected_action']} "
                    f"branch={trace_step['selected_branch']} "
                    f"a_after={trace_step['acceleration_after_step']} "
                    f"next_margin={trace_step['post_stopping_margin']:.6f}"
                )
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_length': float(np.mean(lengths)),
        'safety_violations': safety_violations,
        'n_episodes': n_episodes,
        'episodes': episode_results,
    }


def _raise_json_type_error(value):
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dl-path', type=Path, default=proj_root / 'test' / 'input' / 'withBound' / 'train_safety.kyx')
    parser.add_argument('--env-name', type=str, default='KeYmaeraTrainSafety')
    parser.add_argument('--timesteps', type=int, default=20000)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--eval-seed', type=int, default=1000)
    parser.add_argument('--trace-window-steps', type=int, default=20)
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Skip training and evaluate the saved PPO policy.',
    )
    parser.add_argument('--out-dir', type=Path, default=proj_root / 'logs' / 'plots')
    parser.add_argument('--model-dir', type=Path, default=proj_root / 'models')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    env_out = proj_root / f"generated_{args.dl_path.stem.lower()}_env.py"

    EnvClass = build_env_class(args.dl_path, args.env_name, env_out)

    def make_env(use_initial_state_buffer: bool = True):
        env = EnvClass(render_mode=None, config_override={
            'initial_state_buffer_config': {
                'enabled': use_initial_state_buffer,
                'strategy': 'z3_diverse',
                'capacity': 100,
                'max_populate_attempts': 1000
            }
        })
        return Monitor(env)

    model_path = args.model_dir / f"ppo_{args.dl_path.stem.lower()}.zip"
    if args.eval_only:
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved PPO policy not found: {model_path}")
        validation_env = make_env(use_initial_state_buffer=False)
        action_space_n = getattr(validation_env.action_space, 'n', None)
        validation_env.close()
        if action_space_n != 5:
            raise RuntimeError(
                "Expected controlFunc to expose five discrete acceleration choices, "
                f"but generated action space has n={action_space_n}. Ensure the "
                "modal program calls controlFunc with the expected branches."
            )
        model = MaskablePPO.load(str(model_path))
        print(f"Loaded saved PPO policy from {model_path}")
    else:
        train_env = DummyVecEnv([make_env])
        action_space_n = getattr(train_env.get_attr('action_space')[0], 'n', None)
        if action_space_n != 5:
            train_env.close()
            raise RuntimeError(
                "Expected controlFunc to expose five discrete acceleration choices, "
                f"but generated action space has n={action_space_n}. Ensure the "
                "modal program calls controlFunc with the expected branches."
            )
        model = MaskablePPO('MlpPolicy', train_env, verbose=1)
        callback = EpisodeStatsCallback()
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
        model.save(str(model_path))
        print(f"Saved trained model to {model_path}")
        plot_training_curve(
            callback,
            args.out_dir / f"{args.dl_path.stem.lower()}_training_curve.png",
        )
        train_env.close()

    eval_env = make_env(use_initial_state_buffer=False)
    run_eval_episode(
        eval_env, model,
        args.out_dir / f"{args.dl_path.stem.lower()}_episode.png",
        seed=args.eval_seed - 1,
    )
    evaluation = evaluate_policy_metrics(
        eval_env, model, args.eval_episodes,
        eval_seed=args.eval_seed,
        trace_window_steps=args.trace_window_steps,
    )
    eval_env.close()
    diagnostics_path = args.out_dir / 'train_safety_evaluation_episodes.json'
    with diagnostics_path.open('w', encoding='utf-8') as diagnostics_file:
        json.dump(
            evaluation['episodes'],
            diagnostics_file,
            indent=2,
            default=lambda value: (
                value.item() if isinstance(value, np.generic)
                else _raise_json_type_error(value)
            ),
        )
    print(f"Saved per-episode termination diagnostics to {diagnostics_path}")


if __name__ == '__main__':
    main()
