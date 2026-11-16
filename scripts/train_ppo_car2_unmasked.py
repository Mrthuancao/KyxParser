"""Train Car 2 PPO without action masks and compare it with the masked policy.

Both policies are evaluated on the same sampled initial states. The unmasked policy
is trained with ordinary PPO and invalid actions terminate with the model's failure
penalty; the reference policy is the existing MaskablePPO checkpoint.

Run from the project root:

    .\\.venv\\Scripts\\python.exe scripts\\train_ppo_car2_unmasked.py
"""
import argparse
import copy
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from scripts.train_ppo_car2 import (
    EpisodeStatsCallback,
    build_env_class,
    collect_episodes,
    dynamic_import_module,
    evaluate_policy_metrics,
    plot_single_episode,
    plot_training_curve,
)


def create_env(env_class, invalid_action_mode: str, buffer_capacity: int):
    return env_class(render_mode=None, config_override={
        'ode_failure_penalty': 0,
        'invalid_action_mode': invalid_action_mode,
        'ode_solver_step_size': 0.02,
        'initial_state_buffer_config': {
            'enabled': True,
            'strategy': 'z3_diverse',
            'capacity': buffer_capacity,
            'max_populate_attempts': buffer_capacity * 2,
        },
    })


def install_initial_states(env, initial_states: list[dict]) -> None:
    env.initial_state_buffer_config['enabled'] = True
    env.initial_state_buffer = deque(
        (copy.deepcopy(state) for state in initial_states),
        maxlen=len(initial_states),
    )


def write_paired_results(unmasked_episodes: list, masked_episodes: list, path: Path) -> None:
    fields = [
        'episode', 'initial_xr', 'initial_vr', 'initial_xg',
        'unmasked_success', 'unmasked_termination', 'unmasked_steps', 'unmasked_reward',
        'masked_success', 'masked_termination', 'masked_steps', 'masked_reward',
    ]
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for index, (unmasked, masked) in enumerate(
            zip(unmasked_episodes, masked_episodes), start=1
        ):
            initial = unmasked['state_history'][0]
            writer.writerow({
                'episode': index,
                'initial_xr': initial.get('xr'),
                'initial_vr': initial.get('vr'),
                'initial_xg': initial.get('xg'),
                'unmasked_success': unmasked['success'],
                'unmasked_termination': unmasked['termination_reason'],
                'unmasked_steps': unmasked['steps'],
                'unmasked_reward': unmasked['total_reward'],
                'masked_success': masked['success'],
                'masked_termination': masked['termination_reason'],
                'masked_steps': masked['steps'],
                'masked_reward': masked['total_reward'],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dl-path', type=Path,
        default=proj_root / 'test' / 'input' / 'withBound' / 'car2.txt',
    )
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--eval-episodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument(
        '--masked-model', type=Path,
        default=proj_root / 'models' / 'car2' / 'ppo_car2.zip',
    )
    parser.add_argument(
        '--out-dir', type=Path,
        default=proj_root / 'logs' / 'car2_unmasked_comparison',
    )
    parser.add_argument(
        '--model-dir', type=Path,
        default=proj_root / 'models' / 'car2_unmasked_comparison',
    )
    args = parser.parse_args()
    if args.timesteps < 1 or args.eval_episodes < 1:
        parser.error('--timesteps and --eval-episodes must be positive')
    if not args.masked_model.is_file():
        raise FileNotFoundError(f'Masked Car 2 checkpoint not found: {args.masked_model}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    env_path = args.out_dir / 'generated_car2_env.py'
    env_class = build_env_class(args.dl_path, 'KeYmaeraCar2UnmaskedComparison', env_path)

    train_capacity = min(max(args.timesteps, 2000), 30000)
    train_env = DummyVecEnv([
        lambda: Monitor(create_env(env_class, 'penalize', train_capacity))
    ])
    callback: BaseCallback = EpisodeStatsCallback()
    unmasked_model = PPO(
        'MlpPolicy', train_env, verbose=1, ent_coef=0.01, seed=args.seed
    )
    unmasked_model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,
    )
    unmasked_model_path = args.model_dir / 'ppo_car2_unmasked.zip'
    unmasked_model.save(str(unmasked_model_path))
    plot_training_curve(callback, args.out_dir / 'car2_unmasked_training_curve.png')
    train_env.close()

    eval_capacity = min(args.eval_episodes, 30000)
    sampler = create_env(env_class, 'penalize', eval_capacity)
    if len(sampler.initial_state_buffer) < args.eval_episodes:
        raise RuntimeError(
            'Initial-state buffer contains fewer states than requested evaluations: '
            f'{len(sampler.initial_state_buffer)} < {args.eval_episodes}'
        )
    initial_states = [
        copy.deepcopy(sampler.initial_state_buffer.popleft())
        for _ in range(args.eval_episodes)
    ]
    sampler.close()

    unmasked_env = create_env(env_class, 'penalize', 1)
    masked_env = create_env(env_class, 'mask', 1)
    install_initial_states(unmasked_env, initial_states)
    install_initial_states(masked_env, initial_states)

    masked_model = MaskablePPO.load(str(args.masked_model))
    max_steps = getattr(unmasked_env, 'max_episode_steps', 200)
    unmasked_episodes = collect_episodes(
        unmasked_env, unmasked_model, args.eval_episodes, max_steps, use_masks=False
    )
    masked_episodes = collect_episodes(
        masked_env, masked_model, args.eval_episodes, max_steps, use_masks=True
    )

    for label, episodes, env in (
        ('unmasked', unmasked_episodes, unmasked_env),
        ('masked_reference', masked_episodes, masked_env),
    ):
        for index, episode in enumerate(
            (item for item in episodes if item['success']), start=0
        ):
            plot_single_episode(
                episode,
                args.out_dir / f'car2_{label}_success_{index}.png',
                float(env.current_state['GDelta']),
            )
        evaluate_policy_metrics(
            episodes, args.out_dir / f'car2_{label}_validation.png'
        )

    paired_path = args.out_dir / 'car2_paired_comparison.csv'
    write_paired_results(unmasked_episodes, masked_episodes, paired_path)
    summary = {
        'training_timesteps_requested': args.timesteps,
        'seed': args.seed,
        'evaluation_episodes': args.eval_episodes,
        'initial_states_paired': True,
        'unmasked_successes': sum(e['success'] for e in unmasked_episodes),
        'unmasked_success_rate': float(np.mean([e['success'] for e in unmasked_episodes])),
        'masked_successes': sum(e['success'] for e in masked_episodes),
        'masked_success_rate': float(np.mean([e['success'] for e in masked_episodes])),
        'unmasked_model': str(unmasked_model_path),
        'masked_model': str(args.masked_model),
    }
    summary_path = args.out_dir / 'car2_comparison_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Paired comparison saved to {paired_path}')
    print(f'Summary saved to {summary_path}')
    print(
        f"Unmasked: {summary['unmasked_successes']}/{args.eval_episodes} "
        f"({summary['unmasked_success_rate']:.1%}); "
        f"masked reference: {summary['masked_successes']}/{args.eval_episodes} "
        f"({summary['masked_success_rate']:.1%})"
    )
    unmasked_env.close()
    masked_env.close()


if __name__ == '__main__':
    main()
