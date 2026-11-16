"""Analyse how a Car 1 policy uses the flattened controller-choice leaves.

Compares the trained policy with a uniformly random legal-path policy and writes
figures and a JSON summary. Run after scripts/train_ppo_car1.py:

    .venv\\Scripts\\python.exe scripts\\analyze_car1_paths.py --out-dir logs\\other_exp\\car1
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ppo_car1 as car1


def rollout(env, choose, episodes, seed):
    base = env.unwrapped
    n_paths = int(base.action_space.n)
    legal_counts = []
    records = []
    for index in range(episodes):
        obs, info = env.reset(seed=seed + index)
        done = False
        steps = []
        while not done:
            masks = env.get_wrapper_attr('action_masks')()
            path_mask = masks[:n_paths]
            legal_counts.append(path_mask.astype(int))
            action = choose(obs, masks, n_paths)
            obs, reward, terminated, truncated, info = env.step(action)
            state = info['state']
            separation = float(np.hypot(state['x'] - state['xo'], state['y'] - state['yo']))
            steps.append({
                'path': int(action[0]), 'acc_bin': int(action[1]), 'rad_bin': int(action[2]),
                'v': float(state['v']), 'separation': separation, 'reward': float(reward),
                'legal': int(path_mask.sum()),
            })
            done = terminated or truncated
        records.append({
            'steps': steps, 'terminated': bool(terminated), 'truncated': bool(truncated),
            'reason': info.get('termination_reason'),
        })
    return records, np.array(legal_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=car1.proj_root / 'models' / 'other_exp' / 'ppo_car1.zip')
    parser.add_argument('--out-dir', type=Path, default=car1.proj_root / 'logs' / 'other_exp' / 'car1')
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env_class = car1.load_environment_class(
        car1.proj_root / 'test' / 'input' / 'withBound' / 'car1.txt',
        car1.proj_root / 'generated_car1_env.py',
    )
    env = Monitor(car1.HierarchicalCar1Env(env_class(
        render_mode=None,
        config_override={'ode_solver_step_size': 0.02, 'initial_state_buffer_config': {'enabled': False}},
    )))
    base = env.unwrapped
    paths = [tuple(p) for p in base.control_choice_paths]
    print('Leaf paths:', paths)
    model = MaskablePPO.load(str(args.model))
    rng = np.random.default_rng(0)

    def trained(obs, masks, n_paths):
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        return action

    def random_legal(obs, masks, n_paths):
        legal = np.flatnonzero(masks[:n_paths])
        return np.array([rng.choice(legal), rng.integers(car1.HierarchicalCar1Env.acceleration_bins),
                         rng.integers(car1.HierarchicalCar1Env.radius_bins)])

    results = {}
    for name, policy in (('trained', trained), ('random_legal', random_legal)):
        records, legal = rollout(env, policy, args.episodes, args.seed)
        results[name] = (records, legal)

    labels = [''.join(map(str, p)) or 'root' for p in paths]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Legal availability of each leaf over all visited states.
    width = 0.4
    x = np.arange(len(paths))
    for offset, name in ((-width / 2, 'trained'), (width / 2, 'random_legal')):
        axes[0, 0].bar(x + offset, results[name][1].mean(axis=0), width, label=name)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel('Fraction of states where leaf is legal')
    axes[0, 0].set_xlabel('Leaf (left/right decisions)')
    axes[0, 0].set_title('Mask availability per leaf')
    axes[0, 0].legend()

    # Selection frequency.
    for offset, name in ((-width / 2, 'trained'), (width / 2, 'random_legal')):
        counts = Counter(s['path'] for r in results[name][0] for s in r['steps'])
        total = sum(counts.values())
        axes[0, 1].bar(x + offset, [counts.get(i, 0) / total for i in range(len(paths))], width, label=name)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel('Selection frequency')
    axes[0, 1].set_xlabel('Leaf')
    axes[0, 1].set_title('Leaf selection frequency')
    axes[0, 1].legend()

    # Episode length / outcome.
    for name, color in (('trained', 'tab:blue'), ('random_legal', 'tab:orange')):
        lengths = [len(r['steps']) for r in results[name][0]]
        axes[1, 0].hist(lengths, bins=15, alpha=0.6, color=color, label=f'{name} (mean {np.mean(lengths):.0f})')
    axes[1, 0].set_xlabel('Episode length (steps)')
    axes[1, 0].set_ylabel('Episodes')
    axes[1, 0].set_title('Episode length')
    axes[1, 0].legend()

    # Number of legal leaves over time for the trained policy.
    max_len = max(len(r['steps']) for r in results['trained'][0])
    grid = np.full((len(results['trained'][0]), max_len), np.nan)
    for i, r in enumerate(results['trained'][0]):
        grid[i, :len(r['steps'])] = [s['legal'] for s in r['steps']]
    image = axes[1, 1].imshow(grid, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Episode')
    axes[1, 1].set_title('Legal leaves per step (trained policy)')
    fig.colorbar(image, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(args.out_dir / 'car1_leaf_analysis.png', dpi=150)
    plt.close(fig)

    # Parameter choices and speed/separation of the trained policy.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    heat = np.zeros((car1.HierarchicalCar1Env.acceleration_bins, car1.HierarchicalCar1Env.radius_bins))
    for r in results['trained'][0]:
        for s in r['steps']:
            heat[s['acc_bin'], s['rad_bin']] += 1
    image = axes[0].imshow(heat, aspect='auto', origin='lower')
    axes[0].set_xlabel('Radius bin')
    axes[0].set_ylabel('Acceleration bin')
    axes[0].set_title('Trained policy: parameter choices')
    fig.colorbar(image, ax=axes[0])
    for r in results['trained'][0][:10]:
        axes[1].plot([s['v'] for s in r['steps']], alpha=0.7)
        axes[2].plot([s['separation'] for s in r['steps']], alpha=0.7)
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('v'); axes[1].set_title('Speed (10 episodes)')
    axes[2].set_xlabel('Step'); axes[2].set_ylabel('Distance to obstacle'); axes[2].set_title('Robot-obstacle distance')
    fig.tight_layout()
    fig.savefig(args.out_dir / 'car1_parameters_and_trajectories.png', dpi=150)
    plt.close(fig)

    summary = {'leaf_paths': [list(p) for p in paths]}
    for name, (records, legal) in results.items():
        counts = Counter(s['path'] for r in records for s in r['steps'])
        summary[name] = {
            'episodes': len(records),
            'mean_length': float(np.mean([len(r['steps']) for r in records])),
            'terminated': sum(r['terminated'] for r in records),
            'truncated': sum(r['truncated'] for r in records),
            'reasons': dict(Counter(str(r['reason']) for r in records)),
            'leaf_selection_counts': {str(k): counts.get(k, 0) for k in range(len(paths))},
            'leaf_legal_fraction': legal.mean(axis=0).tolist(),
            'mean_reward': float(np.mean([sum(s['reward'] for s in r['steps']) for r in records])),
            'min_separation': float(min(s['separation'] for r in records for s in r['steps'])),
        }
    (args.out_dir / 'car1_leaf_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
