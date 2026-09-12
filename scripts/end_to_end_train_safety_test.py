"""End-to-end test: parse `train_safety.kyx` -> generate Gym env -> smoke test -> longer episode

Run from project root (d:/Reinforcement learning/kyx_parser) in the .venv:

python scripts/end_to_end_train_safety_test.py

"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

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


def run_end_to_end(dl_filepath: Path, env_name: str = "GeneratedTrainSafetyEnv", out_filename: Path = None, long_steps: int = 200):
    dl_filepath = Path(dl_filepath)
    if not dl_filepath.exists():
        raise FileNotFoundError(f"DL input file not found: {dl_filepath}")

    with open(dl_filepath, 'r', encoding='utf-8') as f:
        dl_content = f.read()

    if out_filename is None:
        out_filename = proj_root / f"generated_{dl_filepath.stem.lower()}_env.py"

    print(f"Generating environment module to: {out_filename}")
    ok = generate_gym_environment_file(dl_content, env_name, str(out_filename))
    if not ok:
        raise RuntimeError("Environment generation failed")

    module = dynamic_import_module(out_filename, f"gen_{dl_filepath.stem}")

    EnvClass = getattr(module, 'KeYmaeraGymEnv', None)
    if EnvClass is None:
        raise RuntimeError("Generated module does not expose `KeYmaeraGymEnv` class")

    print("Instantiating environment with initial-state buffer config...")
    env = EnvClass(render_mode=None, config_override={
        'initial_state_buffer_config': {
            'enabled': True,
            'strategy': 'z3_diverse',
            'capacity': 100,
            'max_populate_attempts': 1000
        }
    })

    def analyze_initial_sampling(env, per_var_samples=1000, joint_samples=200):
        samplers = getattr(env, 'initial_state_samplers', {})
        print('\n--- Initial-state sampling analysis (unique joint samples) ---')
        if not samplers:
            print('No initial-state samplers defined in the environment.')
            return None

        var_names = list(samplers.keys())
        print('Sampler variables count:', len(samplers))
        print('Sampler variable names:', var_names)

        joint_draws = []
        for _ in range(per_var_samples):
            st = tuple(round(float(samplers[name]()), 6) for name in var_names)
            joint_draws.append(st)
        unique_joints = list(dict.fromkeys(joint_draws))

        unique_count = len(unique_joints)
        print(f'Unique joint states from {per_var_samples} draws: {unique_count} ({unique_count/per_var_samples*100:.1f}%)')

        if unique_count > 0:
            arr = np.array(unique_joints, dtype=float)
            for i, name in enumerate(var_names):
                col = arr[:, i]
                print(f"{name}: min={float(col.min()):.6f}, max={float(col.max()):.6f}, mean={float(col.mean()):.6f}, std={float(col.std()):.6f}")
        else:
            print('No unique joint samples to compute per-variable stats.')

        buf_cfg = getattr(env, 'initial_state_buffer_config', {})
        print('Initial state buffer config:', buf_cfg)
        buf_len = len(getattr(env, 'initial_state_buffer', []))
        print('Current buffer length:', buf_len)

        small_states = []
        for _ in range(joint_samples):
            st = tuple(round(float(samplers[name]()), 6) for name in var_names)
            small_states.append(st)
        print(f'Unique full-state tuples (of {joint_samples} draws):', len(set(small_states)))
        print('--- end sampling analysis ---\n')

    analyze_initial_sampling(env)

    if not hasattr(env, 'action_space_type'):
        env.action_space_type = ('discrete' if getattr(env.action_space, 'n', None) is not None else 'continuous')
        print('action_space_type set to', env.action_space_type)

    print('\n=== Smoke test ===')
    obs, info = env.reset()
    print('RESET obs shape/type:', type(obs), getattr(obs, 'shape', None))
    print('RESET info keys:', list(info.keys()))

    action = 0 if getattr(env.action_space, 'n', None) is not None else np.zeros(env.action_space.shape)
    res = env.step(action)
    print('STEP result len:', len(res) if hasattr(res, '__len__') else 'non-sequence')
    try:
        print('obs:', res[0])
        print('reward:', res[1])
        print('terminated:', res[2])
        print('truncated:', res[3])
        print('info keys:', list(res[4].keys()))
    except Exception:
        print('Unexpected step return:', res)

    print('\n=== Longer run test ===')
    env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False
    step_count = 0
    max_steps = min(long_steps, getattr(env, 'max_episode_steps', long_steps))

    start_time = time.time()
    while step_count < max_steps and not (terminated or truncated):
        if getattr(env.action_space, 'n', None) is not None:
            a = 0
        else:
            a = np.zeros(env.action_space.shape)
        out = env.step(a)
        try:
            _, r, terminated, truncated, info = out
        except Exception:
            if len(out) == 4:
                _, r, done, info = out
                terminated = done
                truncated = False
            else:
                raise
        total_reward += float(r)
        step_count += 1
        if step_count % 50 == 0:
            print(f"... step {step_count}: total_reward={total_reward:.3f}, terminated={terminated}, truncated={truncated}")

    elapsed = time.time() - start_time
    print(f"Long run finished: steps={step_count}, total_reward={total_reward:.3f}, terminated={terminated}, truncated={truncated}, elapsed={elapsed:.2f}s")

    env.close()
    return {
        'module_path': str(out_filename),
        'steps': step_count,
        'total_reward': total_reward,
        'terminated': terminated,
        'truncated': truncated,
    }


if __name__ == '__main__':
    dl_path = proj_root / 'test' / 'input' / 'withBound' / 'train_safety.kyx'
    result = run_end_to_end(dl_path, env_name='KeYmaeraTrainSafety', out_filename=proj_root / 'generated_train_safety_env.py', long_steps=300)
    print('\nResult summary:', result)
