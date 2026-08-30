"""End-to-end test: parse `car2.txt` -> generate Gym env -> smoke test -> longer episode

Run from project root (d:/Reinforcement learning/kyx_parser) in the .venv:

python scripts/end_to_end_car2_test.py

"""
import importlib.util
import json
import sys
from pathlib import Path
import time
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


def run_end_to_end(dl_filepath: Path, env_name: str = "GeneratedCar2Env", out_filename: Path = None, long_steps: int = 200):
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

    # Import generated module
    module = dynamic_import_module(out_filename, f"gen_{dl_filepath.stem}")

    # Locate environment class
    EnvClass = getattr(module, 'KeYmaeraGymEnv', None)
    if EnvClass is None:
        raise RuntimeError("Generated module does not expose `KeYmaeraGymEnv` class")

    print("Instantiating environment...")
    env = EnvClass(render_mode=None, config_override=None)

    # Ensure attribute exists
    if not hasattr(env, 'action_space_type'):
        env.action_space_type = ('discrete' if getattr(env.action_space, 'n', None) is not None else 'continuous')
        print('action_space_type set to', env.action_space_type)

    # --- Smoke test: reset + single step ---
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

    # --- Longer test: run up to `long_steps` or env.max_episode_steps ---
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
        # step returns (obs, reward, terminated, truncated, info)
        try:
            _, r, terminated, truncated, info = out
        except Exception:
            # Older gym variants may return (obs, reward, done, info)
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
        'truncated': truncated
    }


if __name__ == '__main__':
    dl_path = proj_root / 'test' / 'input' / 'withBound' / 'car2.txt'
    result = run_end_to_end(dl_path, env_name='KeYmaeraCar2', out_filename=proj_root / 'generated_car2_env_nb.py', long_steps=500)
    print('\nResult summary:', result)
