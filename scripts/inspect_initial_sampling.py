import importlib.util
import sys
from pathlib import Path
import numpy as np

proj_root = Path(__file__).resolve().parents[1]
ast2 = proj_root / 'AST_parser_2'
if str(ast2) not in sys.path:
    sys.path.insert(0, str(ast2))
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

module_path = proj_root / 'generated_car2_env_nb.py'

# dynamic import
spec = importlib.util.spec_from_file_location('generated_car2_env_nb', str(module_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

EnvClass = getattr(module, 'KeYmaeraGymEnv')
if EnvClass is None:
    raise RuntimeError('KeYmaeraGymEnv not found')

# instantiate with GDelta override so assumptions can be satisfied
env = EnvClass(render_mode=None, config_override={
    'constants': [{'name': 'GDelta', 'value': '40.0'}]
})

samplers = getattr(env, 'initial_state_samplers', {})
print('Number of sampler variables:', len(samplers))
print('Sampler variable names:', list(samplers.keys()))

# sample each sampler many times and compute stats
n_samples = 1000
stats = {}
for name, fn in samplers.items():
    vals = np.array([fn() for _ in range(n_samples)], dtype=float)
    stats[name] = {
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'unique': int(len(np.unique(np.round(vals, decimals=6))))
    }

for name, s in stats.items():
    print(f"{name}: min={s['min']}, max={s['max']}, mean={s['mean']:.4f}, std={s['std']:.4f}, unique={s['unique']}")

# initial state buffer config
buf_cfg = env.initial_state_buffer_config
print('Initial state buffer config:', buf_cfg)
print('Initial state buffer length currently:', len(env.initial_state_buffer))

# Optionally populate and examine buffer if empty
if len(env.initial_state_buffer) == 0 and buf_cfg.get('populate_on_init', False):
    env._populate_initial_state_buffer()
    print('Populated buffer, new length:', len(env.initial_state_buffer))

# Generate a set of full sampled states
if samplers:
    states = []
    for _ in range(200):
        st = {name: samplers[name]() for name in samplers}
        states.append(st)
    # show sample diversity for combined states (unique tuples)
    tuples = [tuple(round(v,6) for v in s.values()) for s in states]
    print('Unique full-state tuples (of 200 samples):', len(set(tuples)))

print('Done')
