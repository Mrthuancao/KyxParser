import importlib.util, json
from pathlib import Path
import z3

p = Path('tmp_generated_envs/GeneratedDiversityEnv.py')
spec = importlib.util.spec_from_file_location('genenv', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
env = mod.KeYmaeraGymEnv()
print('function names:', [f['name'] for f in env.config['function_definitions'][:10]], '... total', len(env.config['function_definitions']))

cfg = env.config
z3_symbols = {v['name']: z3.Real(v['name']) for v in cfg['variables']}
fixed = {c['name']: float(c['value']) for c in cfg['constants']}
exprs = [
    'self._bounds_impl()',
    'self._initialState_impl(xr, vr, self._ixr_impl(), xo, vo, self._ixo_impl(), T)',
    cfg['problem']['initial_condition'],
]
for expr in exprs:
    print('EXPR:', expr)
    try:
        val = env._expr_to_z3(expr, z3_symbols, fixed)
        print('RESULT', val)
    except Exception as e:
        import traceback; traceback.print_exc()
    print('---')
