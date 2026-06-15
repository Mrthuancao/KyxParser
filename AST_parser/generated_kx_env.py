# GeneratedKXEnv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import re
import math
from typing import (Dict, Any, Tuple, List, Optional, Union)
from gym_generator.kx_env_base import _KeYmaeraXEnvBase

import z3

from pyast_config import (
    GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig,
    ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig,
    HybridProgramConfig, ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig,
    VariableDeclarationConfig, PreconditionConfig, PostconditionConfig,
    LoopExecutionConfig, HybridProgramStepConfig,
    ActionType, ChoiceActionConfig, SequenceActionConfig,
    DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig,
    ActionConstraintConfig, CallHybridProgramConfig
)

class GeneratedKXEnv(_KeYmaeraXEnvBase):
    metadata = {"render_modes": ["human", "plot_data", "matplotlib"], "render_fps": 30}

    def __init__(self, render_mode=None, cisb_size: int = 1000, cisb_populate_attempts: int = 1000, seed: Optional[int] = None, debug_mode: bool = False):
        super().__init__(config_serialized_str='''GymEnvironmentConfig(
            env_name='IJRR17_Theorem_14__Reach_waypoint_with_deadline',
            definitions=DefinitionsConfig(
                constants=[
                    ConstantDefinitionConfig(
                        name='ep',
                        type='Real',
                        value=None,
                        metadata={
                            'type': 'const',
                            'bound': '0.1:0.5',
                            'def': 'time limit for control decisions'
                        }
                    ),
                    ConstantDefinitionConfig(
                        name='b',
                        type='Real',
                        value=None,
                        metadata={
                            'type': 'const',
                            'bound': '0.5:5.0',
                            'def': 'minimum braking capability of the robot'
                        }
                    ),
                    ConstantDefinitionConfig(
                        name='A',
                        type='Real',
                        value=None,
                        metadata={
                            'type': 'const',
                            'bound': '1.0:10.0',
                            'def': 'maximum acceleration'
                        }
                    ),
                    ConstantDefinitionConfig(
                        name='GDelta',
                        type='Real',
                        value=None,
                        metadata={
                            'type': 'const',
                            'bound': '1.0:10.0',
                            'def': 'goal area size'
                        }
                    ),
                    ConstantDefinitionConfig(
                        name='Vmax',
                        type='Real',
                        value=None,
                        metadata={
                            'type': 'const',
                            'bound': '5.0:30.0',
                            'def': 'maximum robot velocity'
                        }
                    ),
                    ConstantDefinitionConfig(
                        name='minV',
                        type='Real',
                        value='A*ep',
                        metadata={}
                    )
                ],
                functions=[
                    FunctionDefinitionConfig(
                        name='waypointStartDist',
                        return_type='Real',
                        parameters=[
                            ('Real', 'xg')
                        ],
                        body='xg-GDelta',
                        metadata={}
                    ),
                    FunctionDefinitionConfig(
                        name='waypointEndDist',
                        return_type='Real',
                        parameters=[
                            ('Real', 'xg')
                        ],
                        body='xg+GDelta',
                        metadata={}
                    ),
                    FunctionDefinitionConfig(
                        name='maxTravelTime',
                        return_type='Real',
                        parameters=[
                            ('Real', 'g')
                        ],
                        body='waypointStartDist(g)/minV',
                        metadata={}
                    ),
                    FunctionDefinitionConfig(
                        name='stopTime',
                        return_type='Real',
                        parameters=[
                            ('Real', 'v')
                        ],
                        body='v/b',
                        metadata={}
                    ),
                    FunctionDefinitionConfig(
                        name='speedUpTime',
                        return_type='Real',
                        parameters=[
                            ('Real', 'v')
                        ],
                        body='ep-v/A',
                        metadata={}
                    ),
                    FunctionDefinitionConfig(
                        name='stopDist',
                        return_type='Real',
                        parameters=[
                            ('Real', 'v')
                        ],
                        body='v^2/(2*b)',
                        metadata={}
                    ),
                    FunctionDefinitionConfig(
                        name='accComp',
                        return_type='Real',
                        parameters=[
                            ('Real', 'v')
                        ],
                        body='( (A/b + 1)*(A/2*ep^2 + ep*v) )',
                        metadata={}
                    )
                ],
                predicates=[
                    PredicateDefinitionConfig(
                        name='bounds',
                        parameters=[],
                        body="""A > 0                                                                                                             
        & b > 0                                                                                                             
        & ep > 0                                                                                                            
        & Vmax >= 2*A*ep
        & GDelta > Vmax*ep + Vmax^2/(2*b)""",
                        metadata={}
                    ),
                    PredicateDefinitionConfig(
                        name='initialState',
                        parameters=[
                            ('Real', 'xr'),
                            ('Real', 'vr'),
                            ('Real', 'xg'),
                            ('Real', 'T')
                        ],
                        body="""vr = 0
        & xr < waypointStartDist(xg)
        & T > ep + maxTravelTime(xg-xr) + ep + stopTime(Vmax)""",
                        metadata={}
                    ),
                    PredicateDefinitionConfig(
                        name='assumptions',
                        parameters=[
                            ('Real', 'xr'),
                            ('Real', 'vr'),
                            ('Real', 'xg'),
                            ('Real', 'T')
                        ],
                        body='bounds() & initialState(xr, vr, xg, T)',
                        metadata={}
                    ),
                    PredicateDefinitionConfig(
                        name='loopinv',
                        parameters=[
                            ('Real', 'xr'),
                            ('Real', 'vr'),
                            ('Real', 'xg'),
                            ('Real', 'T')
                        ],
                        body="""0 <= vr & vr <= Vmax & xr + stopDist(vr) < waypointEndDist(xg)
                & (waypointStartDist(xg) < xr -> (vr = 0 | T >= stopTime(vr)))
                & (xr <= waypointStartDist(xg) ->
                    (vr >= minV & T > maxTravelTime(xg-xr) + ep + stopTime(Vmax))
                    | (vr <= minV & T > speedUpTime(vr) + maxTravelTime(xg-xr) + ep + stopTime(Vmax)))""",
                        metadata={
                            'type': 'safety',
                            'penalty': '-1000',
                            'on_failure': {
                                'action': 'truncate',
                                'status': 'failure'
                            }
                        }
                    ),
                    PredicateDefinitionConfig(
                        name='ensure',
                        parameters=[
                            ('Real', 'xr'),
                            ('Real', 'vr'),
                            ('Real', 'xg')
                        ],
                        body='waypointStartDist(xg) < xr & vr = 0',
                        metadata={
                            'type': 'progress',
                            'reward': '1000',
                            'on_success': {
                                'action': 'terminate',
                                'status': 'success'
                            },
                            'on_failure': {
                                'action': 'terminate',
                                'status': 'failure'
                            }
                        }
                    )
                ],
                hybrid_programs=[
                    ControlFuncConfig(
                        name='controlFunc',
                        action_space_config=None,
                        metadata={}
                    ),
                    StepFuncConfig(
                        name='stepFunc',
                        odes=[
                            ODEConfig(
                                variable='xr',
                                equation='vr'
                            ),
                            ODEConfig(
                                variable='vr',
                                equation='ar'
                            ),
                            ODEConfig(
                                variable='t',
                                equation='1'
                            ),
                            ODEConfig(
                                variable='T',
                                equation='-1'
                            )
                        ],
                        evolution_domain=EvolutionDomainConfig(
                            constraints=[
                                't <= ep',
                                'vr >= 0'
                            ]
                        ),
                        duration=None,
                        raw_body_snippet="{xr' = vr, vr' = ar, t' = 1, T'=-1 & t <= ep & vr >= 0}",
                        metadata={}
                    )
                ]
            ),
            variables=VariablesConfig(
                state_variables=[
                    VariableDeclarationConfig(
                        name='xr',
                        type='Real',
                        lower_bound='0',
                        upper_bound='100'
                    ),
                    VariableDeclarationConfig(
                        name='vr',
                        type='Real',
                        lower_bound='0',
                        upper_bound='20'
                    ),
                    VariableDeclarationConfig(
                        name='ar',
                        type='Real',
                        lower_bound='-10',
                        upper_bound='10'
                    ),
                    VariableDeclarationConfig(
                        name='xg',
                        type='Real',
                        lower_bound='0',
                        upper_bound='100'
                    ),
                    VariableDeclarationConfig(
                        name='t',
                        type='Real',
                        lower_bound='0',
                        upper_bound='1'
                    ),
                    VariableDeclarationConfig(
                        name='T',
                        type='Real',
                        lower_bound='0',
                        upper_bound='1000'
                    )
                ],
                auxiliary_variables=[]
            ),
            problem=ProblemConfig(
                precondition=PreconditionConfig(
                    formula='assumptions(xr, vr, xg, T)'
                ),
                loop_execution=LoopExecutionConfig(
                    iterations='inf',
                    loop_invariant_formula='',
                    loop_body=SequenceActionConfig(
                        actions=[]
                    )
                ),
                postcondition=PostconditionConfig(
                    formula="""(xr < waypointEndDist(xg) & (T <= 0 -> ensure(xr, vr, xg))) 
        & < dwwpdl; > ensure(xr, vr, xg)"""
                )
            )
        )''', render_mode=render_mode, cisb_size=cisb_size, cisb_populate_attempts=cisb_populate_attempts, seed=seed, debug_mode=debug_mode)

        self.constants['ep'] = 0.0
        self.constants['b'] = 0.0
        self.constants['A'] = 0.0
        self.constants['GDelta'] = 0.0
        self.constants['Vmax'] = 0.0
        self.constants['minV'] = ((self.constants['A']) * (self.constants['ep']))
        self.functions['waypointStartDist'] = lambda xg: ((xg) - (self.constants['GDelta']))
        self.functions['waypointEndDist'] = lambda xg: ((xg) + (self.constants['GDelta']))
        self.functions['maxTravelTime'] = lambda g: ((self.functions['waypointStartDist'](g)) / (self.constants['minV']))
        self.functions['stopTime'] = lambda v: ((v) / (self.constants['b']))
        self.functions['speedUpTime'] = lambda v: ((self.constants['ep']) - (((v) / (self.constants['A']))))
        self.functions['stopDist'] = lambda v: ((((v) ** (2))) / (((2) * (self.constants['b']))))
        self.functions['accComp'] = lambda v: ((((((self.constants['A']) / (self.constants['b']))) + (1))) * (((((((self.constants['A']) / (2))) * (((self.constants['ep']) ** (2))))) + (((self.constants['ep']) * (v))))))
        self.predicates['bounds'] = lambda : ((((((((((self.constants['A']) > (0))) and (((self.constants['b']) > (0))))) and (((self.constants['ep']) > (0))))) and (((self.constants['Vmax']) >= (((((2) * (self.constants['A']))) * (self.constants['ep']))))))) and (((self.constants['GDelta']) > (((((self.constants['Vmax']) * (self.constants['ep']))) + (((((self.constants['Vmax']) ** (2))) / (((2) * (self.constants['b']))))))))))
        self.predicates['initialState'] = lambda xr, vr, xg, T: ((((self._approx_eq(vr, 0)) and (((xr) < (self.functions['waypointStartDist'](xg)))))) and (((T) > (((((((self.constants['ep']) + (self.functions['maxTravelTime'](((xg) - (xr)))))) + (self.constants['ep']))) + (self.functions['stopTime'](self.constants['Vmax'])))))))
        self.predicates['assumptions'] = lambda xr, vr, xg, T: ((self.predicates['bounds']()) and (self.predicates['initialState'](xr, vr, xg, T)))
        self.predicates['loopinv'] = lambda xr, vr, xg, T: ((((((((((0) <= (vr))) and (((vr) <= (self.constants['Vmax']))))) and (((((xr) + (self.functions['stopDist'](vr)))) < (self.functions['waypointEndDist'](xg)))))) and (((not (((self.functions['waypointStartDist'](xg)) < (xr)))) or (((self._approx_eq(vr, 0)) or (((T) >= (self.functions['stopTime'](vr)))))))))) and (((not (((xr) <= (self.functions['waypointStartDist'](xg))))) or (((((((vr) >= (self.constants['minV']))) and (((T) > (((((self.functions['maxTravelTime'](((xg) - (xr)))) + (self.constants['ep']))) + (self.functions['stopTime'](self.constants['Vmax'])))))))) or (((((vr) <= (self.constants['minV']))) and (((T) > (((((((self.functions['speedUpTime'](vr)) + (self.functions['maxTravelTime'](((xg) - (xr)))))) + (self.constants['ep']))) + (self.functions['stopTime'](self.constants['Vmax'])))))))))))))
        self.predicates['ensure'] = lambda xr, vr, xg: ((((self.functions['waypointStartDist'](xg)) < (xr))) and (self._approx_eq(vr, 0)))

        self._state_keys = ['xr', 'vr', 'ar', 'xg', 't', 'T']
        self.state = {key: 0.0 for key in self._state_keys}

        low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)
        self._action_space_keys = []
        self._control_func_action_config = ChoiceActionConfig(
                level_name='Default',
                choices=[]
            )

        self._ode_state_map = {'xr': 0, 'vr': 1, 'ar': 2, 'xg': 3, 't': 4, 'T': 5}
        self._ode_state_idx_map = {0: 'xr', 1: 'vr', 2: 'ar', 3: 'xg', 4: 't', 5: 'T'}

        _ode_func_str_local = '''def ode_func(t_val, y_array):
    # Map y array to named variables for easier access
    xr = y_array[0]
    vr = y_array[1]
    ar = y_array[2]
    xg = y_array[3]
    t = y_array[4]
    T = y_array[5]

    # Calculate derivatives
    dydt = [
        vr, # dxr/dt
        ar, # dvr/dt
        0.0, # dar/dt
        0.0, # dxg/dt
        1, # dt/dt
        (-(1)), # dT/dt
    ]
    return dydt'''
        _ode_events_str_local = '''def event_t_max(t_val, y_array):
    return self.constants['ep'] - y_array[4]
event_t_max.terminal = True
event_t_max.direction = -1

ode_events = [
    event_t_max,
]'''
        self._ode_func_compiled = self._compile_ode_func(_ode_func_str_local)
        self._ode_events_compiled = self._compile_ode_events(_ode_events_str_local)

        self.sample_ranges = {
                    'xr': (-100.0, 100.0),
                    'vr': (-100.0, 100.0),
                    'ar': (-100.0, 100.0),
                    'xg': (-100.0, 100.0),
                    't': (0.0, 0.0),
                    'T': (-100.0, 100.0),
                    'ep': (0.1, 5.0),
                    'b': (0.1, 5.0),
                    'A': (0.1, 5.0),
                    'GDelta': (-100.0, 100.0),
                    'Vmax': (-100.0, 100.0),
                    'minV': (-100.0, 100.0),
                }

        self._populate_cisb_diverse_z3()

        self._force_acceleration_action = True

        self.reset()

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)

        return observation, info

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        loop_invariant_formula_kx = self._problem_config.loop_execution.loop_invariant_formula
        expanded_loop_inv = self._expand_formula(loop_invariant_formula_kx, self._definitions_config)
        loop_inv_conjuncts = self._split_top_level_and_local(expanded_loop_inv)

        loop_invariant_satisfied_pre_step = True
        for conj_str in loop_inv_conjuncts:
            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', conj_str)
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {}

            if not conjunct_result:
                loop_invariant_satisfied_pre_step = False
                if self.debug_mode:
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (pre-control): '{conj_str}'")
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                    if terminated or truncated:
                        state_record_before_early_return = self.state.copy()
                        state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                        self.state_history.append(state_record_before_early_return)
                        return self._get_obs(), reward, terminated, truncated, info

        if not loop_invariant_satisfied_pre_step:
            if self.debug_mode:
                print(f"DEBUG_STEP: General loop invariant FAILED (pre-control).")
            reward, terminated, truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-500.0', 'on_failure': 'terminate:status=failure'},
                reward, terminated, truncated, is_failure_case=True
            )
            state_record_before_early_return = self.state.copy()
            state_record_before_early_return['episode_time'] = self.cumulative_episode_time
            self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info


        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: State before control action: x={self.state['x']:.6f}, y={self.state['y']:.6f}, v={self.state['v']:.6f}, a={self.state['a']:.6f}, w={self.state['w']:.6f}, ep={self.constants['ep']:.6f}")
        control_reward, control_terminated, control_applied = self._apply_control_action(action)
        reward += control_reward
        terminated = terminated or control_terminated
        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: State after _apply_control_action (before forced override): x={self.state['x']:.6f}, y={self.state['y']:.6f}, v={self.state['v']:.6f}, a={self.state['a']:.6f}, w={self.state['w']:.6f}, control_applied={control_applied}")

        if self._force_acceleration_action:
            if self.constants['A'] <= 0.0: self.constants['A'] = 1.0
            if self.constants['W'] <= 0.0: self.constants['W'] = 1.0
            if self.constants['b'] <= 0.0: self.constants['b'] = 1.0
            if self.constants['ep'] <= 0.0: self.constants['ep'] = 1.0

            self.state['a'] = self.constants['A']
            self.state['w'] = 0.0
            if self.debug_mode:
                print(f"DEBUG_FORCED: FORCED OVERRIDE: Setting a={self.state['a']:.6f}, w={self.state['w']:.6f}")
                print(f"DEBUG_FORCED: Current constants for forced action: A={self.constants['A']:.6f}, b={self.constants['b']:.6f}, ep={self.constants['ep']:.6f}")

            control_applied = True

        if not control_applied:
            if self.debug_mode:
                print(f"DEBUG_STEP: Control action not applied. Terminating early.")
            state_record_before_early_return = self.state.copy()
            state_record_before_early_return['episode_time'] = self.cumulative_episode_time
            self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info

        y0 = np.array([self.state[var_name] for var_name in self._ode_state_map.keys()], dtype=np.float64)

        ep_val = self.constants['ep']
        t_span = (0, ep_val)

        ode_terminated_by_event = False

        if self.debug_mode:
            print(f"DEBUG_ODE: Before solve_ivp: y0={y0}, t_span={t_span}, ep_val={ep_val:.6f}")

        if ep_val <= 0:
            self.state['t'] = 0.0
            print(f"DEBUG_ODE: Warning: 'ep' constant is non-positive ({ep_val:.6f}). Skipping ODE integration and penalizing.")
            bounds_meta = self._get_predicate_metadata('bounds')
            reward, terminated, truncated = self._apply_meta_effect(
                bounds_meta or {'type': 'safety', 'penalty': '-200.0', 'on_failure': 'terminate:status=failure'},
                reward, terminated, truncated, is_failure_case=True
            )
            ode_terminated_by_event = True
        else:
            try:
                solution = solve_ivp(
                    self._ode_func_compiled,
                    t_span,
                    y0,
                    events=self._ode_events_compiled,
                    dense_output=True,
                    rtol=1e-6, atol=1e-8
                )

                if self.debug_mode:
                    print(f"DEBUG_ODE: After solve_ivp: solution.status={solution.status}, solution.message={solution.message}")
                    print(f"DEBUG_ODE: solution.t={solution.t}")
                    if solution.y.shape[1] > 0:
                        print(f"DEBUG_ODE: solution.y_initial={solution.y[:,0]}, solution.y_final={solution.y[:,-1]}")
                    if solution.t_events:
                        print(f"DEBUG_ODE: solution.t_events={solution.t_events}")

                if solution.success and solution.sol is not None:
                    actual_ode_duration = solution.t[-1] if solution.t.size > 0 else 0.0

                    if actual_ode_duration > 1e-9:
                        num_ode_points = max(20, int(actual_ode_duration * 100))
                        t_eval = np.linspace(0, actual_ode_duration, num_ode_points)

                        ode_solution_points = solution.sol(t_eval)

                        start_idx = 1 if len(self.state_history) > 0 and \
                                         np.isclose(self.state_history[-1]['episode_time'], self.cumulative_episode_time) and \
                                         np.isclose(t_eval[0], 0.0) else 0

                        for k in range(start_idx, ode_solution_points.shape[1]):
                            current_ode_state = {}
                            for i, var_name in enumerate(self._ode_state_map.keys()):
                                current_ode_state[var_name] = ode_solution_points[i, k]

                            intermediate_episode_time = self.cumulative_episode_time + current_ode_state['t']
                            current_ode_state['episode_time'] = intermediate_episode_time
                            self.state_history.append(current_ode_state)
                            if self.debug_mode and k % (ode_solution_points.shape[1] // 5 + 1) == 0:
                                print(f"DEBUG_ODE_APPEND: Appended intermediate state k={k}: x={current_ode_state['x']:.2f}, y={current_ode_state['y']:.2f}, v={current_ode_state['v']:.2f}, w={current_ode_state['w']:.2f}, a={current_ode_state['a']:.2f}, ep_time={intermediate_episode_time:.2f}")

                if solution.status == 0:
                    final_y = solution.y[:, -1]
                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]
                elif solution.status == 1:
                    if solution.y_events and solution.y_events[0].size > 0:
                        final_y = solution.y_events[0][0]
                    else:
                        final_y = solution.y[:, -1]

                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]

                    if self.state.get('v', 0.0) < -self.equality_tolerance:
                        if self.debug_mode:
                            print(f"DEBUG_ODE: Event: v became negative ({self.state['v']:.6f}).")
                        phys_valid_meta = self._get_predicate_metadata('isPhysicallyValid')
                        reward, terminated, truncated = self._apply_meta_effect(
                            phys_valid_meta or {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'truncate:status=failure'},
                            reward, terminated, truncated, is_failure_case=True
                        )
                        ode_terminated_by_event = True
                else:
                    if self.debug_mode:
                        print(f"DEBUG_ODE: ODE integration failed with status {solution.status}.")
                    reward, terminated, truncated = self._apply_meta_effect(
                        {'type': 'safety', 'penalty': '-150.0', 'on_failure': 'terminate:status=failure'},
                        reward, terminated, truncated, is_failure_case=True
                    )
                    ode_terminated_by_event = True

            except Exception as e:
                if self.debug_mode:
                    print(f"DEBUG_ODE: Exception during ODE integration: {e}")
                reward, terminated, truncated = self._apply_meta_effect(
                    {'type': 'safety', 'penalty': '-200.0', 'on_failure': 'terminate:status=failure'},
                    reward, terminated, truncated, is_failure_case=True
                )
                ode_terminated_by_event = True

        terminated = terminated or ode_terminated_by_event
        truncated = truncated or ode_terminated_by_event

        self.cumulative_episode_time += self.state['t']
        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: Updated cumulative_episode_time={self.cumulative_episode_time:.6f}")
            print(f"DEBUG_STEP {self.current_step}: State after ODE integration: x={self.state['x']:.6f}, y={self.state['y']:.6f}, v={self.state['v']:.6f}, a={self.state['a']:.6f}, w={self.state['w']:.6f}, t={self.state['t']:.6f}")


        if terminated or truncated:
            if self.debug_mode:
                print(f"DEBUG_STEP: Terminated/Truncated after ODE. Returning early.")
            if not (solution.success and solution.sol is not None and solution.t.size > 1 and actual_ode_duration > 1e-9):
                state_record_before_early_return = self.state.copy()
                state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info

        loop_invariant_satisfied_post_ode = True
        for conj_str in loop_inv_conjuncts:
            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', conj_str)
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {}

            if not conjunct_result:
                loop_invariant_satisfied_post_ode = False
                if self.debug_mode:
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (post-ODE): '{conj_str}'")
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                    if terminated or truncated:
                        state_record_before_early_return = self.state.copy()
                        state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                        self.state_history.append(state_record_before_early_return)
                        return self._get_obs(), reward, terminated, truncated, info

        if not loop_invariant_satisfied_post_ode:
            if self.debug_mode:
                print(f"DEBUG_STEP: General loop invariant FAILED (post-ODE).")
            reward, terminated, truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-300.0', 'on_failure': 'terminate:status=failure'},
                reward, terminated, truncated, is_failure_case=True
            )
            if terminated or truncated:
                state_record_before_early_return = self.state.copy()
                state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                self.state_history.append(state_record_before_early_return)
                return self._get_obs(), reward, terminated, truncated, info

        wellformed_meta = self._get_predicate_metadata('isWellformedDir')
        if wellformed_meta.get('type') == 'progress' and 'shape' in wellformed_meta:
            shape_value = self._evaluate_shape_expression(wellformed_meta['shape'])
            reward += shape_value
            if self.debug_mode:
                print(f"DEBUG_STEP: Applied isWellformedDir shape reward: {shape_value:.6f}")

        efficient_meta = self._get_predicate_metadata('isEfficient')
        if efficient_meta.get('type') == 'progress' and 'shape' in efficient_meta:
            shape_value = self._evaluate_shape_expression(efficient_meta['shape'])
            reward += shape_value
            if self.debug_mode:
                print(f"DEBUG_STEP: Applied isEfficient shape reward: {shape_value:.6f}")


        postcondition_formula_kx = self._problem_config.postcondition.formula
        postcondition_satisfied = self._evaluate_expression(postcondition_formula_kx, is_constraint=True)
        ensure_meta = self._get_predicate_metadata('ensure')

        if postcondition_satisfied:
            if self.debug_mode:
                print(f"DEBUG_STEP: Postcondition '{postcondition_formula_kx}' SATISFIED.")
            reward, terminated, truncated = self._apply_meta_effect(
                ensure_meta, reward, terminated, truncated, is_failure_case=False
            )
        else:
            if self.debug_mode:
                print(f"DEBUG_STEP: Postcondition '{postcondition_formula_kx}' NOT SATISFIED.")
            reward, terminated, truncated = self._apply_meta_effect(
                ensure_meta, reward, terminated, truncated, is_failure_case=True
            )

            if ensure_meta.get('type') == 'progress' and 'shape' in ensure_meta:
                shape_value = self._evaluate_shape_expression(ensure_meta['shape'])
                reward += shape_value
                if self.debug_mode:
                    print(f"DEBUG_STEP: Applied ensure shape reward: {shape_value:.6f}")

        if self.current_step >= self.max_episode_steps:
            truncated = True
            if self.debug_mode:
                print(f"DEBUG_STEP: Max episode steps ({self.max_episode_steps}) reached. Truncating.")
            if not terminated and not postcondition_satisfied:
                reward -= 10.0

        if not (solution.success and solution.sol is not None and solution.t.size > 1 and actual_ode_duration > 1e-9):
            state_record_after_step = self.state.copy()
            state_record_after_step['episode_time'] = self.cumulative_episode_time
            self.state_history.append(state_record_after_step)
            if self.debug_mode:
                print(f"DEBUG_STEP: Appended final state for step {self.current_step} to history.")

        observation = self._get_obs()
        info = self._get_info()
        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: Step complete. Reward={reward:.6f}, Terminated={terminated}, Truncated={truncated}")
        return observation, reward, terminated, truncated, info
