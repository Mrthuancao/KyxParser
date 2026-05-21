# GeneratedKXEnv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import re
from typing import (Dict, Any, Tuple, List, Optional, Union)
from gym_generator.kx_env_base import _KeYmaeraXEnvBase

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
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.constants = {}
        self.functions = {}
        self.predicates = {}

        self.constants['ep'] = 0.0
        self.constants['b'] = 0.0
        self.constants['A'] = 0.0
        self.constants['W'] = 0.0
        self.functions['stopDist'] = lambda v: ((((v) ** (2))) / (((2) * (self.constants['b']))))
        self.functions['accelComp'] = lambda v: ((((((self.constants['A']) / (self.constants['b']))) + (1))) * (((((((self.constants['A']) / (2))) * (((self.constants['ep']) ** (2))))) + (((self.constants['ep']) * (v))))))
        self.functions['admissibleSeparation'] = lambda v: ((self.functions['stopDist'](v)) + (self.functions['accelComp'](v)))
        self.predicates['isWellformedDir'] = lambda dx, dy: self._approx_eq(((((dx) ** (2))) + (((dy) ** (2)))), 1)
        self.predicates['isPhysicallyValid'] = lambda v, r: ((((v) >= (0))) and ((not self._approx_eq(r, 0))))
        self.predicates['isSafe'] = lambda x, y, v, xo, yo: ((((np.abs(((x) - (xo)))) > (self.functions['stopDist'](v)))) or (((np.abs(((y) - (yo)))) > (self.functions['stopDist'](v)))))
        self.predicates['isEfficient'] = lambda v: ((v) >= (0))
        self.predicates['bounds'] = lambda : ((((((self.constants['A']) >= (0))) and (((self.constants['b']) > (0))))) and (((self.constants['ep']) > (0))))
        self.predicates['initialState'] = lambda x, y, v, dx, dy, xo, yo: ((((((self._approx_eq(v, 0)) and (((((((((x) - (xo))) ** (2))) + (((((y) - (yo))) ** (2))))) > (0))))) and (self.predicates['isWellformedDir'](dx, dy)))) and ((not self._approx_eq(self.state['r'], 0))))
        self.predicates['assumptions'] = lambda x, y, v, dx, dy, xo, yo: ((self.predicates['bounds']()) and (self.predicates['initialState'](x, y, v, dx, dy, xo, yo)))
        self.predicates['loopinv'] = lambda x, y, v, dx, dy, xo, yo, r: ((((((self.predicates['isPhysicallyValid'](v, r)) and (self.predicates['isWellformedDir'](dx, dy)))) and (self.predicates['isSafe'](x, y, v, xo, yo)))) and (self.predicates['isEfficient'](v)))
        self.predicates['ensure'] = lambda x, y, xo, yo: ((((((((x) - (xo))) ** (2))) + (((((y) - (yo))) ** (2))))) > (0))

        super().__init__(config_serialized_str='''GymEnvironmentConfig(
            env_name='IJRR17_Theorem_1__Static_safety',
            definitions=DefinitionsConfig(
                constants=[
                    ConstantDefinitionConfig(
                        name='ep',
                        type='Real',
                        value='',
                        metadata={}
                    ),
                    ConstantDefinitionConfig(
                        name='b',
                        type='Real',
                        value='',
                        metadata={}
                    ),
                    ConstantDefinitionConfig(
                        name='A',
                        type='Real',
                        value='',
                        metadata={}
                    ),
                    ConstantDefinitionConfig(
                        name='W',
                        type='Real',
                        value='',
                        metadata={}
                    )
                ],
                functions=[
                    FunctionDefinitionConfig(
                        name='stopDist',
                        return_type='Real',
                        parameters=[
                            ('Real', 'v')
                        ],
                        body='v^2 / (2*b)',
                        metadata={}
                    ),
                    FunctionDefinitionConfig(
                        name='accelComp',
                        return_type='Real',
                        parameters=[
                            ('Real', 'v')
                        ],
                        body='((A/b + 1) * (A/2 * ep^2 + ep*v))',
                        metadata={}
                    ),
                    FunctionDefinitionConfig(
                        name='admissibleSeparation',
                        return_type='Real',
                        parameters=[
                            ('Real', 'v')
                        ],
                        body='stopDist(v) + accelComp(v)',
                        metadata={}
                    )
                ],
                predicates=[
                    PredicateDefinitionConfig(
                        name='isWellformedDir',
                        parameters=[
                            ('Real', 'dx'),
                            ('Real', 'dy')
                        ],
                        body='dx^2 + dy^2 = 1',
                        metadata={
                            'type': 'progress',
                            'reward': '10',
                            'shape': '-((dx**2 + dy**2) - 1)**2',
                            'on_success': 'continue:status=none',
                            'on_failure': 'continue:status=none'
                        }
                    ),
                    PredicateDefinitionConfig(
                        name='isPhysicallyValid',
                        parameters=[
                            ('Real', 'v'),
                            ('Real', 'r')
                        ],
                        body='v >= 0 & r != 0',
                        metadata={
                            'type': 'safety',
                            'penalty': '-500',
                            'on_failure': 'truncate:status=failure',
                            'on_success': 'continue:status=none'
                        }
                    ),
                    PredicateDefinitionConfig(
                        name='isSafe',
                        parameters=[
                            ('Real', 'x'),
                            ('Real', 'y'),
                            ('Real', 'v'),
                            ('Real', 'xo'),
                            ('Real', 'yo')
                        ],
                        body='abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v)',
                        metadata={
                            'type': 'safety',
                            'penalty': '-1000',
                            'shape': '(abs(x-xo) - stopDist(v)) + (abs(y-yo) - stopDist(v))',
                            'on_failure': 'truncate:status=failure',
                            'on_success': 'continue:status=none'
                        }
                    ),
                    PredicateDefinitionConfig(
                        name='isEfficient',
                        parameters=[
                            ('Real', 'v')
                        ],
                        body='v >= 0',
                        metadata={
                            'type': 'progress',
                            'shape': '1/v if v > 0 else v',
                            'on_success': 'continue:status=none',
                            'on_failure': 'continue:status=none'
                        }
                    ),
                    PredicateDefinitionConfig(
                        name='bounds',
                        parameters=[],
                        body="""A >= 0
    & b > 0
    & ep > 0""",
                        metadata={}
                    ),
                    PredicateDefinitionConfig(
                        name='initialState',
                        parameters=[
                            ('Real', 'x'),
                            ('Real', 'y'),
                            ('Real', 'v'),
                            ('Real', 'dx'),
                            ('Real', 'dy'),
                            ('Real', 'xo'),
                            ('Real', 'yo')
                        ],
                        body="""v = 0
    & (x-xo)^2 + (y-yo)^2 > 0
    & isWellformedDir(dx, dy)
    & r != 0""",
                        metadata={}
                    ),
                    PredicateDefinitionConfig(
                        name='assumptions',
                        parameters=[
                            ('Real', 'x'),
                            ('Real', 'y'),
                            ('Real', 'v'),
                            ('Real', 'dx'),
                            ('Real', 'dy'),
                            ('Real', 'xo'),
                            ('Real', 'yo')
                        ],
                        body='bounds() & initialState(x, y, v, dx, dy, xo, yo)',
                        metadata={}
                    ),
                    PredicateDefinitionConfig(
                        name='loopinv',
                        parameters=[
                            ('Real', 'x'),
                            ('Real', 'y'),
                            ('Real', 'v'),
                            ('Real', 'dx'),
                            ('Real', 'dy'),
                            ('Real', 'xo'),
                            ('Real', 'yo'),
                            ('Real', 'r')
                        ],
                        body="""isPhysicallyValid(v, r)
    & isWellformedDir(dx, dy)
    & isSafe(x, y, v, xo, yo)
    & isEfficient(v)""",
                        metadata={}
                    ),
                    PredicateDefinitionConfig(
                        name='ensure',
                        parameters=[
                            ('Real', 'x'),
                            ('Real', 'y'),
                            ('Real', 'xo'),
                            ('Real', 'yo')
                        ],
                        body='(x - xo)^2 + (y - yo)^2 > 0',
                        metadata={
                            'type': 'progress',
                            'reward': '500',
                            'penalty': '-1000',
                            'on_success': 'terminate:status=success',
                            'on_failure': 'terminate:status=failure'
                        }
                    )
                ],
                hybrid_programs=[
                    ControlFuncConfig(
                        name='controlFunc',
                        action_space_config=ChoiceActionConfig(
                            level_name='ChoiceLevel_0',
                            choices=[
                                SequenceActionConfig(
                                    actions=[
                                        DeterministicAssignmentActionConfig(
                                            variable='a',
                                            value='-b'
                                        )
                                    ]
                                ),
                                SequenceActionConfig(
                                    actions=[
                                        ActionConstraintConfig(
                                            formula='v = 0'
                                        ),
                                        DeterministicAssignmentActionConfig(
                                            variable='a',
                                            value='0'
                                        ),
                                        DeterministicAssignmentActionConfig(
                                            variable='w',
                                            value='0'
                                        )
                                    ]
                                ),
                                SequenceActionConfig(
                                    actions=[
                                        DeterministicAssignmentActionConfig(
                                            variable='a',
                                            value='A'
                                        ),
                                        NondeterministicAssignmentActionConfig(
                                            variable='w'
                                        ),
                                        ActionConstraintConfig(
                                            formula='-W<=w & w<=W'
                                        ),
                                        NondeterministicAssignmentActionConfig(
                                            variable='r'
                                        ),
                                        NondeterministicAssignmentActionConfig(
                                            variable='xo'
                                        ),
                                        NondeterministicAssignmentActionConfig(
                                            variable='yo'
                                        ),
                                        ActionConstraintConfig(
                                            formula='r!=0 & r*w = v'
                                        ),
                                        ActionConstraintConfig(
                                            formula="""abs(x-xo) > admissibleSeparation(v)
        | abs(y-yo) > admissibleSeparation(v)"""
                                        )
                                    ]
                                )
                            ]
                        ),
                        metadata={}
                    ),
                    StepFuncConfig(
                        name='stepFunc',
                        odes=[
                            ODEConfig(
                                variable='x',
                                equation='v * dx'
                            ),
                            ODEConfig(
                                variable='y',
                                equation='v * dy'
                            ),
                            ODEConfig(
                                variable='v',
                                equation='a'
                            ),
                            ODEConfig(
                                variable='dx',
                                equation='-w * dy'
                            ),
                            ODEConfig(
                                variable='dy',
                                equation='w * dx'
                            ),
                            ODEConfig(
                                variable='w',
                                equation='a/r'
                            ),
                            ODEConfig(
                                variable='t',
                                equation='1'
                            )
                        ],
                        evolution_domain=EvolutionDomainConfig(
                            constraints=[
                                't <= ep',
                                'v >= 0'
                            ]
                        ),
                        duration=None,
                        raw_body_snippet="""{
        x' = v * dx, y' = v * dy, v' = a,
        dx' = -w * dy, dy' = w * dx, w' = a/r,
        t' = 1 & t <= ep & v >= 0
      }""",
                        metadata={}
                    )
                ]
            ),
            variables=VariablesConfig(
                state_variables=[
                    VariableDeclarationConfig(
                        name='x',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='y',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='v',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='a',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='dx',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='dy',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='w',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='r',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='xo',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='yo',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    ),
                    VariableDeclarationConfig(
                        name='t',
                        type='Real',
                        lower_bound=None,
                        upper_bound=None
                    )
                ],
                auxiliary_variables=[]
            ),
            problem=ProblemConfig(
                precondition=PreconditionConfig(
                    formula='assumptions(x, y, v, dx, dy, xo, yo)'
                ),
                loop_execution=LoopExecutionConfig(
                    iterations='inf',
                    loop_invariant_formula='loopinv(x, y, v, dx, dy, xo, yo, r)',
                    loop_body=SequenceActionConfig(
                        actions=[
                            SequenceActionConfig(
                                actions=[
                                    CallHybridProgramConfig(
                                        hp_name='controlFunc'
                                    ),
                                    DeterministicAssignmentActionConfig(
                                        variable='t',
                                        value='0'
                                    )
                                ]
                            ),
                            HybridProgramStepConfig(
                                hp_name='stepFunc',
                                invariant_formula='t>=0, isWellformedDir(dx, dy)'
                            )
                        ]
                    )
                ),
                postcondition=PostconditionConfig(
                    formula='ensure(x, y, xo, yo)'
                )
            )
        )''', render_mode=render_mode)

        self._state_keys = ['x', 'y', 'v', 'a', 'dx', 'dy', 'w', 'r', 'xo', 'yo', 't']
        self.state = {key: 0.0 for key in self._state_keys}

        low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)

        self.action_space = spaces.Dict({
                'choice_ChoiceLevel_0': spaces.Discrete(3),
                'assign_w': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'assign_r': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'assign_xo': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'assign_yo': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                })
        self._action_space_keys = ['choice_ChoiceLevel_0', 'assign_w', 'assign_r', 'assign_xo', 'assign_yo']
        self._control_func_action_config = ChoiceActionConfig(
                level_name='ChoiceLevel_0',
                choices=[
                    SequenceActionConfig(
                        actions=[
                            DeterministicAssignmentActionConfig(
                                variable='a',
                                value='-b'
                            )
                        ]
                    ),
                    SequenceActionConfig(
                        actions=[
                            ActionConstraintConfig(
                                formula='v = 0'
                            ),
                            DeterministicAssignmentActionConfig(
                                variable='a',
                                value='0'
                            ),
                            DeterministicAssignmentActionConfig(
                                variable='w',
                                value='0'
                            )
                        ]
                    ),
                    SequenceActionConfig(
                        actions=[
                            DeterministicAssignmentActionConfig(
                                variable='a',
                                value='A'
                            ),
                            NondeterministicAssignmentActionConfig(
                                variable='w'
                            ),
                            ActionConstraintConfig(
                                formula='-W<=w & w<=W'
                            ),
                            NondeterministicAssignmentActionConfig(
                                variable='r'
                            ),
                            NondeterministicAssignmentActionConfig(
                                variable='xo'
                            ),
                            NondeterministicAssignmentActionConfig(
                                variable='yo'
                            ),
                            ActionConstraintConfig(
                                formula='r!=0 & r*w = v'
                            ),
                            ActionConstraintConfig(
                                formula="""abs(x-xo) > admissibleSeparation(v)
        | abs(y-yo) > admissibleSeparation(v)"""
                            )
                        ]
                    )
                ]
            )

        self._ode_state_map = {'x': 0, 'y': 1, 'v': 2, 'a': 3, 'dx': 4, 'dy': 5, 'w': 6, 'r': 7, 'xo': 8, 'yo': 9, 't': 10}
        self._ode_state_idx_map = {0: 'x', 1: 'y', 2: 'v', 3: 'a', 4: 'dx', 5: 'dy', 6: 'w', 7: 'r', 8: 'xo', 9: 'yo', 10: 't'}

        _ode_func_str_local = '''def ode_func(t_val, y_array):
    # Map y array to named variables for easier access
    x = y_array[0]
    y = y_array[1]
    v = y_array[2]
    a = y_array[3]
    dx = y_array[4]
    dy = y_array[5]
    w = y_array[6]
    r = y_array[7]
    xo = y_array[8]
    yo = y_array[9]
    t = y_array[10]

    # Ensure self.state is updated with current ODE values for expression evaluation
    for i, var_name in self._ode_state_idx_map.items():
        self.state[var_name] = y_array[i]

    # Calculate derivatives
    dydt = [
        ((v) * (dx)), # dx/dt
        ((v) * (dy)), # dy/dt
        a, # dv/dt
        0.0, # da/dt
        (((-(w))) * (dy)), # ddx/dt
        ((w) * (dx)), # ddy/dt
        ((a) / (r)), # dw/dt
        0.0, # dr/dt
        0.0, # dxo/dt
        0.0, # dyo/dt
        1, # dt/dt
    ]
    return dydt'''
        _ode_events_str_local = '''def event_t_max(t_val, y_array):
    # Event: t exceeds ep
    # Returns 0 when t == ep
    return self.constants['ep'] - y_array[10]
event_t_max.terminal = True
event_t_max.direction = -1 # Trigger when ep - t goes from positive to negative (t > ep)

def event_v_min(t_val, y_array):
    # Event: v becomes negative
    # Returns 0 when v == 0
    return y_array[2]
event_v_min.terminal = True
event_v_min.direction = -1 # Trigger when v goes from positive to negative (v < 0)

ode_events = [
    event_t_max,
    event_v_min,
]'''
        self._ode_func_compiled = self._compile_ode_func(_ode_func_str_local)
        self._ode_events_compiled = self._compile_ode_events(_ode_events_str_local)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Default initial values for state variables
        self.state['x'] = 0.0
        self.state['y'] = 0.0
        self.state['v'] = 1.0
        self.state['a'] = 0.0
        self.state['dx'] = 1.0
        self.state['dy'] = 0.0
        self.state['w'] = 0.0
        self.state['r'] = 1.0
        self.state['xo'] = 5.0
        self.state['yo'] = 0.0
        self.state['t'] = 0.0

        z3_base_solution = self._solve_initial_state_with_z3(
            self._variables_config.state_variables, 
            self._problem_config.precondition.formula
        )
        final_initial_state = None
        if z3_base_solution:
            # Z3 found a base solution. Now, perturb it to get 'new' values.
            # We will try to perturb it and verify, falling back to the base if perturbation fails.
            perturbation_magnitude = 0.05 # Small perturbation range
            num_perturbation_attempts = 5
            perturbed_state_found = False

            for _ in range(num_perturbation_attempts):
                trial_state = z3_base_solution.copy()
                for var_name, value in z3_base_solution.items():
                    if var_name in self.state:
                        perturb = self.np_random.uniform(-perturbation_magnitude, perturbation_magnitude)
                        trial_state[var_name] = value + perturb

                # Verify the perturbed state against the precondition
                # Temporarily apply trial state to self.state and self.constants for precondition check
                original_state_copy = self.state.copy()
                original_constants_copy = self.constants.copy()
                for var_name, value in trial_state.items():
                    if var_name in self.state: # Update state variables
                        self.state[var_name] = value
                    elif var_name in self.constants: # Update constants that Z3 solved for
                        self.constants[var_name] = value

                if self._evaluate_expression('assumptions(x, y, v, dx, dy, xo, yo)', is_constraint=True):
                    final_initial_state = trial_state
                    perturbed_state_found = True
                    # Revert self.state and self.constants before breaking, as final_initial_state will be applied later
                    self.state.update(original_state_copy)
                    self.constants.update(original_constants_copy)
                    break
                self.state.update(original_state_copy) # Revert if perturbation failed
                self.constants.update(original_constants_copy)

            if not perturbed_state_found:
                print(f"Warning: Could not find a valid perturbed state after {{num_perturbation_attempts}} attempts. Using base Z3 solution.")
                final_initial_state = z3_base_solution # Fallback to the unperturbed Z3 solution

            # Apply the chosen initial state (perturbed or base Z3 solution)
            for var_name, value in final_initial_state.items():
                if var_name in self.state:
                    self.state[var_name] = value
                elif var_name in self.constants:
                    self.constants[var_name] = value
            self.last_failing_precondition_formula = None
            print(f"Initial state set by Z3 (possibly perturbed): {self.state}")
            print(f"Constants possibly set by Z3: {self.constants}")
        else:
            # Z3 failed, fall back to heuristics (existing code).
            print(f"Warning: Z3 solver failed to find a satisfying initial state for the precondition: '{self._problem_config.precondition.formula}'. Falling back to heuristics. This may lead to the initial state not satisfying the precondition, which could cause immediate termination or warnings during reset. Consider revising the precondition or ensuring it is satisfiable with the given definitions.")
            if not z3_base_solution and 1 > 0:
                # Apply explicit assignments from the precondition (fallback)
                self.state['v'] = 0.0

        # Verify the full precondition after setting initial states
        # Evaluate each conjunct of the precondition to identify specific failures
        failing_formula = None
        precondition_conjuncts = self._split_top_level_and_local('((((A >= 0) & (b > 0)) & (ep > 0)) & ((((v = 0) & ((((x - xo) ^ 2) + ((y - yo) ^ 2)) > 0)) & (((dx ^ 2) + (dy ^ 2)) = 1)) & (r != 0)))')
        precondition_satisfied = True
        for conj_str in precondition_conjuncts:
            # Evaluate each conjunct. _evaluate_expression will handle the parsing.
            if not self._evaluate_expression(conj_str, is_constraint=True):
                precondition_satisfied = False
                failing_formula = conj_str
                break # Report the first failing conjunct
        if not precondition_satisfied:
            self.last_failing_precondition_formula = failing_formula
            print(f"Warning: Initial state does not satisfy precondition: 'assumptions(x, y, v, dx, dy, xo, yo)'.")
            print(f"         Failing conjunct: '{failing_formula}'.")
        else:
            self.last_failing_precondition_formula = None

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        reward, terminated, truncated = 0.0, False, False
        info = self._get_info()

        # --- Initial Loop Invariant Check ---
        expanded_loop_inv = self._expand_formula(self._problem_config.loop_execution.loop_invariant_formula)
        for conj in self._split_top_level_and_local(expanded_loop_inv):
            if not self._evaluate_expression(conj, is_constraint=True):
                m = re.match(r'^\s*(\w+)', conj)
                meta = self._get_predicate_metadata(m.group(1)) if m else {}
                reward, terminated, truncated = self._apply_meta_effect(meta or {'penalty': -100}, reward, terminated, truncated, True)
                if terminated or truncated: return self._get_obs(), reward, terminated, truncated, info

        # --- 1. Control ---
        c_reward, c_terminated, c_applied = self._apply_control_action(action)
        reward += c_reward
        if c_terminated: return self._get_obs(), reward, True, False, info

        # --- 2. Dynamics ---
        y0 = np.array([self.state[k] for k in self._ode_state_map.keys()])
        sol = solve_ivp(self._ode_func_compiled, (0, self.constants['ep']), y0, events=self._ode_events_compiled, rtol=1e-6)

        final_y = sol.y[:, -1]
        for i, name in self._ode_state_idx_map.items(): self.state[name] = final_y[i]

        if sol.status == 1: # Event triggered (e.g. v < 0)
            reward, terminated, truncated = self._apply_meta_effect({'penalty': -100}, reward, False, True, True)
            return self._get_obs(), reward, False, True, info

        # --- 3. Post-Dynamics Invariant & Shaping ---
        for conj in self._split_top_level_and_local(expanded_loop_inv):
            if not self._evaluate_expression(conj, is_constraint=True):
                return self._get_obs(), reward - 100, True, False, info

        # Reward Shaping
        for p_config in self._definitions_config.predicates:
            if p_config.metadata.get('type') == 'progress' and 'shape' in p_config.metadata:
                reward += self._evaluate_shape_expression(p_config.metadata['shape'])

        # --- 4. Postcondition ---
        if self._evaluate_expression(self._problem_config.postcondition.formula, is_constraint=True):
            meta = self._get_predicate_metadata('ensure')
            reward, terminated, truncated = self._apply_meta_effect(meta, reward, terminated, truncated, False)

        if self.current_step >= self.max_episode_steps: truncated = True

        return self._get_obs(), reward, terminated, truncated, info
