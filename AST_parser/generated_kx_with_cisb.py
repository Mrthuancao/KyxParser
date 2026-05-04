import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import math # For math functions like sqrt, sin, cos, etc.
import textwrap
import re # For regex replacements in _evaluate_expression
import z3 # For SMT solving in reset()
from typing import (Dict, Any, Tuple, List, Optional, Union) # Import typing hints
from pyast_config import (ActionType, ChoiceActionConfig, SequenceActionConfig, DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig, ActionConstraintConfig, CallHybridProgramConfig, HybridProgramStepConfig, GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig, ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig, HybridProgramConfig, ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig, VariableDeclarationConfig, PreconditionConfig, PostconditionConfig, LoopExecutionConfig) # Import AST classes

# Import matplotlib for plotting
import matplotlib.pyplot as plt
import pandas as pd # For easier data handling with matplotlib

class GeneratedKXEnv(gym.Env):
    # Added "matplotlib" to render_modes
    metadata = {"render_modes": ["human", "plot_data", "matplotlib"], "render_fps": 30}

    def __init__(self, render_mode=None, cisb_size: int = 1000, cisb_populate_attempts: int = 5000, seed: Optional[int] = None, debug_mode: bool = False):
        super().__init__()
        self.render_mode = render_mode
        self.np_random = np.random.default_rng(seed=seed)
        self.debug_mode = debug_mode # <--- NEW: Debug flag

        # --- Store original AST config objects for Z3 and other runtime needs ---
        self._definitions_config = DefinitionsConfig(
            constants=[
                ConstantDefinitionConfig(
                    name='ep',
                    type='Real',
                    value='', # Value can be empty, Z3 will solve for it or it's sampled
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
    'reward': '10.0 if v > 0 else -5.0', # Add a direct reward for positive velocity
    # 'shape': '1/v if v > 0 else v',
    'on_success': 'continue:status=none',
    'on_failure': 'continue:status=none'
}
                ),
                PredicateDefinitionConfig(
                    name='bounds',
                    parameters=[],
                    body='A >= 0\n    & b > 0\n    & ep > 0',
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
                    body='v = 0\n    & (x-xo)^2 + (y-yo)^2 > 0\n    & isWellformedDir(dx, dy)\n    & r != 0', # r is a state variable, but used in initialState. Needs to be part of the solved variables.
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
                    body='isPhysicallyValid(v, r)\n    & isWellformedDir(dx, dy)\n    & isSafe(x, y, v, xo, yo)\n    & isEfficient(v)',
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
    'on_success': 'continue:status=none',
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
                                        formula='abs(x-xo) > admissibleSeparation(v)\n        | abs(y-yo) > admissibleSeparable(v)' # Corrected typo here
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
                        constraints=['t <= ep', 'v >= 0']
                    ),
                    duration=None,
                    raw_body_snippet="{\n        x' = v * dx, y' = v * dy, v' = a,\n        dx' = -w * dy, dy' = w * dx, w' = a/r,\n        t' = 1 & t <= ep & v >= 0\n      }",
                    metadata={}
                )
            ]
        )
        self._variables_config = VariablesConfig(
            state_variables=[
                VariableDeclarationConfig(
                    name='x',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='y',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='v',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='a',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='dx',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='dy',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='w',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='r',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='xo',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='yo',
                    type='Real'
                ),
                VariableDeclarationConfig(
                    name='t',
                    type='Real'
                )
            ]
        )
        self._problem_config = ProblemConfig(
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
                formula='(x - xo)^2 + (y - yo)^2 > 0'
            )
        )

        # --- Semantic Gap Bridging: Floating Point Equality Tolerance ---
        self.equality_tolerance = 1e-9 # Default absolute tolerance for np.isclose
        self.z3_equality_tolerance = 0.05 # Increased tolerance for Z3's approximate equality for '=' and '!='

        # --- KeYmaera X Definitions ---
        self.constants = {}
        self.functions = {}
        self.predicates = {}

        # Constants (initialized to 0.0 as placeholders; will be set by CISB or Z3)
        self.constants['ep'] = 0.0
        self.constants['b'] = 0.0
        self.constants['A'] = 0.0
        self.constants['W'] = 0.0

        # Functions (lambdas for easy evaluation)
        # Note: These lambdas capture 'self' at definition time, so they will use the current self.constants values.
        self.functions['stopDist'] = lambda v_val: v_val**2 / (2*self.constants['b']) if self.constants['b'] != 0 else float('inf') # Handle division by zero
        self.functions['accelComp'] = lambda v_val: (self.constants['A']/self.constants['b'] + 1) * (self.constants['A']/2 * self.constants['ep']**2 + self.constants['ep']*v_val) if self.constants['b'] != 0 else float('inf') # Handle division by zero
        self.functions['admissibleSeparation'] = lambda v_val: self.functions['stopDist'](v_val) + self.functions['accelComp'](v_val)

        # Predicates (lambdas for easy evaluation)
        # Note: The 'r' in initialState body is a state variable, it must be accessed via self.state['r']
        self.predicates['isWellformedDir'] = lambda dx, dy: self._approx_eq(dx**2 + dy**2, 1)
        self.predicates['isPhysicallyValid'] = lambda v, r: ((v) >= (0)) and ((r) != (0))
        self.predicates['isSafe'] = lambda x, y, v, xo, yo: ((np.abs(x-xo)) > (self.functions['stopDist'](v))) or ((np.abs(y-yo)) > (self.functions['stopDist'](v)))
        self.predicates['isEfficient'] = lambda v: (v) >= (0)
        self.predicates['bounds'] = lambda : ((self.constants['A']) >= (0)) and (((self.constants['b']) > (0)) and ((self.constants['ep']) > (0)))
        # The 'r' in initialState body is a state variable, it must be accessed via self.state['r']
        self.predicates['initialState'] = lambda x, y, v, dx, dy, xo, yo: (self._approx_eq(v, 0)) and ((((x-xo)**2 + (y-yo)**2) > (0)) and ((self.predicates['isWellformedDir'](dx, dy)) and ((self.state['r']) != (0))))
        self.predicates['assumptions'] = lambda x, y, v, dx, dy, xo, yo: (self.predicates['bounds']()) and (self.predicates['initialState'](x, y, v, dx, dy, xo, yo))
        self.predicates['loopinv'] = lambda x, y, v, dx, dy, xo, yo, r_val: (self.predicates['isPhysicallyValid'](v, r_val)) and ((self.predicates['isWellformedDir'](dx, dy)) and ((self.predicates['isSafe'](x, y, v, xo, yo)) and (self.predicates['isEfficient'](v))))
        self.predicates['ensure'] = lambda x, y, xo, yo: ((x - xo)**2 + (y - yo)**2) > (0)

        # --- State Variables ---
        # Mapping of state variable names to their index in the observation array
        self._state_keys = ['x', 'y', 'v', 'a', 'dx', 'dy', 'w', 'r', 'xo', 'yo', 't']
        # Current state values (dictionary for easy access by name)
        # Initialize to 0.0 as placeholders; these will be overwritten by reset()
        self.state = {key: 0.0 for key in self._state_keys}
        self.last_failing_precondition_formula = None # Initialize for debugging

        # --- Observation Space ---
        # All state variables are Real, so Box space is appropriate.
        # Bounds are infinite by default, as not specified in KeYmaera X input.
        low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)
        
        # Define reasonable bounds for random sampling for state variables and constants.
        # These ranges should reflect the expected operational domain of the system.
        # Adjust these ranges based on the specific problem.
        self.sample_ranges = {
            'x': (-100.0, 100.0), 'y': (-100.0, 100.0), 'v': (0.0, 0.0), # v is fixed to 0 by precondition
            'dx': (-1.0, 1.0), 'dy': (-1.0, 1.0),
            'w': (-2.0, 2.0), # w is not fixed by precondition, can be anything
            'r': (0.1, 10.0), # r must be non-zero by precondition
            'xo': (-100.0, 100.0), 'yo': (-100.0, 100.0),
            # Constants that Z3 might solve for, or need to be within bounds
            'ep': (0.1, 5.0), # ep > 0 by bounds()
            'b': (0.1, 5.0),  # b > 0 by bounds()
            'A': (0.1, 5.0),  # A >= 0 by bounds()
            'W': (0.0, 2.0)   # W >= 0 (often implicit for magnitude)
        }

        # --- Action Space (from controlFunc) ---
        # The action space is a Dict to handle choices and nondeterministic assignments.
        self.action_space = spaces.Dict({
                'choice_ChoiceLevel_0': spaces.Discrete(3),
                'assign_w': spaces.Box(low=self.sample_ranges['w'][0], high=self.sample_ranges['w'][1], shape=(), dtype=np.float32),
                'assign_r': spaces.Box(low=self.sample_ranges['r'][0], high=self.sample_ranges['r'][1], shape=(), dtype=np.float32),
                'assign_xo': spaces.Box(low=self.sample_ranges['xo'][0], high=self.sample_ranges['xo'][1], shape=(), dtype=np.float32),
                'assign_yo': spaces.Box(low=self.sample_ranges['yo'][0], high=self.sample_ranges['yo'][1], shape=(), dtype=np.float32),
                })
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
                                formula='abs(x-xo) > admissibleSeparation(v)\n        | abs(y-yo) > admissibleSeparation(v)' # Corrected typo here
                            )
                        ]
                    )
                ]
            ) # Stored for step logic


        # --- ODE Solver Setup (from stepFunc) ---
        # Mapping from state variable names to their index in the ODE solver's y array
        self._ode_state_map = {'x': 0, 'y': 1, 'v': 2, 'a': 3, 'dx': 4, 'dy': 5, 'w': 6, 'r': 7, 'xo': 8, 'yo': 9, 't': 10}
        self._ode_state_idx_map = {0: 'x', 1: 'y', 2: 'v', 3: 'a', 4: 'dx', 5: 'dy', 6: 'w', 7: 'r', 8: 'xo', 9: 'yo', 10: 't'} # Map index to name

        self._ode_func_str = '''
            def ode_func(t_val, y_array):
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

                # Calculate derivatives
                # Check for division by zero for 'a/r'
                if r == 0:
                    dwdt = 0.0 # Or some other appropriate handling for r=0
                else:
                    dwdt = a/r

                dydt = [
                    v * dx, # dx/dt
                    v * dy, # dy/dt
                    a, # dv/dt
                    0.0, # da/dt (a is a control input, not dynamically changing here)
                    -w * dy, # ddx/dt
                    w * dx, # ddy/dt
                    dwdt, # dw/dt
                    0.0, # dr/dt (r is a control input, not dynamically changing here)
                    0.0, # dxo/dt (xo is an obstacle position, not dynamically changing here)
                    0.0, # dyo/dt (yo is an obstacle position, not dynamically changing here)
                    1.0, # dt/dt
                ]
                return dydt
        '''
        self._ode_events_str = '''
            def event_t_max(t_val, y_array):
                # Event: t exceeds ep
                # Returns 0 when t == ep
                # y_array[10] is 't'
                # self.constants['ep'] is 'ep'
                return self.constants['ep'] - y_array[10]
            event_t_max.terminal = True
            event_t_max.direction = -1 # Trigger when ep - t goes from positive to negative (t > ep)

            def event_v_min(t_val, y_array):
                # Event: v becomes negative
                # Returns 0 when v == 0
                # y_array[2] is 'v'
                # y_array[3] is 'a' (acceleration)
                v_val = y_array[2]
                a_val = y_array[3]

                # If v is exactly 0 and acceleration is positive,
                # the event (v crossing 0 from positive to negative) should not trigger.
                # We return a non-zero value to prevent premature termination.
                # solve_ivp seems to trigger direction=-1 events at t=0 if the value is exactly 0,
                # even if the derivative is positive.
                if v_val == 0 and a_val > 0:
                    return 1.0 # Arbitrary non-zero value to prevent trigger
                return v_val
            event_v_min.terminal = True
            event_v_min.direction = -1 # Trigger when v goes from positive to negative (v < 0)

            ode_events = [
                event_t_max,
                event_v_min,
            ]
        '''

        # Compile ODE function and events dynamically
        self._ode_func_compiled = self._compile_ode_func(self._ode_func_str)
        self._ode_events_compiled = self._compile_ode_events(self._ode_events_str)

        # --- Episode Management ---
        self.current_step = 0
        self.max_episode_steps = 400 # Default max steps for an episode
        self.reward_range = (-float('inf'), float('inf'))
        self.state_history = [] # Added: List to store state at each step for plotting
        self.cumulative_episode_time = 0.0 # NEW: Cumulative time for the entire episode
        self.last_episode_history = [] # NEW: To store history of the just completed episode

        # --- Contrastive Initial State Buffer (CISB) ---
        self._cisb = []
        self._cisb_size = cisb_size
        self._cisb_populate_attempts = cisb_populate_attempts


        # Call the new diverse CISB population method
        self._populate_cisb_diverse_z3() 

        # The first reset will be called by the Gymnasium API, no need to call it here.

        # Flag for temporary forced acceleration action for demonstration
        self._force_acceleration_action = True # <--- SET THIS TO TRUE TO SEE MOVEMENT IN PLOTS

    def _compile_ode_func(self, func_str):
        # Create a dictionary for globals that includes 'self'
        exec_globals = {'np': np, 'math': math, 'self': self} # 'self' in globals
        exec_locals = {} # No special locals needed, self is in globals
        # Dedent the string before execution, as it was indented for the triple-quote literal
        exec(textwrap.dedent(func_str), exec_globals, exec_locals)
        return exec_locals['ode_func']

    def _compile_ode_events(self, events_str):
        if not events_str.strip():
            return []
        exec_globals = {'np': np, 'math': math, 'self': self} # 'self' in globals
        exec_locals = {} # No special locals needed, self is in globals
        # Dedent the string before execution
        exec(textwrap.dedent(events_str), exec_globals, exec_locals)
        return exec_locals['ode_events']

    def _get_obs(self):
        return np.array([self.state[key] for key in self._state_keys], dtype=np.float32)

    def _get_info(self):
        return self.state.copy() # Return a copy of the current state dictionary

    def _approx_eq(self, a, b):
        """
        Performs an approximate equality check using numpy.isclose.
        This bridges the semantic gap between formal exact equality and float approximation.
        """
        return np.isclose(a, b, atol=self.equality_tolerance)

    def _expand_formula(self, formula_str: str, definitions: DefinitionsConfig) -> str:
        """
        Recursively expands predicate calls within a formula string.
        This returns the formula in KeYmaera X syntax, suitable for further parsing.
        It handles simple argument substitution.
        """
        expanded_formula = formula_str

        predicate_map = {p.name: (p.body, [param[1] for param in p.parameters]) for p in definitions.predicates}

        while True:
            num_substitutions_in_pass = 0
            # Sort by length descending to handle cases where one predicate name is a substring of another
            sorted_pred_names = sorted(predicate_map.keys(), key=len, reverse=True)

            for pred_name in sorted_pred_names:
                pred_body, pred_params = predicate_map[pred_name]
                # Regex to find calls to this specific predicate, capturing arguments
                pattern = r'\b' + re.escape(pred_name) + r'\s*\(([^)]*)\)'

                def replace_match(match):
                    nonlocal num_substitutions_in_pass
                    num_substitutions_in_pass += 1
                    call_args_str = match.group(1)
                    call_args = [arg.strip() for arg in call_args_str.split(',') if arg.strip()]

                    if len(call_args) != len(pred_params):
                        # Mismatch in args, return original call for now.
                        return match.group(0)

                    substituted_body = pred_body
                    for i, param_name in enumerate(pred_params):
                        # Replace `param_name` with `call_args[i]` in the body
                        # Use word boundaries to avoid partial replacements (e.g., replacing 'x' in 'dx')
                        substituted_body = re.sub(r'\b' + re.escape(param_name) + r'\b', call_args[i], substituted_body)

                    return f"({substituted_body})" # Enclose in parens to maintain logical grouping

                expanded_formula = re.sub(pattern, replace_match, expanded_formula)

            if num_substitutions_in_pass == 0:
                break

        return expanded_formula

    def _get_predicate_metadata(self, predicate_name: str) -> Dict[str, Any]:
        """Retrieves metadata for a given predicate name."""
        for pred_config in self._definitions_config.predicates:
            if pred_config.name == predicate_name:
                return pred_config.metadata
        return {} # Return empty dict if not found

    def _apply_meta_effect(self, metadata: Dict[str, Any], current_reward: float, current_terminated: bool, current_truncated: bool, is_failure_case: bool) -> Tuple[float, bool, bool]:
        """
        Applys reward/penalty/termination/truncation effects based on metadata.
        'on_success': 'action:status=value' (e.g., 'terminate:status=success', 'continue:status=none')
        'on_failure': 'action:status=value' (e.g., 'truncate:status=failure', 'terminate:status=failure')

        action can be 'continue', 'terminate', 'truncate'.
        status can be 'success', 'failure', 'none'.

        is_failure_case: True if this call is due to a condition failing, False if it's a success path.
        """
        reward_change = 0.0
        new_terminated = current_terminated
        new_truncated = current_truncated

        if metadata:
            event_key = 'on_failure' if is_failure_case else 'on_success'
            event_config_str = metadata.get(event_key)

            if is_failure_case:
                penalty = float(metadata.get('penalty', 0.0))
                reward_change += penalty
            else: # Success case
                success_reward = float(metadata.get('reward', 0.0))
                reward_change += success_reward

            if event_config_str:
                parts = event_config_str.split(':')
                action = parts[0]
                # status_part = parts[1] if len(parts) > 1 else 'status=none'
                # status = status_part.split('=')[1] # Not directly used for logic, but could be for info

                if action == 'terminate':
                    new_terminated = True
                elif action == 'truncate':
                    new_truncated = True
                # 'continue' means no change to terminated/truncated state, so no 'else' needed here.

        return current_reward + reward_change, new_terminated, new_truncated

    def _evaluate_shape_expression(self, shape_expr: str) -> float:
        """
        Evaluates a shape expression from metadata.
        Shape expressions might use KeYmaera X ternary operators (e.g., '1/v if v > 0 else v').
        """
        if not shape_expr:
            return 0.0

        # Handle KeYmaera X style ternary operator 'value_if_true if condition else value_if_false'
        # Regex: (.+) if (.+) else (.+)
        match = re.match(r'(.+)\s+if\s+(.+)\s+else\s+(.+)', shape_expr)
        if match:
            true_val_str, condition_str, false_val_str = match.groups()
            try:
                condition_result = self._evaluate_expression(condition_str)
                if condition_result:
                    return float(self._evaluate_expression(true_val_str))
                else:
                    return float(self._evaluate_expression(false_val_str))
            except Exception as e:
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_EVAL: Warning: Error evaluating shape ternary expression '{shape_expr}': {e}")
                return 0.0
        else:
            # If not a ternary, just evaluate as a regular expression
            try:
                return float(self._evaluate_expression(shape_expr))
            except Exception as e:
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_EVAL: Warning: Error evaluating shape expression '{shape_expr}': {e}")
                return 0.0


    def _kx_expr_to_z3_formula(self, kx_expr: str, z3_symbol_map: Dict[str, Any], fixed_constants: Dict[str, Union[int, float]]) -> Any:
        """
        Converts a KeYmaera X expression string into a Z3 formula.
        This function performs string replacements to convert KeYmaera X syntax
        to a Python string that, when evaluated, constructs a Z3 expression.
        It processes operators respecting precedence: arithmetic, relational, then logical.
        For logical operators, it uses a recursive approach to handle nested parentheses.
        """
        expr_transformed = kx_expr.strip()

        # Combine predicates and functions for expansion
        combined_definitions = DefinitionsConfig(
            constants=self._definitions_config.constants,
            functions=self._definitions_config.functions, # Include functions
            predicates=self._definitions_config.predicates # Include predicates
        )

        # A helper for expanding functions/predicates for Z3
        def _expand_for_z3(formula_to_expand: str, definitions_config: DefinitionsConfig) -> str:
            expanded = formula_to_expand
            # Map of name -> (body, params) for both functions and predicates
            expandables = {}
            for f in definitions_config.functions:
                expandables[f.name] = (f.body, [p[1] for p in f.parameters])
            for p in definitions_config.predicates:
                expandables[p.name] = (p.body, [p[1] for p in p.parameters])

            while True:
                num_substitutions_in_pass = 0
                sorted_names = sorted(expandables.keys(), key=len, reverse=True)

                for name in sorted_names:
                    body, params = expandables[name]
                    pattern = r'\b' + re.escape(name) + r'\s*\(([^)]*)\)'

                    def replace_match_z3(match):
                        nonlocal num_substitutions_in_pass
                        num_substitutions_in_pass += 1
                        call_args_str = match.group(1)
                        call_args = [arg.strip() for arg in call_args_str.split(',') if arg.strip()]

                        if len(call_args) != len(params):
                            # Mismatch in args, return original call for Z3 to potentially fail later
                            return match.group(0)

                        substituted_body = body
                        for i, param_name in enumerate(params):
                            substituted_body = re.sub(r'\b' + re.escape(param_name) + r'\b', call_args[i], substituted_body)
                        return f"({substituted_body})" # Enclose in parens

                    expanded = re.sub(pattern, replace_match_z3, expanded)
                if num_substitutions_in_pass == 0:
                    break
            return expanded

        # First, expand all function and predicate calls within the expression
        expr_transformed = _expand_for_z3(expr_transformed, combined_definitions)

        # 1. Replace identifiers with their Z3 symbol map reference.
        # Fixed constants will be handled by being added to eval_globals as Z3 literals.
        # Sort by length descending to handle cases where one identifier is a substring of another (e.g., 'dx' before 'x')
        all_known_kx_identifiers = list(z3_symbol_map.keys()) + \
                                   list(fixed_constants.keys())

        sorted_identifiers = sorted(all_known_kx_identifiers, key=len, reverse=True)

        for ident_name in sorted_identifiers:
            replacement = ""
            if ident_name in z3_symbol_map:
                replacement = f"z3_symbol_map['{ident_name}']"
            elif ident_name in fixed_constants:
                replacement = ident_name # Will be available in eval_globals

            if replacement:
                # Use word boundaries to avoid partial replacements (e.g., 'x' in 'dx')
                expr_transformed = re.sub(r'\b' + re.escape(ident_name) + r'\b', replacement, expr_transformed)

        # 2. Basic arithmetic and math functions
        expr_transformed = expr_transformed.replace('^', '**')
        expr_transformed = expr_transformed.replace('abs(', 'z3.Abs(') # Use z3.Abs for Z3 context

        # 3. Handle logical and relational operators respecting precedence and parentheses
        # This recursive function will parse the logical and relational structure.
        def _parse_logical_and_relational_expression_z3(expression_str: str) -> str:
            expression_str = expression_str.strip()
            if not expression_str:
                return ""

            # 1. Strip outermost balanced parentheses
            while expression_str.startswith('(') and expression_str.endswith(')'):
                balance = 0
                is_fully_enclosed = True
                for i in range(len(expression_str) - 1):
                    char = expression_str[i]
                    if char == '(': balance += 1
                    elif char == ')': balance -= 1
                    if balance == 0:
                        is_fully_enclosed = False
                        break
                if is_fully_enclosed:
                    expression_str = expression_str[1:-1].strip()
                else:
                    break

            # --- Logical Operators (highest level of this parser) ---

            # Implication (lowest logical precedence, right-associative)
            balance = 0
            implication_idx = -1
            for i in range(len(expression_str) - 2, -1, -1): # Scan right-to-left for implication
                char = expression_str[i]
                if char == ')': balance += 1
                elif char == '(': balance -= 1
                elif balance == 0 and expression_str[i:i+2] == '->':
                    implication_idx = i
                    break
            if implication_idx != -1:
                left = _parse_logical_and_relational_expression_z3(expression_str[:implication_idx].strip())
                right = _parse_logical_and_relational_expression_z3(expression_str[implication_idx+2:].strip())
                return f"z3.Implies({left}, {right})"

            # Disjunction (medium precedence, left-associative)
            balance = 0
            disjunction_idx = -1
            for i in range(len(expression_str)): # Scan left-to-right for disjunction
                char = expression_str[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0 and char == '|':
                    disjunction_idx = i
                    break
            if disjunction_idx != -1:
                left = _parse_logical_and_relational_expression_z3(expression_str[:disjunction_idx].strip())
                right = _parse_logical_and_relational_expression_z3(expression_str[disjunction_idx+1:].strip())
                return f"z3.Or({left}, {right})"

            # Conjunction (highest precedence, left-associative)
            balance = 0
            conjunction_idx = -1
            for i in range(len(expression_str)): # Scan left-to-right for conjunction
                char = expression_str[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0 and char == '&':
                    conjunction_idx = i
                    break
            if conjunction_idx != -1:
                left = _parse_logical_and_relational_expression_z3(expression_str[:conjunction_idx].strip())
                right = _parse_logical_and_relational_expression_z3(expression_str[conjunction_idx+1:].strip())
                return f"z3.And({left}, {right})"

            # Logical NOT (prefix operator)
            if expression_str.startswith('!'):
                operand = _parse_logical_and_relational_expression_z3(expression_str[1:].strip())
                return f"z3.Not({operand})"

            # --- Relational Operators (next level down) ---
            # Note: '=' in KeYmaera X maps to '==' for Z3 exact equality, but we want approximate
            relational_ops = [('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>'), ('=', '==')]

            balance = 0
            rel_op_found_kx = None
            rel_op_found_py = None # This will be the Z3 equivalent string
            rel_op_idx = -1

            # Scan left-to-right for relational operators
            for i in range(len(expression_str)):
                char = expression_str[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0:
                    # Check for multi-character operators first
                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) > 1 and expression_str[i:i+len(kx_op)] == kx_op:
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break

                    # Check for single-character operators if no multi-char op found yet
                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) == 1 and expression_str[i] == kx_op:
                            # Special check for '=', ensure it's not part of '==', '!=', '<=', '>='
                            if kx_op == '=' and ((i > 0 and expression_str[i-1] in '<>=!') or \
                                            (i < len(expression_str) - 1 and expression_str[i+1] == '=')):
                                continue
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break

            if rel_op_found_kx:
                left = _parse_logical_and_relational_expression_z3(expression_str[:rel_op_idx].strip())
                right = _parse_logical_and_relational_expression_z3(expression_str[rel_op_idx+len(rel_op_found_kx):].strip())
                if rel_op_found_kx == '=':
                    # For Z3, KeYmaera X's '=' should be treated with a tolerance similar to _approx_eq
                    return f"z3.And({left} >= ({right}) - self.z3_equality_tolerance, {left} <= ({right}) + self.z3_equality_tolerance)"
                elif rel_op_found_kx == '!=':
                    # For Z3, KeYmaera X's '!=' should be the negation of approximate equality
                    return f"z3.Not(z3.And({left} >= ({right}) - self.z3_equality_tolerance, {left} <= ({right}) + self.z3_equality_tolerance))"
                else:
                    # Other relational operators are direct
                    return f"({left}) {rel_op_found_py} ({right})"


            # Base case: no logical or relational operators found, return the expression as is.
            # This should now be an arithmetic expression or a literal.
            return expression_str

        # Apply the combined logical and relational expression parser
        expr_transformed = _parse_logical_and_relational_expression_z3(expr_transformed)

        # Final evaluation in a Z3-aware context
        eval_globals = {'z3': z3, 'z3_symbol_map': z3_symbol_map, 'self': self} # Pass self for z3_equality_tolerance
        for const_name, const_value in fixed_constants.items():
            if isinstance(const_value, int):
                eval_globals[const_name] = z3.IntVal(const_value)
            else: # float or other numeric
                eval_globals[const_name] = z3.RealVal(const_value)

        try:
            return eval(expr_transformed, eval_globals)
        except Exception as e:
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_EVAL: Failed to evaluate Z3 expression '{kx_expr}' (transformed: '{expr_transformed}'): {e}")
            raise ValueError(f"Failed to evaluate Z3 expression '{kx_expr}' (transformed: '{expr_transformed}'): {e}")


    def _evaluate_expression(self, expr_str: str, local_vars: Optional[Dict[str, Any]] = None, is_constraint: bool = False) -> Any:
        """
        Safe evaluation of expressions using environment's context.
        This method parses KeYmaera X expressions at runtime.
        If is_constraint is True, returns False on evaluation error.
        """

        if local_vars == None:
            local_vars = {}

        # Strip KeYmaera X constraint prefix '?' if present
        if expr_str.startswith('?'):
            expr_str = expr_str[1:].strip()

        # --- NEW PRE-PROCESSING STEP: Prefix identifiers ---
        # This step converts bare KeYmaera X identifiers (like 'x', 'A', 'stopDist')
        # into Python-compatible references (like 'self.state["x"]', 'self.constants["A"]', 'self.functions["stopDist"]')
        # before the expression is further parsed and evaluated.

        processed_expr = expr_str.strip()

        # Combine all known identifiers from the environment that need prefixing
        # Exclude 'np' and 'math' as they are global modules and should not be prefixed.
        all_env_identifiers = list(self.state.keys()) + \
                               list(self.constants.keys()) + \
                               list(self.functions.keys()) + \
                               list(self.predicates.keys())

        # Sort by length descending to handle cases where one identifier is a substring of another (e.g., 'dx' before 'x')
        all_env_identifiers_sorted = sorted(all_env_identifiers, key=len, reverse=True)

        # Create a temporary list of identifiers that are *local* to this evaluation (e.g., lambda parameters passed in local_vars)
        # These should NOT be prefixed.
        local_identifiers_in_scope = list(local_vars.keys()) if local_vars else []

        # Perform the prefixing
        for ident_name in all_env_identifiers_sorted:
            if ident_name in local_identifiers_in_scope:
                continue # Skip if it's a local variable (e.g., 'v' in 'stopDist(v)' if 'v' is in local_vars)

            replacement = ""
            if ident_name in self.state:
                replacement = f"self.state['{ident_name}']"
            elif ident_name in self.constants:
                replacement = f"self.constants['{ident_name}']"
            elif ident_name in self.functions:
                replacement = f"self.functions['{ident_name}']"
            elif ident_name in self.predicates:
                replacement = f"self.predicates['{ident_name}']"

            if replacement:
                # Use regex with word boundaries to ensure we replace whole identifiers only
                processed_expr = re.sub(r'\b' + re.escape(ident_name) + r'\b', replacement, processed_expr)

        # --- END NEW PRE-PROCESSING STEP ---

        expr_py_transformed = processed_expr # Now use the processed_expr

        # 1. Basic syntax replacements (power, abs)
        expr_py_transformed = expr_py_transformed.replace('^', '**').replace('abs(', 'np.abs(')

        # 2. Recursive parsing for operator precedence (relational then logical)
        def _parse_expression_recursive_runtime(sub_expr: str) -> str:
            sub_expr = sub_expr.strip()
            if not sub_expr:
                return ""

            # Handle outermost balanced parentheses
            if sub_expr.startswith('(') and sub_expr.endswith(')'):
                balance = 0
                is_fully_enclosed = True
                for i in range(len(sub_expr) - 1): # Check up to second to last char
                    char = sub_expr[i]
                    if char == '(': balance += 1
                    elif char == ')': balance -= 1
                    if balance == 0:
                        is_fully_enclosed = False
                        break
                if is_fully_enclosed:
                    return _parse_expression_recursive_runtime(sub_expr[1:-1]) # Recursively parse inner part

            # Find lowest precedence operator at top level

            # Implication (lowest logical precedence) - right-associative
            balance = 0
            implication_idx = -1
            for i in range(len(sub_expr) - 2, -1, -1): # Scan right-to-left
                char_pair = sub_expr[i:i+2]
                if char_pair == '->' and balance == 0:
                    implication_idx = i
                    break
                if sub_expr[i] == ')': balance += 1
                elif sub_expr[i] == '(': balance -= 1

            if implication_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:implication_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[implication_idx+2:].strip())
                return f"((not ({left})) or ({right}))"

            # Disjunction (next lowest logical) - left-associative
            balance = 0
            disjunction_idx = -1
            for i in range(len(sub_expr)): # Scan left-to-right
                if sub_expr[i] == '|' and balance == 0:
                    disjunction_idx = i
                    break
                if sub_expr[i] == '(': balance += 1
                elif sub_expr[i] == ')': balance -= 1

            if disjunction_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:disjunction_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[disjunction_idx+1:].strip())
                return f"({left}) or ({right})"

            # Conjunction (highest precedence, left-associative)
            balance = 0
            conjunction_idx = -1
            for i in range(len(sub_expr)): # Scan left-to-right
                if sub_expr[i] == '&' and balance == 0:
                    conjunction_idx = i
                    break
                if sub_expr[i] == '(': balance += 1
                elif sub_expr[i] == ')': balance -= 1

            if conjunction_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:conjunction_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[conjunction_idx+1:].strip())
                return f"({left}) and ({right})"

            # Negation (highest logical precedence) - prefix operator
            if sub_expr.startswith('!'):
                operand = _parse_expression_recursive_runtime(sub_expr[1:].strip())
                return f"(not ({operand}))"

            # Relational operators (including '=')
            relational_ops = [('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>'), ('=', '_approx_eq')]

            balance = 0
            rel_op_found_kx = None
            rel_op_found_py = None
            rel_op_idx = -1

            # Scan left-to-right for relational operators
            for i in range(len(sub_expr)):
                char = sub_expr[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0:
                    # Check for multi-character operators first
                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) > 1 and sub_expr[i:i+len(kx_op)] == kx_op:
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break

                    # Check for single-character operators if no multi-char op found yet
                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) == 1 and sub_expr[i] == kx_op:
                            # Special check for '=', ensure it's not part of '==', '!=', '<=', '>='
                            if kx_op == '=' and ((i > 0 and sub_expr[i-1] in '<>=!') or \
                                            (i < len(sub_expr) - 1 and sub_expr[i+1] == '=')):
                                continue
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break

            if rel_op_found_kx:
                left = _parse_expression_recursive_runtime(sub_expr[:rel_op_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[rel_op_idx+len(rel_op_found_kx):].strip())
                if rel_op_found_kx == '=':
                    return f"{rel_op_found_py}({left}, {right})"
                else:
                    return f"({left}) {rel_op_found_py} ({right})"

            # If no logical or relational operators, it's an arithmetic expression, literal, or function/predicate call
            return sub_expr

        expr_py_final = _parse_expression_recursive_runtime(expr_py_transformed)
        # --- End KeYmaera X to Python conversion logic ---

        try:
            # With explicit prefixing, eval_locals can be much simpler.
            # Only local_vars are truly 'local' to the eval call.
            # All other references are now explicit (self.state['x'], self.constants['A'], etc.)
            eval_globals_for_eval = {'np': np, 'math': math, '_approx_eq': self._approx_eq, 'self': self}
            result = eval(expr_py_final, eval_globals_for_eval, local_vars)
            return result
        except Exception as e:
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_EVAL: Error evaluating expression '{expr_str}' (transformed: '{expr_py_final}'): {e}")
            if is_constraint:
                return False # For constraints, return False on error
            raise # Re-raise for non-constraint expressions


    def _execute_action_config(self, action_config: ActionType, agent_action: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """
        Recursively executes a given action configuration (Choice, Sequence, Assignment, Constraint).
        Returns (reward, terminated, applied_successfully).
        """
        reward = 0.0
        terminated = False
        applied_successfully = True

        if isinstance(action_config, ChoiceActionConfig):
            choice_key = f"choice_{action_config.level_name}"
            if choice_key not in agent_action:
                print(f"DEBUG_ACTION: Error: Choice key '{choice_key}' missing from agent action.") # Not wrapped (critical)
                reward -= 200.0
                terminated = True
                applied_successfully = False
                return reward, terminated, applied_successfully

            choice_idx = agent_action[choice_key]
            if not (0 <= choice_idx < len(action_config.choices)):
                print(f"DEBUG_ACTION: Error: Invalid choice index {choice_idx} for '{choice_key}'.") # Not wrapped (critical)
                reward -= 200.0
                terminated = True
                applied_successfully = False
                return reward, terminated, applied_successfully

            chosen_sub_config = action_config.choices[choice_idx]
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_ACTION: Chosen action branch: {choice_idx}")
            return self._execute_action_config(chosen_sub_config, agent_action)

        elif isinstance(action_config, SequenceActionConfig):
            for sub_action in action_config.actions:
                r, t, s = self._execute_action_config(sub_action, agent_action)
                reward += r
                if t: terminated = True
                if not s:
                    applied_successfully = False
                    break # Stop processing sequence if any sub-action fails
            return reward, terminated, applied_successfully

        elif isinstance(action_config, DeterministicAssignmentActionConfig):
            try:
                # Value is a raw KeYmaera X string, parsed at runtime by _evaluate_expression
                assigned_value = self._evaluate_expression(action_config.value)
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_ACTION: Assigning {action_config.variable} = {assigned_value:.6f} (from '{action_config.value}')")
                self.state[action_config.variable] = assigned_value
            except Exception as e:
                print(f"DEBUG_ACTION: Error evaluating deterministic assignment for {action_config.variable}='{action_config.value}': {e}") # Not wrapped (critical)
                reward -= 100.0
                terminated = True
                applied_successfully = False
            return reward, terminated, applied_successfully

        elif isinstance(action_config, NondeterministicAssignmentActionConfig):
            assign_key = f"assign_{action_config.variable}"
            if assign_key in agent_action:
                self.state[action_config.variable] = agent_action[assign_key]
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_ACTION: Nondeterministic assignment for {action_config.variable} = {agent_action[assign_key]:.6f}")
            else:
                print(f"DEBUG_ACTION: Warning: Nondeterministic assignment for '{action_config.variable}' (key '{assign_key}') not found in agent action. Defaulting to current value.") # Not wrapped (warning)
                # This might indicate an incomplete action by the agent, penalize.
                reward -= 50.0
                # Do not terminate here, as it might be a valid choice to not assign.
                # However, if the problem requires it, this might need to be `terminated = True`.
                # For now, we assume it's okay to skip, but with a penalty.
            return reward, terminated, applied_successfully

        elif isinstance(action_config, ActionConstraintConfig):
            constraint_met = self._evaluate_expression(action_config.formula, is_constraint=True)
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_ACTION: Evaluating constraint '{action_config.formula}'. Result: {constraint_met}")

            # Try to find metadata for this constraint if it's a predicate call
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', action_config.formula)
            if predicate_name_match:
                pred_name = predicate_name_match.group(1)
                metadata = self._get_predicate_metadata(pred_name)
            else:
                # Default metadata for generic constraints
                metadata = {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'}

            if not constraint_met:
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_ACTION: Constraint FAILED: '{action_config.formula}'. Setting applied_successfully=False.")
                reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, False, is_failure_case=True)
                # If metadata didn't specify termination/truncation, and it's not 'continue', then it's a failure to apply
                if not terminated and not truncated and metadata.get('on_failure', '').split(':')[0] == 'continue':
                    applied_successfully = True # Constraint failed but we are told to continue
                    if self.debug_mode: # <--- Wrapped debug print
                        print(f"DEBUG_ACTION: Constraint FAILED but metadata says 'continue'. applied_successfully=True.")
                else:
                    applied_successfully = False # Constraint failed and we should terminate/truncate
            else:
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_ACTION: Constraint PASSED: '{action_config.formula}'. Setting applied_successfully=True.")
                # If constraint is met, apply 'on_success' effects if any
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                applied_successfully = True # Constraint met, so action was successfully applied

            return reward, terminated, applied_successfully

        elif isinstance(action_config, CallHybridProgramConfig):
            # This generator assumes controlFunc is the primary action selection.
            # If CallHybridProgramConfig appears here, it's an unsupported nested HP call within controlFunc.
            print(f"DEBUG_ACTION: Warning: CallHybridProgramConfig '{action_config.hp_name}' encountered in _execute_action_config. Not fully supported yet.") # Not wrapped (warning)
            reward -= 50.0
            applied_successfully = False
            return reward, terminated, applied_successfully

        else:
            print(f"DEBUG_ACTION: Error: Unknown action config type: {type(action_config)}") # Not wrapped (critical)
            reward -= 200.0
            terminated = True
            applied_successfully = False
            return reward, terminated, applied_successfully

    def _apply_control_action(self, action):
        """
        Applies the agent's chosen action by traversing the control function's AST.
        """
        reward, terminated, applied_successfully = self._execute_action_config(self._control_func_action_config, action)
        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_CONTROL: _apply_control_action final result: reward={reward:.2f}, terminated={terminated}, applied_successfully={applied_successfully}")

        # Apply t := 0; after controlFunc, as specified in the problem loop body
        if applied_successfully: # Only reset t if the action was valid and applied
            self.state['t'] = 0.0

        return reward, terminated, applied_successfully


    def _split_top_level_and_local(self, formula_str: str) -> List[str]:
        """
        Splits a formula string by top-level '&' operators, respecting brace nesting.
        This is a simplified version of the parser's split_by_operator_outside_braces
        tailored for logical conjunctions.
        """
        parts = []
        brace_level = 0
        current_part_start = 0

        # Normalize spaces around '&' for consistent splitting
        formula_str = re.sub(r'\s*&\s*', '&', formula_str)

        i = 0
        while i < len(formula_str):
            if formula_str[i] == '(':
                brace_level += 1
            elif formula_str[i] == ')':
                brace_level -= 1

            if formula_str[i] == '&' and brace_level == 0:
                part = formula_str[current_part_start:i].strip()
                if part:
                    parts.append(part)
                current_part_start = i + 1
            i += 1

        # Add the last part after the loop finishes
        last_part = formula_str[current_part_start:].strip()
        if last_part:
            parts.append(last_part)
        return parts if parts else [formula_str.strip()]


    def _populate_cisb_diverse_z3(self):
        """
        Populates the Contrastive Initial State Buffer (CISB) with diverse initial states
        that satisfy the problem's precondition, using Z3.Optimize with soft constraints
        and blocking clauses for diversity.
        """
        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_CISB: Populating Contrastive Initial State Buffer (CISB) with {self._cisb_size} diverse states using Z3.Optimize...")
        current_cisb_count = 0
        attempts = 0

        # 1. Prepare Z3 symbols for all state variables and constants
        z3_symbols = {}
        for var_decl in self._variables_config.state_variables:
            z3_symbols[var_decl.name] = z3.Real(var_decl.name)
        for const_decl in self._definitions_config.constants:
            # Only create Z3 symbols for constants that are meant to be solved for.
            # Fixed value constants will be handled by fixed_constants_map.
            if not (const_decl.value is not None and re.fullmatch(r'^-?\d+(\.\d+)?$', const_decl.value)):
                z3_symbols[const_decl.name] = z3.Real(const_decl.name)

        # Separate fixed constants for formula conversion
        fixed_constants_map = {
            const_decl.name: float(const_decl.value)
            for const_decl in self._definitions_config.constants
            if const_decl.value is not None and re.fullmatch(r'^-?\d+(\.\d+)?$', const_decl.value)
        }

        # 2. Create Z3 Optimizer
        opt = z3.Optimize()
        opt.set(timeout=10000) # Increased timeout to 10 seconds per iteration

        # 3. Add Hard Constraints (from precondition and sample_ranges)
        precondition_kx = self._problem_config.precondition.formula
        try:
            z3_precondition_formula = self._kx_expr_to_z3_formula(precondition_kx, z3_symbols, fixed_constants_map)
            opt.add(z3_precondition_formula)
        except ValueError as e:
            print(f"CRITICAL ERROR: Failed to convert precondition to Z3 formula during CISB population: {e}") # Not wrapped (critical)
            return # Cannot populate CISB without a valid precondition

        # Add general bounds for all Z3 variables
        for var_name, z3_var in z3_symbols.items():
            low, high = self.sample_ranges.get(var_name, (-100.0, 100.0))
            opt.add(z3_var >= low, z3_var <= high)

        # 4. Iterate and solve for diverse samples
        while current_cisb_count < self._cisb_size and attempts < self._cisb_populate_attempts:
            attempts += 1
            opt.push() # Push current state of the optimizer

            # Add soft constraints to guide the solver to different regions
            soft_constraints_added = False
            for var_name, z3_var in z3_symbols.items():
                # Skip 'v' and 't' as they are fixed by precondition
                if var_name in ['v', 't']:
                    continue
                
                low, high = self.sample_ranges.get(var_name, (-100.0, 100.0))
                if low == high: # If the variable has a fixed value (e.g., v=0), no need for soft constraint
                    continue
                
                target = self.np_random.uniform(low, high)
                # Use a weight of 1 for soft constraints
                opt.add_soft(z3_var == target, weight=1) 
                soft_constraints_added = True
            
            if not soft_constraints_added:
                pass
                # if self.debug_mode: # <--- Wrapped debug print
                # print("DEBUG_CISB: Warning: No variables available for soft constraints to encourage diversity. CISB might not be diverse.")


            check_result = opt.check()

            if check_result == z3.sat:
                model = opt.model()
                solved_values = {}
                for k, v in z3_symbols.items():
                    evaluated_val = model.evaluate(v, model_completion=True)
                    if evaluated_val is not None:
                        if hasattr(evaluated_val, 'as_fraction'):
                            solved_values[k] = float(evaluated_val.as_fraction())
                        elif hasattr(evaluated_val, 'as_long'):
                            solved_values[k] = float(evaluated_val.as_long())
                        else:
                            solved_values[k] = float(str(evaluated_val))
                solved_values.update(fixed_constants_map) # Add fixed constants

                # Normalize dx, dy to ensure `isWellformedDir` holds for the Python `_approx_eq` check.
                if 'dx' in solved_values and 'dy' in solved_values:
                    dx_val = solved_values['dx']
                    dy_val = solved_values['dy']
                    magnitude = math.sqrt(dx_val**2 + dy_val**2)
                    if magnitude > 1e-6: # Avoid division by zero
                        solved_values['dx'] = dx_val / magnitude
                        solved_values['dy'] = dy_val / magnitude
                    else: # Default to unit vector if magnitude is zero
                        # If dx, dy are both near zero, set to a default unit vector
                        solved_values['dx'] = 1.0
                        solved_values['dy'] = 0.0

                self._cisb.append(solved_values)
                current_cisb_count += 1
                if self.debug_mode and current_cisb_count % 100 == 0: # <--- Wrapped debug print
                    print(f"DEBUG_CISB: ... {current_cisb_count}/{self._cisb_size} states populated.")

                # Add a blocking clause to ensure diversity for the next sample.
                # The next solution must NOT be within an epsilon-neighborhood
                # of the current solution for AT LEAST ONE solved variable.
                blocking_conditions = []
                for var_name, z3_var in z3_symbols.items():
                    val = solved_values.get(var_name)
                    if val is not None:
                        # Use a small but significant epsilon for diversity
                        diversity_epsilon = 0.1 # This value can be tuned
                        blocking_conditions.append(
                            z3.Or(z3_var < val - diversity_epsilon, z3_var > val + diversity_epsilon)
                        )
                
                if blocking_conditions: # Only add if there are variables to block
                    opt.add(z3.Or(blocking_conditions)) # Add blocking clause to the base solver
                
                opt.pop() # Pop the soft constraints for this iteration

            elif check_result == z3.unsat:
                print(f"DEBUG_CISB: Z3 UNSAT during CISB population attempt {attempts}. No more diverse solutions found within current constraints.") # Not wrapped (warning)
                opt.pop()
                break # Break if no more solutions are possible
            else: # z3.unknown (e.g., timeout)
                print(f"DEBUG_CISB: Z3 UNKNOWN during CISB population attempt {attempts}. Retrying with new soft constraints.") # Not wrapped (warning)
                opt.pop() # Pop the soft constraints for this iteration, try again

        if current_cisb_count < self._cisb_size:
            print(f"DEBUG_CISB: Warning: Could only populate {current_cisb_count} states for CISB after {attempts} attempts.") # Not wrapped (warning)
        else:
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_CISB: CISB population complete with {current_cisb_count} states.")

        unique_samples = set()
        for sample in self._cisb:
            rounded_sample = tuple(sorted([(k, round(v, 6)) for k, v in sample.items()]))
            unique_samples.add(rounded_sample)

        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_CISB: Number of unique initial states in CISB: {len(unique_samples)}")

            print("DEBUG_CISB: Sample initial states from CISB (first 5):")
            for i, sample in enumerate(self._cisb[:min(5, len(self._cisb))]):
                print(f"  Sample {i+1}: {sample}")
                print(f"  Solved constants for CISB entry: A={sample.get('A')}, b={sample.get('b')}, ep={sample.get('ep')}") # Added debug print
        if not self._cisb:
            print("CRITICAL ERROR: CISB is empty! This environment cannot function correctly without initial states.") # Not wrapped (critical)
            print("Please check precondition and sampling ranges for satisfiability.") # Not wrapped (critical)


    # The original _populate_cisb method is now replaced by _populate_cisb_diverse_z3.
    # If you wanted to keep both, you could rename the original to _populate_cisb_simple_z3
    # and have _populate_cisb act as a dispatcher based on a flag.
    # For this example, I'm directly calling the diverse one from __init__.
    def _populate_cisb(self):
        self._populate_cisb_diverse_z3()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Store the history of the *previous* episode before clearing
        self.last_episode_history = self.state_history 
        
        self.state_history = [] # Reset state history for new episode
        self.cumulative_episode_time = 0.0 # NEW: Reset cumulative time for new episode

        if not self._cisb:
            # Fallback if CISB is somehow empty (should be caught by _populate_cisb warning)
            print("DEBUG_RESET: Error: CISB is empty. Generating a single initial state using Z3 as fallback.") # Not wrapped (critical)
            # Default initial values for state variables (as a last resort)
            self.state['x'] = 0.0
            self.state['y'] = 0.0
            self.state['v'] = 0.0 # Must be 0 by precondition
            self.state['a'] = 0.0
            self.state['dx'] = 1.0
            self.state['dy'] = 0.0
            self.state['w'] = 0.0
            self.state['r'] = 1.0 # Must be non-zero by precondition
            self.state['xo'] = 5.0
            self.state['yo'] = 0.0
            self.state['t'] = 0.0

            # Default constants (must satisfy bounds)
            self.constants['ep'] = 1.0
            self.constants['b'] = 1.0
            self.constants['A'] = 1.0
            self.constants['W'] = 1.0

            # No _solve_initial_state_with_z3 method defined in the provided code,
            # so this fallback path would currently fail.
            # Assuming for now that CISB population is successful.
            # If it fails, this would need a Z3 call to find a valid initial state.
            final_initial_state = {k: self.state[k] for k in self._state_keys}
            final_initial_state.update({k: self.constants[k] for k in self.constants.keys()})
        else:
            # Sample a random state from the Contrastive Initial State Buffer
            chosen_initial_state_entry = self.np_random.choice(self._cisb)
            final_initial_state = chosen_initial_state_entry.copy()

        # Apply the chosen initial state (from CISB or fallback)
        for var_name, value in final_initial_state.items():
            if var_name in self.state:
                self.state[var_name] = value
            elif var_name in self.constants:
                self.constants[var_name] = value

        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_RESET: Initial state: x={self.state['x']:.6f}, y={self.state['y']:.6f}, v={self.state['v']:.6f}, a={self.state['a']:.6f}, w={self.state['w']:.6f}")
            print(f"DEBUG_RESET: Initial constants: A={self.constants['A']:.6f}, b={self.constants['b']:.6f}, ep={self.constants['ep']:.6f}")

        self.last_failing_precondition_formula = None # Reset for the new episode

        # Record the initial state, including cumulative time
        initial_state_record = self.state.copy()
        initial_state_record['episode_time'] = self.cumulative_episode_time # NEW: Add cumulative time
        self.state_history.append(initial_state_record)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        # --- Initial Loop Invariant Check (before control/dynamics) ---
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
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (pre-control): '{conj_str}'")
                # Apply failure effects
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time # NEW: Add cumulative time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                # Apply success effects if any (only if action is not 'continue')
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                    if terminated or truncated:
                        state_record_before_early_return = self.state.copy()
                        state_record_before_early_return['episode_time'] = self.cumulative_episode_time # NEW: Add cumulative time
                        self.state_history.append(state_record_before_early_return)
                        return self._get_obs(), reward, terminated, truncated, info

        if not loop_invariant_satisfied_pre_step:
            # Fallback for general loopinv failure if not caught by specific conjuncts
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: General loop invariant FAILED (pre-control).")
            reward, terminated, truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-500.0', 'on_failure': 'terminate:status=failure'},
                reward, terminated, truncated, is_failure_case=True
            )
            state_record_before_early_return = self.state.copy()
            state_record_before_early_return['episode_time'] = self.cumulative_episode_time # NEW: Add cumulative time
            self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info # Terminate early


        # --- 1. Apply Control Function (Discrete Action) ---
        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_STEP {self.current_step}: State before control action: x={self.state['x']:.6f}, y={self.state['y']:.6f}, v={self.state['v']:.6f}, a={self.state['a']:.6f}, w={self.state['w']:.6f}, ep={self.constants['ep']:.6f}")
        control_reward, control_terminated, control_applied = self._apply_control_action(action)
        reward += control_reward
        terminated = terminated or control_terminated
        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_STEP {self.current_step}: State after _apply_control_action (before forced override): x={self.state['x']:.6f}, y={self.state['y']:.6f}, v={self.state['v']:.6f}, a={self.state['a']:.6f}, w={self.state['w']:.6f}, control_applied={control_applied}")

        # --- TEMPORARY: Force an accelerating action for demonstration ---
        # This is for debugging the plotting and ODEs, not for actual training.
        # It ensures that 'a' is positive and 'v' can increase.
        if self._force_acceleration_action:
            # Ensure A, W, b, ep are set in constants for the forced action to work
            if self.constants['A'] <= 0.0: self.constants['A'] = 1.0
            if self.constants['W'] <= 0.0: self.constants['W'] = 1.0
            if self.constants['b'] <= 0.0: self.constants['b'] = 1.0
            if self.constants['ep'] <= 0.0: self.constants['ep'] = 1.0

            # Directly set 'a' to 'A' and 'w' to 0 for simplicity, bypassing action parsing for these.
            # This is a temporary debug step to ensure acceleration is applied.
            self.state['a'] = self.constants['A']
            # Make sure w is consistent with v=0 and r!=0 for the constraint r*w=v
            # If v=0, then r*w=0, and since r!=0, w must be 0.
            self.state['w'] = 0.0 
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_FORCED: FORCED OVERRIDE: Setting a={self.state['a']:.6f}, w={self.state['w']:.6f}")
                print(f"DEBUG_FORCED: Current constants for forced action: A={self.constants['A']:.6f}, b={self.constants['b']:.6f}, ep={self.constants['ep']:.6f}")

            # Also ensure control_applied is True for the forced action path to continue to ODE
            control_applied = True # Force it to be true if we are overriding

        # --- END TEMPORARY ---

        if not control_applied:
            # Invalid action chosen or constraint violated during control phase
            # Record state before early return
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: Control action not applied. Terminating early.")
            state_record_before_early_return = self.state.copy()
            state_record_before_early_return['episode_time'] = self.cumulative_episode_time # NEW: Add cumulative time
            self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info

        # --- 2. Simulate Differential Equations (Continuous Dynamics) ---
        # Prepare initial state for ODE solver
        y0 = np.array([self.state[var_name] for var_name in self._ode_state_map.keys()], dtype=np.float64)

        ep_val = self.constants['ep'] # Get the 'ep' constant value
        t_span = (0, ep_val) # Integrate up to time limit 'ep'

        ode_terminated_by_event = False

        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_ODE: Before solve_ivp: y0={y0}, t_span={t_span}, ep_val={ep_val:.6f}")

        if ep_val <= 0:
            # If ep is non-positive, no meaningful time evolution occurs.
            self.state['t'] = 0.0 # Time does not advance or is reset
            # This is a violation of 'ep > 0' from bounds() predicate.
            # If the CISB is populated correctly, ep should always be > 0.
            # This case might indicate an an issue with CISB or a very unusual state.
            print(f"DEBUG_ODE: Warning: 'ep' constant is non-positive ({ep_val:.6f}). Skipping ODE integration and penalizing.") # Not wrapped (warning)
            # Use bounds predicate for this check
            bounds_meta = self._get_predicate_metadata('bounds')
            reward, terminated, truncated = self._apply_meta_effect(
                bounds_meta or {'type': 'safety', 'penalty': '-200.0', 'on_failure': 'terminate:status=failure'}, # Default to truncate as per problem description example
                reward, terminated, truncated, is_failure_case=True
            )
            ode_terminated_by_event = True # Treat as terminated by event
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

                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_ODE: After solve_ivp: solution.status={solution.status}, solution.message={solution.message}")
                    print(f"DEBUG_ODE: solution.t={solution.t}")
                    if solution.y.shape[1] > 0:
                        print(f"DEBUG_ODE: solution.y_initial={solution.y[:,0]}, solution.y_final={solution.y[:,-1]}")
                    if solution.t_events:
                        print(f"DEBUG_ODE: solution.t_events={solution.t_events}")


                # --- NEW: Record intermediate ODE states for plotting ---
                if solution.success and solution.sol is not None:
                    actual_ode_duration = solution.t[-1] if solution.t.size > 0 else 0.0

                    if actual_ode_duration > 1e-9: # Only generate intermediate points if duration is meaningful
                        # Generate a reasonable number of time points for dense output
                        # Ensure a minimum number of points for visualization over any duration
                        num_ode_points = max(20, int(actual_ode_duration * 100)) # Min 20 points, 100 points per unit of time
                        t_eval = np.linspace(0, actual_ode_duration, num_ode_points)

                        # Evaluate the solution at these points
                        ode_solution_points = solution.sol(t_eval) # This returns a 2D array: (num_vars, num_t_eval_points)

                        # Append these intermediate states to history
                        # Start from 1 to avoid duplicating the point already recorded from previous step/reset
                        start_idx = 1 if len(self.state_history) > 0 and \
                                         np.isclose(self.state_history[-1]['episode_time'], self.cumulative_episode_time) and \
                                         np.isclose(t_eval[0], 0.0) else 0

                        for k in range(start_idx, ode_solution_points.shape[1]): # Iterate through each time point
                            current_ode_state = {}
                            for i, var_name in enumerate(self._ode_state_map.keys()):
                                current_ode_state[var_name] = ode_solution_points[i, k]
                            
                            # Calculate the cumulative episode time for this intermediate point
                            # self.cumulative_episode_time is the time *before* this ODE step
                            intermediate_episode_time = self.cumulative_episode_time + current_ode_state['t']
                            current_ode_state['episode_time'] = intermediate_episode_time
                            self.state_history.append(current_ode_state)
                            if self.debug_mode and k % (ode_solution_points.shape[1] // 5 + 1) == 0: # Print a few samples # <--- Wrapped debug print
                                print(f"DEBUG_ODE_APPEND: Appended intermediate state k={k}: x={current_ode_state['x']:.2f}, y={current_ode_state['y']:.2f}, v={current_ode_state['v']:.2f}, w={current_ode_state['w']:.2f}, a={current_ode_state['a']:.2f}, ep_time={intermediate_episode_time:.2f}")
                # --- END NEW INTERMEDIATE RECORDING ---

                # Update self.state with the final values from ODE solver
                if solution.status == 0: # Integration completed successfully
                    final_y = solution.y[:, -1]
                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]
                elif solution.status == 1: # An event occurred
                    if solution.y_events and solution.y_events[0].size > 0:
                        final_y = solution.y_events[0][0]
                    else:
                        final_y = solution.y[:, -1] # Fallback if event data is missing

                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]

                    # Check which event terminated it
                    # If v < 0, it's a domain violation -> termination
                    if self.state.get('v', 0.0) < -self.equality_tolerance: # Use tolerance for v < 0 check
                        if self.debug_mode: # <--- Wrapped debug print
                            print(f"DEBUG_ODE: Event: v became negative ({self.state['v']:.6f}).")
                        phys_valid_meta = self._get_predicate_metadata('isPhysicallyValid')
                        reward, terminated, truncated = self._apply_meta_effect(
                            phys_valid_meta or {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'truncate:status=failure'}, # Default to truncate as per problem description example
                            reward, terminated, truncated, is_failure_case=True
                        )
                        ode_terminated_by_event = True
                    # If t > ep, it means the time limit was reached, which is expected behavior.
                    # No additional penalty for that specific event, as it's part of the evolution domain.
                else: # Integration failed (status 2)
                    if self.debug_mode: # <--- Wrapped debug print
                        print(f"DEBUG_ODE: ODE integration failed with status {solution.status}.")
                    reward, terminated, truncated = self._apply_meta_effect(
                        {'type': 'safety', 'penalty': '-150.0', 'on_failure': 'terminate:status=failure'},
                        reward, terminated, truncated, is_failure_case=True
                    )
                    ode_terminated_by_event = True

            except Exception as e:
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_ODE: Exception during ODE integration: {e}")
                reward, terminated, truncated = self._apply_meta_effect(
                    {'type': 'safety', 'penalty': '-200.0', 'on_failure': 'terminate:status=failure'},
                    reward, terminated, truncated, is_failure_case=True
                )
                ode_terminated_by_event = True

        terminated = terminated or ode_terminated_by_event
        truncated = truncated or ode_terminated_by_event # If ODE event truncates, propagate

        # NEW: Update cumulative time based on the duration of this ODE step
        # self.state['t'] now holds the duration of the ODE integration that just completed.
        self.cumulative_episode_time += self.state['t']
        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_STEP {self.current_step}: Updated cumulative_episode_time={self.cumulative_episode_time:.6f}")
            print(f"DEBUG_STEP {self.current_step}: State after ODE integration: x={self.state['x']:.6f}, y={self.state['y']:.6f}, v={self.state['v']:.6f}, a={self.state['a']:.6f}, w={self.state['w']:.6f}, t={self.state['t']:.6f}")


        # If ODE integration or control action led to termination/truncation, return early
        if terminated or truncated:
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: Terminated/Truncated after ODE. Returning early.")
            # Ensure the final state of the episode is recorded if not already added by intermediate points
            if not (solution.success and solution.sol is not None and solution.t.size > 1 and actual_ode_duration > 1e-9):
                state_record_before_early_return = self.state.copy()
                state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info

        # --- 3. Evaluate Loop Invariant and Apply Rewards/Penalties/Shaping ---
        # Re-evaluate loop invariant after ODEs
        loop_invariant_satisfied_post_ode = True
        for conj_str in loop_inv_conjuncts: # Reuse loop_inv_conjuncts from earlier
            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', conj_str)
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {}

            if not conjunct_result:
                loop_invariant_satisfied_post_ode = False
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (post-ODE): '{conj_str}'")
                # Apply failure effects
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time # NEW: Add cumulative time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                # Apply success effects if any (only if action is not 'continue')
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                    if terminated or truncated:
                        state_record_before_early_return = self.state.copy()
                        state_record_before_early_return['episode_time'] = self.cumulative_episode_time # NEW: Add cumulative time
                        self.state_history.append(state_record_before_early_return)
                        return self._get_obs(), reward, terminated, truncated, info

        if not loop_invariant_satisfied_post_ode:
            # Fallback for general loopinv failure if not caught by specific conjuncts
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: General loop invariant FAILED (post-ODE).")
            reward, terminated, truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-300.0', 'on_failure': 'terminate:status=failure'},
                reward, terminated, truncated, is_failure_case=True
            )
            if terminated or truncated:
                state_record_before_early_return = self.state.copy()
                state_record_before_early_return['episode_time'] = self.cumulative_episode_time # NEW: Add cumulative time
                self.state_history.append(state_record_before_early_return)
                return self._get_obs(), reward, terminated, truncated, info

        # --- Reward Shaping (from progress predicates) ---
        # isWellformedDir (progress type, shape: -((dx^2 + dy^2) - 1)^2)
        wellformed_meta = self._get_predicate_metadata('isWellformedDir')
        if wellformed_meta.get('type') == 'progress' and 'shape' in wellformed_meta:
            shape_value = self._evaluate_shape_expression(wellformed_meta['shape'])
            reward += shape_value # Add to reward
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: Applied isWellformedDir shape reward: {shape_value:.6f}")

        # isEfficient (progress type, shape: 1/v if v > 0 else v)
        efficient_meta = self._get_predicate_metadata('isEfficient')
        if efficient_meta.get('type') == 'progress' and 'shape' in efficient_meta:
            shape_value = self._evaluate_shape_expression(efficient_meta['shape'])
            reward += shape_value # Add to reward
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: Applied isEfficient shape reward: {shape_value:.6f}")


        # --- 4. Check Problem Postcondition (ensure) ---
        # This is the goal of the problem.
        postcondition_formula_kx = self._problem_config.postcondition.formula
        postcondition_satisfied = self._evaluate_expression(postcondition_formula_kx, is_constraint=True)
        ensure_meta = self._get_predicate_metadata('ensure') # Get metadata for 'ensure' predicate

        if postcondition_satisfied:
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: Postcondition '{postcondition_formula_kx}' SATISFIED.")
            reward, terminated, truncated = self._apply_meta_effect(
                ensure_meta, reward, terminated, truncated, is_failure_case=False
            )
        else:
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: Postcondition '{postcondition_formula_kx}' NOT SATISFIED.")
            reward, terminated, truncated = self._apply_meta_effect(
                ensure_meta, reward, terminated, truncated, is_failure_case=True
            )

            # Also apply shape if available for the postcondition (ensure)
            if ensure_meta.get('type') == 'progress' and 'shape' in ensure_meta:
                shape_value = self._evaluate_shape_expression(ensure_meta['shape'])
                reward += shape_value # Add to reward
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"DEBUG_STEP: Applied ensure shape reward: {shape_value:.6f}")

        # --- 5. Check Truncation ---
        if self.current_step >= self.max_episode_steps:
            truncated = True
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: Max episode steps ({self.max_episode_steps}) reached. Truncating.")
            # Small penalty for truncating without reaching goal, unless already terminated by ensure failure
            if not terminated and not postcondition_satisfied:
                reward -= 10.0

        # Record the state after all updates for this step
        # This final state for the step is already included if intermediate points were recorded,
        # so we only append it if no intermediate points were generated (e.g., ODE duration was too short)
        if not (solution.success and solution.sol is not None and solution.t.size > 1 and actual_ode_duration > 1e-9):
            state_record_after_step = self.state.copy()
            state_record_after_step['episode_time'] = self.cumulative_episode_time # NEW: Add cumulative time
            self.state_history.append(state_record_after_step)
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_STEP: Appended final state for step {self.current_step} to history.")

        observation = self._get_obs()
        info = self._get_info()
        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_STEP {self.current_step}: Step complete. Reward={reward:.6f}, Terminated={terminated}, Truncated={truncated}")
        return observation, reward, terminated, truncated, info

    def _plot_state_history(self, variables_to_plot: Optional[List[str]] = None, history_data: Optional[List[Dict[str, Any]]] = None):
        """
        Generates and displays a matplotlib plot of the state history.
        The x-axis will always use 'episode_time' for cumulative plotting.
        If history_data is provided, it will be used instead of self.state_history.
        """
        data_to_plot = history_data if history_data is not None else self.state_history

        if not data_to_plot:
            if self.debug_mode: # <--- Wrapped debug print
                print("DEBUG_PLOT: No state history available to plot.")
            return

        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_PLOT: state_history length: {len(data_to_plot)}")
        df = pd.DataFrame(data_to_plot)
        if self.debug_mode: # <--- Wrapped debug print
            print(f"DEBUG_PLOT: DataFrame shape: {df.shape}")
            print(f"DEBUG_PLOT: DataFrame head:\n{df.head()}")
            print(f"DEBUG_PLOT: DataFrame tail:\n{df.tail()}")
        
        x_axis_key = 'episode_time' 

        if x_axis_key not in df.columns:
            print(f"DEBUG_PLOT: Error: Time key '{x_axis_key}' not found in state history for plotting. This indicates an issue with state history recording.") # Not wrapped (critical)
            return

        if variables_to_plot is None:
            # Plot only the variables that are typically expected to change or are key for the problem.
            # Exclude 't' (duration of current ODE step), 'dx', 'dy', 'r', 'xo', 'yo' for cleaner plots by default.
            # User can override by passing variables_to_plot explicitly.
            default_plot_vars = ['x', 'y', 'v', 'a', 'w']
            variables_to_plot = [col for col in default_plot_vars if col in df.columns]
        else:
            variables_to_plot = [v for v in variables_to_plot if v in df.columns]

        if not variables_to_plot:
            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_PLOT: No variables selected for plotting or found in DataFrame.")
            return

        num_vars = len(variables_to_plot)
        cols = 2
        rows = (num_vars + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        fig.suptitle('Agent State Evolution Over Time', fontsize=16)

        # Define a minimum y-axis range to ensure visibility of small changes
        # Increased for better visibility of very subtle changes
        min_y_range_absolute = 0.05 # Increased from 0.02

        for i, var in enumerate(variables_to_plot):
            ax = axes[i]
            ax.plot(df[x_axis_key], df[var], label=var)
            ax.set_xlabel(f"Time ({x_axis_key})")
            ax.set_ylabel(var)
            ax.set_title(f"Evolution of {var}")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Explicitly set x-axis limits
            min_time = df[x_axis_key].min()
            max_time = df[x_axis_key].max()
            if max_time <= min_time + 1e-9: # Very small or zero range
                ax.set_xlim(0, max_time + 0.01) # Add a small buffer if range is tiny
            else:
                ax.set_xlim(0, max_time)

            # Dynamically set y-axis limits based on the actual data range
            min_val = df[var].min()
            max_val = df[var].max()
            
            data_range = max_val - min_val

            if self.debug_mode: # <--- Wrapped debug print
                print(f"DEBUG_PLOT: Plotting {var}: min_val={min_val:.6f}, max_val={max_val:.6f}, data_range={data_range:.6f}")

            if data_range < min_y_range_absolute:
                # If the data range is too small, center the plot around the mean with a fixed minimum range
                avg_val = (min_val + max_val) / 2
                lower_bound = avg_val - min_y_range_absolute / 2
                upper_bound = avg_val + min_y_range_absolute / 2
                ax.set_ylim(lower_bound, upper_bound)
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"  DEBUG_PLOT: Applied forced y-lim: [{lower_bound:.6f}, {upper_bound:.6f}] (range {min_y_range_absolute})")
            else:
                # Otherwise, use a percentage buffer of the actual data range
                buffer = data_range * 0.1 # 10% buffer of the range
                lower_bound = min_val - buffer
                upper_bound = max_val + buffer
                ax.set_ylim(min_val - buffer, max_val + buffer)
                if self.debug_mode: # <--- Wrapped debug print
                    print(f"  DEBUG_PLOT: Applied buffered y-lim: [{lower_bound:.6f}, {upper_bound:.6f}] (range {upper_bound - lower_bound:.6f})")


        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def render(self, mode="human", variables_to_plot: Optional[List[str]] = None):
        if mode == "human":
            # In a real environment, this would draw the robot, obstacle, etc.
            # For now, just print state for demonstration.
            # print(f"Render: Step={self.current_step}, State={self.state}")
            pass # Removed print for human mode as well
        elif mode == "plot_data":
            # Return the full state history for external plotting (e.g., React)
            return self.state_history
        elif mode == "matplotlib":
            # Generate and display a matplotlib plot, always using 'episode_time' for x-axis
            self._plot_state_history(variables_to_plot, history_data=self.last_episode_history)
            return None # Matplotlib handles displaying the plot directly
        return None # Default return if render_mode is not handled

    def close(self):
        # Close any open matplotlib figures
        plt.close('all')
        pass