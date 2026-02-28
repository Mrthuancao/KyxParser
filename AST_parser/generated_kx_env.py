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

class GeneratedKXEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- Store original AST config objects for Z3 and other runtime needs ---
        self._definitions_config = DefinitionsConfig(
            constants=[
                ConstantDefinitionConfig(
                    name='ep',
                    type='Real',
                    value=None
                ),
                ConstantDefinitionConfig(
                    name='b',
                    type='Real',
                    value=None
                ),
                ConstantDefinitionConfig(
                    name='A',
                    type='Real',
                    value=None
                ),
                ConstantDefinitionConfig(
                    name='W',
                    type='Real',
                    value=None
                )
            ],
            functions=[
                FunctionDefinitionConfig(
                    name='stopDist',
                    return_type='Real',
                    parameters=[
    ('Real', 'v')
],
                    body='v^2 / (2*b)'
                ),
                FunctionDefinitionConfig(
                    name='accelComp',
                    return_type='Real',
                    parameters=[
    ('Real', 'v')
],
                    body='((A/b + 1) * (A/2 * ep^2 + ep*v))'
                ),
                FunctionDefinitionConfig(
                    name='admissibleSeparation',
                    return_type='Real',
                    parameters=[
    ('Real', 'v')
],
                    body='stopDist(v) + accelComp(v)'
                )
            ],
            predicates=[
                PredicateDefinitionConfig(
                    name='isWellformedDir',
                    parameters=[
    ('Real', 'dx'),
    ('Real', 'dy')
],
                    body='dx^2 + dy^2 = 1'
                ),
                PredicateDefinitionConfig(
                    name='bounds',
                    parameters=[],
                    body='A >= 0\n    & b > 0\n    & ep > 0'
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
                    body='v = 0\n    & (x-xo)^2 + (y-yo)^2 > 0\n    & isWellformedDir(dx, dy)'
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
                    body='bounds() & initialState(x, y, v, dx, dy, xo, yo)'
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
    ('Real', 'yo')
],
                    body='v >= 0\n    & isWellformedDir(dx, dy)\n    & (abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v))'
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
                                        formula=' abs(x-xo) > admissibleSeparation(v)\n        | abs(y-yo) > admissibleSeparation(v)'
                                    )
                                ]
                            )
                        ]
                    )
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
                    duration=None
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
                loop_invariant_formula='loopinv(x, y, v, dx, dy, xo, yo)',
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

        # --- KeYmaera X Definitions ---
        self.constants = {}
        self.functions = {}
        self.predicates = {}

        # Constants
        self.constants['ep'] = 0.0
        self.constants['b'] = 0.0
        self.constants['A'] = 0.0
        self.constants['W'] = 0.0

        # Functions (lambdas for easy evaluation)
        self.functions['stopDist'] = lambda v: v**2 / (2*self.constants['b'])
        self.functions['accelComp'] = lambda v: (self.constants['A']/self.constants['b'] + 1) * (self.constants['A']/2 * self.constants['ep']**2 + self.constants['ep']*v)
        self.functions['admissibleSeparation'] = lambda v: self.functions['stopDist'](v) + self.functions['accelComp'](v)

        # Predicates (lambdas for easy evaluation)
        self.predicates['isWellformedDir'] = lambda dx, dy: self._approx_eq(dx**2 + dy**2, 1)
        self.predicates['bounds'] = lambda : ((self.constants['A']) >= (0)) and (((self.constants['b']) > (0)) and ((self.constants['ep']) > (0)))
        self.predicates['initialState'] = lambda x, y, v, dx, dy, xo, yo: (self._approx_eq(v, 0)) and ((((x-xo)**2 + (y-yo)**2) > (0)) and (self.predicates['isWellformedDir'](dx, dy)))
        self.predicates['assumptions'] = lambda x, y, v, dx, dy, xo, yo: (self.predicates['bounds']()) and (self.predicates['initialState'](x, y, v, dx, dy, xo, yo))
        self.predicates['loopinv'] = lambda x, y, v, dx, dy, xo, yo: ((v) >= (0)) and ((self.predicates['isWellformedDir'](dx, dy)) and (((np.abs(x-xo)) > (self.functions['stopDist'](v))) or ((np.abs(y-yo)) > (self.functions['stopDist'](v)))))

        # --- State Variables ---
        # Mapping of state variable names to their index in the observation array
        self._state_keys = ['x', 'y', 'v', 'a', 'dx', 'dy', 'w', 'r', 'xo', 'yo', 't']
        # Current state values (dictionary for easy access by name)
        # Initialize to 0.0 as placeholders; these will be overwritten by reset()
        self.state = {key: 0.0 for key in self._state_keys} 
        # Initial state values are set in reset()
        self.last_failing_precondition_formula = None # Initialize for debugging

        # --- Observation Space ---
        # All state variables are Real, so Box space is appropriate.
        # Bounds are infinite by default, as not specified in KeYmaera X input.
        low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)

        # --- Action Space (from controlFunc) ---
        # The action space is a Dict to handle choices and nondeterministic assignments.
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
                                formula=' abs(x-xo) > admissibleSeparation(v)\n        | abs(y-yo) > admissibleSeparation(v)'
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

                # Ensure self.state is updated with current ODE values for expression evaluation
                for i, var_name in self._ode_state_idx_map.items():
                    self.state[var_name] = y_array[i]

                # Calculate derivatives
                dydt = [
                    v * dx, # dx/dt
                    v * dy, # dy/dt
                    a, # dv/dt
                    0.0, # da/dt
                    -w * dy, # ddx/dt
                    w * dx, # ddy/dt
                    a/r, # dw/dt
                    0.0, # dr/dt
                    0.0, # dxo/dt
                    0.0, # dyo/dt
                    1, # dt/dt
                ]
                return dydt
        '''
        self._ode_events_str = '''
            def event_t_max(t_val, y_array):
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
            ]
        '''

        # Compile ODE function and events dynamically
        # This allows the ODE function to use self.state and other environment properties
        # We need to pass the environment instance to the dynamically created function
        # For safety, we'll use a wrapper
        self._ode_func_compiled = self._compile_ode_func(self._ode_func_str)
        self._ode_events_compiled = self._compile_ode_events(self._ode_events_str)

        # --- Episode Management ---
        self.current_step = 0
        self.max_episode_steps = 200 # Default max steps for an episode
        self.reward_range = (-float('inf'), float('inf'))

        # Initialize the environment to a valid state using the reset method
        self.reset() # <--- ADDED THIS LINE

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
                        print(f"Warning: Mismatch in number of arguments for predicate '{pred_name}'. Expected {len(pred_params)}, got {len(call_args)} in call '{match.group(0)}'.")
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


    def _kx_expr_to_z3_formula(self, kx_expr: str, z3_symbol_map: Dict[str, Any], fixed_constants: Dict[str, Union[int, float]]) -> Any:
        """
        Converts a KeYmaera X expression string into a Z3 formula.
        This function performs string replacements to convert KeYmaera X syntax
        to a Python string that, when evaluated, constructs a Z3 expression.
        It processes operators respecting precedence: arithmetic, relational, then logical.
        For logical operators, it uses a recursive approach to handle nested parentheses.
        """
        expr_transformed = kx_expr.strip()

        # 1. Replace identifiers with their Z3 symbol map reference.
        # Fixed constants will be handled by being added to eval_globals as Z3 literals.
        # Sort by length descending to handle cases where one identifier is a substring of another (e.g., 'dx' before 'x')
        sorted_z3_symbols_and_constants = sorted(
            list(z3_symbol_map.keys()) + list(fixed_constants.keys()), 
            key=len, reverse=True
        )

        for ident_name in sorted_z3_symbols_and_constants:
            replacement = ""
            if ident_name in z3_symbol_map:
                replacement = f"z3_symbol_map['{ident_name}']"
            elif ident_name in fixed_constants:
                # Fixed constants are added to eval_globals directly, so they should be referenced by name
                replacement = ident_name 

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
            # Note: '=' in KeYmaera X maps to '==' for Z3 exact equality
            relational_ops = [('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('<', '<'), ('=', '==')]

            balance = 0
            rel_op_found_kx = None
            rel_op_found_py = None
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
                            if kx_op == '=' and ((i > 0 and expression_str[i-1] in '<>=!') or                                             (i < len(expression_str) - 1 and expression_str[i+1] == '=')):
                                continue 
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break 

            if rel_op_found_kx:
                left = _parse_logical_and_relational_expression_z3(expression_str[:rel_op_idx].strip())
                right = _parse_logical_and_relational_expression_z3(expression_str[rel_op_idx+len(rel_op_found_kx):].strip())
                return f"({left}) {rel_op_found_py} ({right})" 

            # Base case: no logical or relational operators found, return the expression as is.
            # This should now be an arithmetic expression or a literal.
            return expression_str

        # Apply the combined logical and relational expression parser
        expr_transformed = _parse_logical_and_relational_expression_z3(expr_transformed)

        # Final evaluation in a Z3-aware context
        eval_globals = {'z3': z3, 'z3_symbol_map': z3_symbol_map}
        for const_name, const_value in fixed_constants.items():
            if isinstance(const_value, int):
                eval_globals[const_name] = z3.IntVal(const_value)
            else: # float or other numeric
                eval_globals[const_name] = z3.RealVal(const_value)

        try:
            return eval(expr_transformed, eval_globals)
        except Exception as e:
            raise ValueError(f"Failed to evaluate Z3 expression '{kx_expr}' (transformed: '{expr_transformed}'): {e}")



    def _solve_initial_state_with_z3(self, precondition_kx: str) -> Optional[Dict[str, float]]:
        """
        Uses Z3 to find a satisfying assignment for initial state variables and constants
        based on the expanded KeYmaera X precondition.
        Returns a dictionary of solved values (variable/constant name -> float) or None if unsatisfiable.
        """
        # 1. Prepare Z3 symbols (Real for all variables and constants that need to be solved)
        z3_symbol_map = {}

        # Store fixed constants separately to avoid creating Z3 variables for them
        fixed_constants = {}

        # State variables (these are the ones we want Z3 to solve for)
        for var_decl in self._variables_config.state_variables:
            z3_symbol_map[var_decl.name] = z3.Real(var_decl.name)

        # Constants: If they have a fixed numeric value, use that value directly in the expression.
        # Otherwise, treat them as Z3 Real variables to be solved for.
        for const_decl in self._definitions_config.constants:
            if const_decl.value is not None and re.fullmatch(r'^-?\d+(\.\d+)?$', const_decl.value):
                # This constant has a fixed numeric value. Store it as a Python float.
                fixed_constants[const_decl.name] = float(const_decl.value)
            else:
                # This constant either has no value, or its value is an expression.
                # For Z3, we treat it as a variable that needs to be solved.
                z3_symbol_map[const_decl.name] = z3.Real(const_decl.name)

        # 2. Expand predicates in the precondition
        expanded_precondition = self._expand_formula(precondition_kx, self._definitions_config)
        # print(f"Expanded precondition for Z3 solving: '{expanded_precondition}'")

        # 3. Convert expanded KeYmaera X formula to Z3 expression
        try:
            # Pass both z3_symbol_map and fixed_constants
            z3_formula = self._kx_expr_to_z3_formula(expanded_precondition, z3_symbol_map, fixed_constants)
        except ValueError as e:
            print(f"Z3 Solver Error: Failed to convert precondition to Z3 formula: {e}")
            return None

        # print(f"Z3 formula for precondition: {z3_formula}")

        # 4. Create Z3 solver and add the formula
        s = z3.Solver()
        s.add(z3_formula)

        # 5. Check for satisfiability
        check_result = s.check()
        if check_result == z3.sat:
            model = s.model()
            solved_values = {}
            # Add values for state variables and constants that were Z3 variables
            for k, v in z3_symbol_map.items():
                evaluated_val = model.evaluate(v, model_completion=True)
                if evaluated_val is not None:
                    if hasattr(evaluated_val, 'as_fraction'): # For Z3 Real values
                        solved_values[k] = float(evaluated_val.as_fraction())
                    elif hasattr(evaluated_val, 'as_long'): # For Z3 Int values
                        solved_values[k] = float(evaluated_val.as_long())
                    else: # Fallback for other Z3 types or if as_fraction/as_long not present
                        solved_values[k] = float(str(evaluated_val)) # Convert to string and then float
            # Add values for fixed constants
            # These are already numeric, no need for Z3 evaluation
            solved_values.update(fixed_constants)
            return solved_values
        elif check_result == z3.unsat:
            print(f"Z3 Solver: Precondition is UNSAT. No initial state exists for: '{precondition_kx}'")
            return None
        else: # z3.unknown
            print(f"Z3 Solver: Precondition satisfiability UNKNOWN for: '{precondition_kx}'")
            return None



    def _evaluate_expression(self, expr_str: str, local_vars: Optional[Dict[str, Any]] = None, is_constraint: bool = False) -> Any:
        # Safe evaluation of expressions using environment's context.
        # This method parses KeYmaera X expressions at runtime.
        # If is_constraint is True, returns False on evaluation error.

        if local_vars is None:
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

            # Conjunction (next lowest logical) - left-associative
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
                            if kx_op == '=' and (i > 0 and sub_expr[i-1] in '<>=!') or                                             (i < len(sub_expr) - 1 and sub_expr[i+1] == '='):
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
            return eval(expr_py_final, eval_globals_for_eval, local_vars)
        except Exception as e:
            print(f"Error evaluating expression '{expr_str}' (transformed: '{expr_py_final}'): {e}")
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
                print(f"Error: Choice key '{{choice_key}}' missing from agent action.")
                reward -= 200.0
                terminated = True
                applied_successfully = False
                return reward, terminated, applied_successfully

            choice_idx = agent_action[choice_key]
            if not (0 <= choice_idx < len(action_config.choices)):
                print(f"Error: Invalid choice index {{choice_idx}} for '{{choice_key}}'.")
                reward -= 200.0
                terminated = True
                applied_successfully = False
                return reward, terminated, applied_successfully

            chosen_sub_config = action_config.choices[choice_idx]
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
                self.state[action_config.variable] = assigned_value
            except Exception as e:
                print(f"Error evaluating deterministic assignment for {{action_config.variable}}='{{action_config.value}}': {{e}}")
                reward -= 100.0
                terminated = True
                applied_successfully = False
            return reward, terminated, applied_successfully

        elif isinstance(action_config, NondeterministicAssignmentActionConfig):
            assign_key = f"assign_{action_config.variable}"
            if assign_key in agent_action:
                self.state[action_config.variable] = agent_action[assign_key]
            else:
                print(f"Warning: Nondeterministic assignment for '{{action_config.variable}}' (key '{{assign_key}}') not found in agent action. Defaulting to current value.")
                # This might indicate an incomplete action by the agent, penalize.
                reward -= 50.0 
                # Do not terminate here, as it might be a valid choice to not assign.
                # However, if the problem requires it, this might need to be `terminated = True`.
                # For now, we assume it's okay to skip, but with a penalty.
            return reward, terminated, applied_successfully

        elif isinstance(action_config, ActionConstraintConfig):
            # Formula is a raw KeYmaera X string, parsed at runtime by _evaluate_expression
            constraint_met = self._evaluate_expression(action_config.formula, is_constraint=True)
            if not constraint_met:
                # print(f"Constraint violated: {{action_config.formula}}")
                reward -= 100.0 # Penalty for invalid action
                terminated = True
                applied_successfully = False
            return reward, terminated, applied_successfully

        elif isinstance(action_config, CallHybridProgramConfig):
            # This generator assumes controlFunc is the primary action selection.
            # If CallHybridProgramConfig appears here, it's an unsupported nested HP call within controlFunc.
            print(f"Warning: CallHybridProgramConfig '{{action_config.hp_name}}' encountered in _execute_action_config. Not fully supported yet.")
            reward -= 50.0
            applied_successfully = False
            return reward, terminated, applied_successfully

        else:
            print(f"Error: Unknown action config type: {{type(action_config)}}")
            reward -= 200.0
            terminated = True
            applied_successfully = False
            return reward, terminated, applied_successfully

    def _apply_control_action(self, action):
        """
        Applies the agent's chosen action by traversing the control function's AST.
        """
        reward, terminated, applied_successfully = self._execute_action_config(self._control_func_action_config, action)

        # Apply t := 0; after controlFunc, as specified in the problem loop body
        if applied_successfully: # Only reset t if the action was valid and applied
            self.state['t'] = 0.0

        return reward, terminated, applied_successfully


    def _split_top_level_and_local(self, formula_str: str) -> List[str]:
        parts = []
        balance = 0
        start = 0
        # Normalize spaces around '&'
        formula_str = re.sub(r'\s*&\s*', '&', formula_str)

        for i, char in enumerate(formula_str):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            elif char == '&' and balance == 0:
                part = formula_str[start:i].strip()
                if part:
                    parts.append(part)
                start = i + 1

        last_part = formula_str[start:].strip()
        if last_part:
            parts.append(last_part)

        return parts if parts else [formula_str.strip()]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # --- Initial State (based on Problem Precondition and defaults) ---
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

        z3_base_solution = self._solve_initial_state_with_z3(self._problem_config.precondition.formula)
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
                    if var_name in self._state_keys:
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
            if not z3_base_solution and 2 > 0:
                # Apply explicit assignments from the precondition (fallback)
                self.state['v'] = 0.0

            if not z3_base_solution:
                # Ensure dx, dy satisfy isWellformedDir (approximate unit vector) (fallback)
                norm_sq = self.state['dx']**2 + self.state['dy']**2
                ud_val = self.constants.get('Ud', 0.0) # Get Ud from constants, default to 0.0 if not found
                # The actual isWellformedDir is '1-Ud <= dx^2+dy^2 & dx^2 + dy^2 <= 1+Ud'
                # For fallback, we normalize to 1 if it's too far off, or if Ud is 0.
                if not self._approx_eq(norm_sq, 1.0):
                    norm = np.sqrt(norm_sq)
                    if norm > self.equality_tolerance: # Avoid division by zero with tolerance
                        self.state['dx'] /= norm
                        self.state['dy'] /= norm
                    else: # Default to (1,0) if too small
                        self.state['dx'] = 1.0
                        self.state['dy'] = 0.0

            if not z3_base_solution:
                # Ensure robot and obstacle are not at the same position (fallback)
                if self._approx_eq(self.state['x'], self.state['xo']) and self._approx_eq(self.state['y'], self.state['yo']):
                    self.state['x'] += 0.1 # Slightly perturb x to ensure separation

        # Verify the full precondition after setting initial states
        # Evaluate each conjunct of the precondition to identify specific failures
        failing_formula = None
        precondition_conjuncts = self._split_top_level_and_local('((A >= 0     & b > 0     & ep > 0) & (v = 0     & (x-xo)^2 + (y-yo)^2 > 0     & (dx^2 + dy^2 = 1)))')
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


        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        # --- 1. Apply Control Function (Discrete Action) ---
        control_reward, control_terminated, control_applied = self._apply_control_action(action)
        reward += control_reward
        terminated = control_terminated

        if not control_applied:
            # Invalid action chosen or constraint violated during control phase
            # print("Control action not applied due to constraint violation or missing assignment.")
            return self._get_obs(), reward, terminated, truncated, info

        # --- 2. Simulate Differential Equations (Continuous Dynamics) ---
        # Prepare initial state for ODE solver
        y0 = np.array([self.state[var_name] for var_name in self._ode_state_map.keys()], dtype=np.float64)

        ep_val = self.constants['ep'] # Get the 'ep' constant value
        t_span = (0, ep_val) # Integrate up to time limit 'ep'

        if ep_val <= 0:
            # If ep is non-positive, no meaningful time evolution occurs.
            # This might indicate an invalid control action or problem setup.
            self.state['t'] = 0.0 # Time does not advance or is reset
            if ep_val < 0:
                print(f"Warning: 'ep' constant is negative ({ep_val}). Skipping ODE integration and penalizing.")
                reward -= 100.0 # Penalty for invalid time duration
                terminated = True
            # else (ep_val == 0): No time passes, but not necessarily an error if intended.
        else:
            # Set up ODE events for domain constraints
            # Events are functions that return 0 when an event occurs.
            # terminal=True means integration stops. direction=-1 means event triggers when value goes from + to -

            # Ensure self.state reflects current values for ODE func/events
            # The ODE func and events will get 'self' and use self.state

            try:
                solution = solve_ivp(
                    self._ode_func_compiled, 
                    t_span, 
                    y0, 
                    events=self._ode_events_compiled, 
                    dense_output=True,
                    rtol=1e-6, atol=1e-8
                )

                if solution.status == 0: # Integration completed successfully
                    # Update state with final values from ODE solver
                    final_y = solution.y[:, -1]
                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]
                elif solution.status == 1: # An event occurred
                    # Update state to the point where the event occurred
                    # Check if y_events is not empty before accessing
                    if solution.y_events and solution.y_events[0].size > 0:
                        final_y = solution.y_events[0][0] 
                    else:
                        final_y = solution.y[:, -1] # Fallback if event data is missing

                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]

                    # Check which event terminated it
                    # For this problem, events are domain constraints (t <= ep, v >= 0)
                    # If v < 0, it's a domain violation -> termination
                    # We need to re-evaluate after updating self.state
                    # Note: ODE events themselves use floating-point comparisons internally.
                    # This check here is a re-evaluation after the fact.
                    if self.state.get('v', 0.0) < -self.equality_tolerance: # Use tolerance for v < 0 check
                        reward -= 100.0 # Penalty for v < 0
                        terminated = True
                    # If t > ep, it means the time limit was reached, which is expected behavior
                    # so no additional penalty for that specific event, as it's part of the evolution domain.
                else: # Integration failed (status 2)
                    # print(f"ODE integration failed: {solution.message}")
                    reward -= 150.0 # Severe penalty for integration failure
                    terminated = True

            except Exception as e:
                # print(f"Error during ODE integration: {e}")
                reward -= 200.0 # Very severe penalty for unexpected errors
                terminated = True

        # --- 3. Check Problem Postcondition ---
        # (x - xo)^2 + (y - yo)^2 > 0
        # This means the robot must NOT be at the obstacle's position.
        # If it becomes <= 0, safety is violated.
        # Postcondition is a raw KeYmaera X string, parsed at runtime by _evaluate_expression
        postcondition_satisfied = self._evaluate_expression(f'(x - xo)^2 + (y - yo)^2 > 0', is_constraint=True)
        if not postcondition_satisfied:
            reward -= 100.0 # Safety violation
            terminated = True
        else:
            # Small positive reward for staying safe, or for making progress
            reward += 0.1 # Small survival reward

        # --- 4. Check Truncation ---
        if self.current_step >= self.max_episode_steps:
            truncated = True
            reward -= 10.0 # Small penalty for truncating without reaching goal

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # In a real environment, this would draw the robot, obstacle, etc.
            # For now, just print state for demonstration.
            print(f"Render: Step={self.current_step}, State={self.state}")

    def close(self):
        # No specific resources to close for this environment
        pass
