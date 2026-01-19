
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import sympy
from typing import List, Dict, Any, Optional, Union, Tuple

# Import ast_config to make dataclass definitions available for _problem_loop_body_actions
import ast_config

class IJRR17_Theorem_1__Static_safety(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def stopDist(self, v):
        return self._eval_expr('v^2 / (2*b)', {'v': v})

    def accelComp(self, v):
        return self._eval_expr('(A/b + 1) * (A/2 * ep^2 + ep*v)', {'v': v})

    def admissibleSeparation(self, v):
        return self._eval_expr('stopDist(v) + accelComp(v)', {'v': v})

    def isWellformedDir(self, dx, dy):
        return bool(self._eval_expr('dx^2 + dy^2 = 1', {'dx': dx, 'dy': dy}))

    def bounds(self, ):
        return bool(self._eval_expr('A >= 0     & b > 0     & ep > 0', {}))

    def assumptions(self, x, y, v, dx, dy, xo, yo):
        return bool(self._eval_expr('True', {'x': x, 'y': y, 'v': v, 'dx': dx, 'dy': dy, 'xo': xo, 'yo': yo}))


    def _eval_expr(self, expr_str: str, context: dict) -> Any:

        if expr_str == "None": # Handle cases where Optional[str] is None
            return None
    
        # Explicitly handle infinity strings to return native Python floats for Gymnasium compatibility.
        # This bypasses potential issues with sympy.sympify converting 'np.inf' to native float
        # and ensures that the final return value for bounds is a float, not a SymPy object.
        if expr_str == 'np.inf' or expr_str == 'inf':
            return float('inf')
        if expr_str == '-np.inf' or expr_str == '-inf':
            return float('-inf')
    
        # Prepare the full context for evaluation
        full_context = {**context}
    
        # Add already-initialized constants from 'self' to the context.
        # It's crucial to check if the attribute exists (hasattr) before trying to get it (getattr).
        # This prevents the AttributeError during the initial assignment of constants.
        for k_name in self.constants_names:
            if hasattr(self, k_name): # <-- Only add if the constant attribute has been set
                full_context[k_name] = getattr(self, k_name)
    
        # Add functions and predicates
        full_context.update(self.kx_functions)
        full_context.update(self.kx_predicates)
    
        full_context['np'] = np # Add numpy module to context for evaluation

        print(f"DEBUG: _eval_expr called with expr_str='{expr_str}'")
        print(f"DEBUG: full_context keys: {full_context.keys()}")
        if 'assumptions' in full_context:
            print(f"DEBUG: full_context['assumptions'] type: {type(full_context['assumptions'])}, callable: {callable(full_context['assumptions'])}")

        # Create a dictionary of sympy symbols for variables and constants,
        # and sympy Function objects for the defined functions/predicates.
        # This dictionary will be passed as 'locals' to sympy.sympify.
        sympy_locals_for_sympify = {}
        for k, v in full_context.items():
            if k == 'np':
                sympy_locals_for_sympify[k] = v
            elif callable(v):
                # CRITICAL FIX: Use sympy.symbols(k, cls=sympy.Function) to explicitly create a function symbol
                # This helps sympify correctly parse function calls like 'assumptions(...)'.
                sympy_locals_for_sympify[k] = sympy.symbols(k, cls=sympy.Function)
            else:
                # Use sympy.Symbol for explicit symbol creation for variables/constants
                sympy_locals_for_sympify[k] = sympy.Symbol(k)
    
        print(f"DEBUG: sympy_locals_for_sympify keys: {sympy_locals_for_sympify.keys()}")
        if 'assumptions' in sympy_locals_for_sympify:
            print(f"DEBUG: sympy_locals_for_sympify['assumptions'] type: {type(sympy_locals_for_sympify['assumptions'])}")

        # Add sympy.oo for 'inf' if it appears directly in expressions (e.g., "inf" instead of "np.inf")
        # This is now less critical due to direct handling above, but good for other cases.
        sympy_locals_for_sympify['inf'] = sympy.oo
        sympy_locals_for_sympify['oo'] = sympy.oo

        # Replace common operators if necessary
        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace('=', '==') # Assuming '=' in formulas means equality check
        expr_str = expr_str.replace('&', 'and')
        expr_str = expr_str.replace('|', 'or')
        # Remove leading '?' if it's a KeYmaera X specific syntax
        expr_str = expr_str.lstrip('?')

        try:
            # Parse the expression using the custom sympy_locals_for_sympify
            raw_sympify_result = sympy.sympify(expr_str, locals=sympy_locals_for_sympify)

            # Ensure the result is a SymPy expression object.
            # If sympify unexpectedly returned a native Python type (e.g., float, int, bool)
            # for a constant expression, convert it back to a SymPy object.
            if not isinstance(raw_sympify_result, sympy.Expr):
                sympy_expr = sympy.sympify(raw_sympify_result) # Convert native type to SymPy object
            else:
                sympy_expr = raw_sympify_result

            print(f"DEBUG: After sympify, expr_str='{expr_str}', sympy_expr={sympy_expr}, type={type(sympy_expr)}")
            print(f"DEBUG: Is sympy_expr a sympy.Function? {isinstance(sympy_expr, sympy.Function)}")
            if isinstance(sympy_expr, sympy.Function):
                print(f"DEBUG: sympy_expr.func={sympy_expr.func}, type={type(sympy_expr.func)}")
                print(f"DEBUG: Is sympy_expr.func a sympy.Function? {isinstance(sympy_expr.func, sympy.Function)}")

            # --- NEW LOGIC: Handle direct function/predicate calls ---
            # If the entire expression is a direct call to a user-defined function/predicate,
            # call the Python callable directly instead of lambdifying the whole expression.
            # This avoids NameError when lambdify tries to generate a function that calls 'assumptions' directly.
            # Check if the expression is an "applied undefined function" (i.e., a function call)
            if isinstance(sympy_expr, sympy.core.function.AppliedUndef):
                func_name = str(sympy_expr.func)
                print(f"DEBUG: Direct call candidate: func_name='{func_name}' (from AppliedUndef)")
                if func_name in self.kx_functions or func_name in self.kx_predicates:
                    print(f"DEBUG: Direct call confirmed for '{func_name}'")
                    python_callable = full_context[func_name]
                    arg_values = []
                    for arg_sym in sympy_expr.args:
                        # Recursively evaluate each argument. This handles cases where arguments are also expressions.
                        # It's important to convert arg_sym back to string for recursive _eval_expr call
                        arg_values.append(self._eval_expr(str(arg_sym), full_context))
                    return python_callable(*arg_values)
                else:
                    print(f"DEBUG: Direct call skipped: '{func_name}' not in kx_functions/predicates. Proceeding to lambdify.")
            else:
                print(f"DEBUG: Not an AppliedUndef. Proceeding to lambdify.")

                # Identify all free symbols in the expression that are not functions themselves.
                # These are the variables that need to be passed as arguments to the lambdified function.
                variable_symbols_in_expr = [s for s in sympy_expr.free_symbols if not isinstance(s, sympy.Function)]
                lambdify_args = variable_symbols_in_expr

                # Prepare the modules dictionary for lambdify.
                # This dictionary maps *string names* of SymPy functions/predicates
                # to their corresponding Python callable implementations.
                lambdify_modules_dict = {}
                for func_name, py_callable in self.kx_functions.items():
                    lambdify_modules_dict[func_name] = py_callable # Changed key from sympy.Function(func_name) to func_name (string)
                for pred_name, py_callable in self.kx_predicates.items():
                    lambdify_modules_dict[pred_name] = py_callable # Changed key from sympy.Function(pred_name) to pred_name (string)
            
                # Add numpy module for common mathematical functions (e.g., np.sin, np.cos)
                # This is important if expressions use numpy functions directly.
                # Note: 'numpy' can also be passed as a string in the modules list.
                # lambdify_modules_dict['numpy'] = np # This can cause issues if 'numpy' is also in the list below

                print(f"DEBUG: Lambdify path - lambdify_args: {[str(s) for s in lambdify_args]}")
                print(f"DEBUG: Lambdify path - lambdify_modules_dict keys: {[str(k) for k in lambdify_modules_dict.keys()]}\n")

                # Values for variables (corresponding to lambdify_args)
                lambdify_values = [full_context[str(s)] for s in variable_symbols_in_expr]

                # Create a numerical function from the sympy expression
                # The 'modules' argument allows passing Python callables for functions
                # and also includes 'numpy' for standard math functions.
                numerical_func = sympy.lambdify(lambdify_args, sympy_expr, modules=[lambdify_modules_dict, 'numpy'])
                return numerical_func(*lambdify_values)

        except (sympy.SympifyError, SyntaxError, TypeError, NameError) as e:
            # If any error occurs during sympify or lambdify setup, re-raise as ValueError.
            # This ensures that malformed expressions for bounds don't silently become booleans.
            raise ValueError(f"Error evaluating expression '{{expr_str}}' with context '{{full_context}}': {e}") from e

    def __init__(self):
        super().__init__()
        self.state = {} # Internal state dictionary
        self.current_time = 0.0
        self.terminated = False
        self.truncated = False
        self.reward = 0.0

        # --- A. Initialize lists/dicts of *names* and *callable references* ---
        # These are needed by _eval_expr to know *what* to look for.
        # They do NOT call _eval_expr themselves, so they can be set first.
        self.constants_names = ['ep', 'b', 'A', 'W']

        self.kx_functions = {}
        for name in ['stopDist', 'accelComp', 'admissibleSeparation']:
            if hasattr(self, name): # Ensure the method exists
                self.kx_functions[name] = getattr(self, name)
            else:
                raise AttributeError(f"Generated function '{name}' expected but not found as a method in class {self.__class__.__name__}. Check definition generation.")

        self.kx_predicates = {}
        for name in ['isWellformedDir', 'bounds', 'assumptions']:
            if hasattr(self, name): # Ensure the method exists
                self.kx_predicates[name] = getattr(self, name)
            else:
                raise AttributeError(f"Generated predicate '{name}' expected but not found as a method in class {self.__class__.__name__}. Check definition generation.")

        # Initialize ODE event function wrapper (does not call _eval_expr directly yet)
        self._ode_event_func = self._create_ode_event_function()

        # Store the problem's loop_body actions as a list of instantiated dataclass objects
        self._problem_loop_body_actions = [
            ast_config.DeterministicAssignmentActionConfig(variable='t', value='0'),
            ast_config.CallHybridProgramConfig(hp_name='stepFunction'),
            ast_config.ActionConstraintConfig(formula='t>=0, isWellformedDir(dx, dy)'),
            ast_config.ActionConstraintConfig(formula='t>=0 and isWellformedDir(dx, dy)')
        ]

        # --- B. Assign constant *values* ---
        # This block sets `self.W`, `self.R`, etc. These calls to _eval_expr
        # will now find `self.constants_names`, `self.kx_functions`, `self.kx_predicates` initialized.
        self.ep = self._eval_expr('0.0', {})
        self.b = self._eval_expr('0.0', {})
        self.A = self._eval_expr('0.0', {})
        self.W = self._eval_expr('0.0', {})

        # --- C. Now, other attributes can be safely initialized ---
        # These may call _eval_expr and use constants whose values are now set.
        self.step_func_name = "stepFunction"
        self.step_func_duration = float(self._eval_expr('1.0', {}))

        # Generated Observation Space (calls _eval_expr for bounds)
        self.observation_space = spaces.Box(
            low=np.array([self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {}), self._eval_expr('-np.inf', {})], dtype=np.float32),
            high=np.array([self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {}), self._eval_expr('np.inf', {})], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_vars = ['x', 'y', 'v', 'a', 'dx', 'dy', 'w', 'r', 'xo', 'yo', 't']

        # Generated Action Space (part 1: definition in __init__) (calls _eval_expr for bounds)
        self.action_space = spaces.Dict({'choice_ChoiceLevel_0': spaces.Discrete(3), 'assign_w': spaces.Box(low=self._eval_expr('-np.inf', {}), high=self._eval_expr('np.inf', {}), shape=(), dtype=np.float32), 'assign_r': spaces.Box(low=self._eval_expr('-np.inf', {}), high=self._eval_expr('np.inf', {}), shape=(), dtype=np.float32), 'assign_xo': spaces.Box(low=self._eval_expr('-np.inf', {}), high=self._eval_expr('np.inf', {}), shape=(), dtype=np.float32), 'assign_yo': spaces.Box(low=self._eval_expr('-np.inf', {}), high=self._eval_expr('np.inf', {}), shape=(), dtype=np.float32)})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize state variables
        self.state['x'] = float(self._eval_expr('0.0', {}))
        self.state['y'] = float(self._eval_expr('0.0', {}))
        self.state['v'] = float(self._eval_expr('0.0', {}))
        self.state['a'] = float(self._eval_expr('0.0', {}))
        self.state['dx'] = float(self._eval_expr('0.0', {}))
        self.state['dy'] = float(self._eval_expr('0.0', {}))
        self.state['w'] = float(self._eval_expr('0.0', {}))
        self.state['r'] = float(self._eval_expr('0.0', {}))
        self.state['xo'] = float(self._eval_expr('0.0', {}))
        self.state['yo'] = float(self._eval_expr('0.0', {}))
        self.state['t'] = float(self._eval_expr('0.0', {}))

        # Apply precondition (e.g., sample initial state satisfying it)
        # For now, we assume precondition defines a fixed initial state or is satisfied by default init.
        # A more advanced generator would sample until self._eval_expr('assumptions(x, y, v, dx, dy, xo, yo)', self.state) is True.
        if not self._eval_expr('assumptions(x, y, v, dx, dy, xo, yo)', self.state):
            print(f"Warning: Initial state does not satisfy precondition: assumptions(x, y, v, dx, dy, xo, yo)")
            # Potentially re-sample or raise error, or terminate immediately
            # self.terminated = True # Example: Terminate if precondition not met

        self.current_time = 0.0
        self.terminated = False
        self.truncated = False
        self.reward = 0.0
        observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        # 1. Apply agent's discrete action (from control_func)
        action_valid = self._apply_agent_action(action)
        if not action_valid:
            # Action was invalid, environment terminated or heavily penalized
            observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)
            info = {"action_invalid": True}
            return observation, self.reward, self.terminated, self.truncated, info

        # Execute the problem's loop_body (environment's internal evolution)
        num_loop_iterations_str = 'inf'
        if num_loop_iterations_str == 'inf':
            num_loop_iterations = 1 # Assuming one integration step if 'inf' is specified
        else:
            num_loop_iterations = int(self._eval_expr(num_loop_iterations_str, self.state))
        if num_loop_iterations <= 0: num_loop_iterations = 1 # Default to at least one iteration

        for _ in range(num_loop_iterations):
            current_dt = self.step_func_duration / num_loop_iterations
            # Process each action defined in the problem's loop_body
            for loop_action_item in self._problem_loop_body_actions:
                self.state['t'] = float(self._eval_expr('0', self.state))
                if self.terminated: break # Break loop if terminated by inner actions/integration
                self._integrate_dynamics(current_dt)
                if self.terminated: break # Break loop if terminated by inner actions/integration
                if not self._eval_expr('t>=0, isWellformedDir(dx, dy)', self.state):
                    self.terminated = True
                    self.reward -= 100.0 # Penalty for violating loop constraint
                    break # Exit loop if constraint violated
                if self.terminated: break # Break loop if terminated by inner actions/integration
                if not self._eval_expr('t>=0 and isWellformedDir(dx, dy)', self.state):
                    self.terminated = True
                    self.reward -= 100.0 # Penalty for violating loop constraint
                    break # Exit loop if constraint violated
                if self.terminated: break # Break loop if terminated by inner actions/integration


        # 3. Evaluate postcondition, truncation, and return
        # Evaluate postcondition for reward and termination
        post_eval_result = self._eval_expr('(x - xo)^2 + (y - yo)^2 > 0', self.state)
        if post_eval_result:
            self.reward += 1.0 # Example: reward for satisfying postcondition
            self.terminated = True # Example: termination upon satisfying postcondition
        else:
            self.reward -= 0.01 # Example: small penalty per step

        # Check for truncation (e.g., max steps, max time)
        if self.current_time >= self._eval_expr('1.0', {}):
            self.truncated = True
            self.reward -= 10.0 # Penalty for truncation

        observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)
        info = {}
        return observation, self.reward, self.terminated, self.truncated, info

    def render(self):
        # Implement rendering logic here (e.g., matplotlib, pygame)
        pass

    def close(self):
        # Clean up resources
        pass

    def _apply_agent_action(self, action):
        # Apply agent's chosen action to the environment state
        # Returns True if action was valid, False otherwise (e.g., violated constraint)
        self.terminated = False # Reset termination flag for action application
        self.reward = 0.0 # Reset reward for action application
        # Process choice for level 0
        chosen_idx = action['choice_ChoiceLevel_0']
        if not (0 <= chosen_idx < 3):
            raise ValueError(f'Invalid choice index {chosen_idx} for level 0')
        if chosen_idx == 0:
            self.state['a'] = float(self._eval_expr('-b', self.state))
        if chosen_idx == 1:
            if not self._eval_expr('?v = 0', self.state):
                self.terminated = True # Invalid action leads to termination
                self.reward -= 100.0 # Penalty for invalid action
                return False # Indicate invalid action
            self.state['a'] = float(self._eval_expr('0', self.state))
            self.state['w'] = float(self._eval_expr('0', self.state))
        if chosen_idx == 2:
            self.state['a'] = float(self._eval_expr('A', self.state))
            self.state['w'] = action['assign_w']
            if not self._eval_expr('?-W<=w & w<=W', self.state):
                self.terminated = True # Invalid action leads to termination
                self.reward -= 100.0 # Penalty for invalid action
                return False # Indicate invalid action
            self.state['r'] = action['assign_r']
            self.state['xo'] = action['assign_xo']
            self.state['yo'] = action['assign_yo']
            if not self._eval_expr('?r!=0 & r*w = v', self.state):
                self.terminated = True # Invalid action leads to termination
                self.reward -= 100.0 # Penalty for invalid action
                return False # Indicate invalid action
            if not self._eval_expr('? abs(x-xo) > admissibleSeparation(v)         | abs(y-yo) > admissibleSeparation(v)', self.state):
                self.terminated = True # Invalid action leads to termination
                self.reward -= 100.0 # Penalty for invalid action
                return False # Indicate invalid action
        return not self.terminated # Return True if no termination occurred due to action

    # --- Generated ODE System and Event Functions ---
    def _ode_system(self, t, y):
        state_dict = {k: v for k, v in zip(self.observation_vars, y)}
        return np.array([self._eval_expr('v * dx', state_dict), self._eval_expr('v * dy', state_dict), self._eval_expr('a', state_dict), 0.0, self._eval_expr('-w * dy', state_dict), self._eval_expr('w * dx', state_dict), self._eval_expr('a/r', state_dict), 0.0, 0.0, 0.0, self._eval_expr('1', state_dict)], dtype=np.float32)

    def _ode_event(self, t, y):
        state_dict = {k: v for k, v in zip(self.observation_vars, y)}
        # Event occurs when evolution domain becomes false (crosses 0 from positive)
        event_condition = self._eval_expr('(t <= ep) and (v >= 0)', state_dict)
        return event_condition

    def _create_ode_event_function(self):
        def event_wrapper(t, y):
            return self._ode_event(t, y)
        event_wrapper.terminal = True
        event_wrapper.direction = -1
        return event_wrapper

    def _integrate_dynamics(self, dt: float):
        y0 = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)
        t_span = (0, dt)
        sol = solve_ivp(self._ode_system, t_span, y0, events=self._ode_event_func, dense_output=True, rtol=1e-6, atol=1e-9)
        if sol.status == 0: # Integration successful, no event
            final_y = sol.y[:, -1]
            self.current_time += dt
        elif sol.status == 1: # Event occurred
            if sol.y_events[0].size > 0 and sol.t_events[0].size > 0:
                final_y = sol.y_events[0][-1]
                self.current_time += sol.t_events[0][-1]
            else:
                final_y = y0
                self.current_time += dt
            self.terminated = True
        else:
            final_y = y0
            self.terminated = True
            print(f"ODE integration failed: {{sol.message}}")

        for i, var_name in enumerate(self.observation_vars):
            self.state[var_name] = final_y[i]

