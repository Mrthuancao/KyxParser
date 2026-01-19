
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import sympy
from typing import List, Dict, Any, Optional, Union, Tuple

class CarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.state = {} # Internal state dictionary
        self.current_time = 0.0
        self.terminated = False
        self.truncated = False
        self.reward = 0.0

        # --- Store configuration for step method ---
        self.step_func_name = "car_dynamics"
        self.step_func_duration = float(self._eval_expr('0.1', {})) # Default to 0.1 if not specified

        # --- Generated Definitions (Constants in __init__) ---


        self.constants_names = []

        # Store generated functions/predicates as callables for _eval_expr context
        self.kx_functions = {}
        for name in []:
            self.kx_functions[name] = getattr(self, name)

        self.kx_predicates = {}
        for name in []:
            self.kx_predicates[name] = getattr(self, name)

        # --- Generated Observation Space ---
        # Note: Why there is only two variable here
        self.observation_space = spaces.Box(
            low=np.array([self._eval_expr('0.0', {}), self._eval_expr('-10.0', {})], dtype=np.float32),
            high=np.array([self._eval_expr('100.0', {}), self._eval_expr('10.0', {})], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_vars = ['x', 'v']

        # --- Generated Action Space (part 1: definition in __init__) ---
        self.action_space = spaces.Dict({'assign_a': spaces.Box(low=self._eval_expr('-1.0', {}), high=self._eval_expr('1.0', {}), shape=(), dtype=np.float32)})



    def _apply_agent_action(self, action):
        # Apply agent's chosen action to the environment state
        # Returns True if action was valid, False otherwise (e.g., violated constraint)
        # Note: This need to be more 
        self.terminated = False # Reset termination flag for action application
        self.reward = 0.0 # Reset reward for action application
        self.state['a'] = action['assign_a']
        return not self.terminated # Return True if no termination occurred due to action

    def _eval_expr(self, expr_str: str, context: dict) -> Any:
        
        if expr_str == "None": # Handle cases where Optional[str] is None
            return None

        # Prepare the full context for evaluation
        # This includes numerical values for variables/constants
        # and callable Python functions for kx_functions/kx_predicates
        full_context = {**context}
        for k_name in self.constants_names:
            full_context[k_name] = getattr(self, k_name)
        full_context.update(self.kx_functions)
        full_context.update(self.kx_predicates)

        # Create a dictionary of sympy symbols for variables and constants
        # and sympy Function objects for the defined functions/predicates.
        # This allows sympy.sympify to correctly parse expressions like 'func(x,y)'.
        sympy_namespace = {}
        for k, v in full_context.items():
            if isinstance(v, (int, float, np.number, bool)): # Include bool for direct evaluation
                sympy_namespace[k] = sympy.symbols(k)
            elif callable(v):
                sympy_namespace[k] = sympy.Function(k)
            else:
                # Fallback for other types, treat as symbol
                sympy_namespace[k] = sympy.symbols(k)

        # Replace common operators if necessary
        expr_str = expr_str.replace('^', '**')

        try:
            # Parse the expression using the custom sympy_namespace
            sympy_expr = sympy.sympify(expr_str, locals=sympy_namespace)

            # Separate free symbols into variables/constants and functions
            free_symbols_in_expr = list(sympy_expr.free_symbols)
            function_symbols_in_expr = [s for s in free_symbols_in_expr if isinstance(s, sympy.Function)]
            variable_symbols_in_expr = [s for s in free_symbols_in_expr if not isinstance(s, sympy.Function)]

            # Prepare arguments and values for lambdify
            lambdify_args = variable_symbols_in_expr + function_symbols_in_expr
            
            # Values for variables/constants
            lambdify_values = [full_context[str(s)] for s in variable_symbols_in_expr]
            
            # Modules for functions/predicates (Python callables)
            lambdify_modules = {str(s): full_context[str(s)] for s in function_symbols_in_expr}
            
            if not lambdify_args:
                # If no free symbols or function calls, just evaluate numerically
                # Use subs for constants if any, then evalf
                subs_dict_for_evalf = {sympy_namespace[k]: v for k, v in full_context.items() if isinstance(v, (int, float, np.number, bool))}
                result = sympy_expr.evalf(subs=subs_dict_for_evalf)
            else:
                # Create a numerical function from the sympy expression
                # The 'modules' argument allows passing Python callables for functions
                numerical_func = sympy.lambdify(lambdify_args, sympy_expr, modules=[lambdify_modules, 'numpy'])
                result = numerical_func(*lambdify_values)

            # Convert boolean sympy results to Python bool
            if result is sympy.true: return True
            if result is sympy.false: return False
            return result
        except (sympy.SympifyError, SyntaxError, TypeError, NameError) as e:
            print(f"Error evaluating expression '{expr_str}' with context '{full_context}': {e}")
            # For robustness, return False for boolean expressions on error, or raise an exception
            return False # Default to False on error for boolean contexts

    # --- Generated ODE System and Event Functions ---
    def _ode_system(self, t, y):
        state_dict = {k: v for k, v in zip(self.observation_vars, y)}
        return np.array([self._eval_expr('v', state_dict), self._eval_expr('a', state_dict)], dtype=np.float32)

    def _ode_event(self, t, y):
        state_dict = {k: v for k, v in zip(self.observation_vars, y)}
        # Event occurs when evolution domain becomes false (crosses 0 from positive)
        # Note: need to take a look at this 
        return self._eval_expr('(True)', state_dict)
        _ode_event.terminal = True
        _ode_event.direction = -1

    def _integrate_dynamics(self, dt: float):
        y0 = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)
        t_span = (0, dt)
        sol = solve_ivp(self._ode_system, t_span, y0, events=self._ode_event, dense_output=True, rtol=1e-6, atol=1e-9)
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


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize state variables
        self.state['x'] = float(self._eval_expr('0.0', {}))
        self.state['v'] = float(self._eval_expr('-10.0', {}))
        self.state['a'] = float(self._eval_expr('-1.0', {}))

        # Apply precondition (e.g., sample initial state satisfying it)
        # For now, we assume precondition defines a fixed initial state or is satisfied by default init.
        # A more advanced generator would sample until self._eval_expr('x == 0 and v == 0', self.state) is True.
        if not self._eval_expr('x == 0 and v == 0', self.state):
            print(f"Warning: Initial state does not satisfy precondition: x == 0 and v == 0")
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
        num_loop_iterations_str = '1'
        num_loop_iterations = int(self._eval_expr(num_loop_iterations_str, self.state))
        if num_loop_iterations <= 0: num_loop_iterations = 1 # Default to at least one iteration

        for _ in range(num_loop_iterations):
            # Process each action in the loop_body
            if self.terminated: break # Break if previous action/integration terminated
            self._integrate_dynamics(self.step_func_duration)
            if self.terminated: break # Break outer loop if terminated by inner actions/integration


        # 3. Evaluate postcondition, truncation, and return
        # Evaluate postcondition for reward and termination
        post_eval_result = self._eval_expr('x >= 100 or x < 0', self.state)
        if post_eval_result:
            self.reward += 1.0 # Example: reward for satisfying postcondition
            self.terminated = True # Example: termination upon satisfying postcondition
        else:
            self.reward -= 0.01 # Example: small penalty per step

        # Check for truncation (e.g., max steps, max time)
        if self.current_time >= self._eval_expr('0.1', {}):
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
