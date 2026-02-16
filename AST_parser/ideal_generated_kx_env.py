import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import math # For math functions like sqrt, sin, cos, etc.
import textwrap
import re # For regex replacements in _evaluate_expression
from AST_parser.pyast_config import (ChoiceActionConfig, SequenceActionConfig, DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig, ActionConstraintConfig, CallHybridProgramConfig) # Import AST classes

class IJRR17_Theorem_1__Static_safety(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- KeYmaera X Definitions ---
        self.constants = {}
        self.functions = {}
        self.predicates = {}

        # Constants
        self.constants['ep'] = 0.1 # Time horizon for ODE, must be > 0
        self.constants['b'] = 2.0  # Braking deceleration, must be > 0
        self.constants['A'] = 1.0  # Acceleration, must be >= 0
        self.constants['W'] = 1.0  # Max angular velocity, typically a positive bound

        # Functions (lambdas for easy evaluation)
        self.functions['stopDist'] = lambda v: self._evaluate_expression('v**2 / (2*b)', {'v': v})
        self.functions['accelComp'] = lambda v: self._evaluate_expression('(A/b + 1) * (A/2 * ep**2 + ep*v)', {'v': v})
        self.functions['admissibleSeparation'] = lambda v: self._evaluate_expression('stopDist(v) + accelComp(v)', {'v': v})

        # Predicates (lambdas for easy evaluation)
        self.predicates['isWellformedDir'] = lambda dx, dy: self._evaluate_expression('dx**2 + dy**2 = 1', {'dx': dx, 'dy': dy})
        self.predicates['bounds'] = lambda : self._evaluate_expression('A >= 0     & b > 0     & ep > 0', {})

        # --- State Variables ---
        # Mapping of state variable names to their index in the observation array
        self._state_keys = ['x', 'y', 'v', 'a', 'dx', 'dy', 'w', 'r', 'xo', 'yo', 't']
        # Current state values (dictionary for easy access by name)
        self.state = {key: 0.0 for key in self._state_keys} 
        # Initial state values are set in reset()

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
        self._control_func_action_config = ChoiceActionConfig(level_name='ChoiceLevel_0', choices=[
            SequenceActionConfig(actions=[
                DeterministicAssignmentActionConfig(variable='a', value='-b')
            ]),
            SequenceActionConfig(actions=[
                ActionConstraintConfig(formula='?v = 0'),
                DeterministicAssignmentActionConfig(variable='a', value='0'),
                DeterministicAssignmentActionConfig(variable='w', value='0')
            ]),
            SequenceActionConfig(actions=[
                DeterministicAssignmentActionConfig(variable='a', value='A'),
                NondeterministicAssignmentActionConfig(variable='w'),
                ActionConstraintConfig(formula='?-W<=w & w<=W'),
                NondeterministicAssignmentActionConfig(variable='r'),
                NondeterministicAssignmentActionConfig(variable='xo'),
                NondeterministicAssignmentActionConfig(variable='yo'),
                # MODIFICATION HERE: Use np.abs and a small tolerance for floating point equality
                ActionConstraintConfig(formula='?r!=0 & np.abs(r*w - v) < 1e-9'), 
                ActionConstraintConfig(formula='? abs(x-xo) > admissibleSeparation(v)         | abs(y-yo) > admissibleSeparation(v)')
            ])
        ]) # Stored for step logic

        # --- ODE Solver Setup (from stepFunc) ---
        # Mapping from state variable names to their index in the ODE solver's y array
        self._ode_state_map = {'x': 0, 'y': 1, 'v': 2, 'a': 3, 'dx': 4, 'dy': 5, 'w': 6, 'r': 7, 'xo': 8, 'yo': 9, 't': 10}
        self._ode_state_idx_map = {0: 'x', 1: 'y', 2: 'v', 3: 'a', 4: 'dx', 5: 'dy', 6: 'w', 7: 'r', 8: 'xo', 9: 'yo', 10: 't'} # Map index to name

        # Define ODE function string, now taking 'env_instance' explicitly
        self._ode_func_str = '''
            def _raw_ode_func(env_instance, t_val, y_array):
                # Map y array to named variables for easier access
                _current_ode_state = {}
                for i, var_name in env_instance._ode_state_idx_map.items():
                    _current_ode_state[var_name] = y_array[i]

                # Calculate derivatives
                dydt = [
                    env_instance._evaluate_expression('v * dx', local_vars=_current_ode_state), # dx/dt for x
                    env_instance._evaluate_expression('v * dy', local_vars=_current_ode_state), # dy/dt for y
                    env_instance._evaluate_expression('a', local_vars=_current_ode_state), # dv/dt for v
                    0.0, # da/dt (a is constant during ODE step as per problem spec)
                    env_instance._evaluate_expression('-w * dy', local_vars=_current_ode_state), # ddx/dt for dx
                    env_instance._evaluate_expression('w * dx', local_vars=_current_ode_state), # ddy/dt for dy
                    env_instance._evaluate_expression('a/r', local_vars=_current_ode_state), # dw/dt for w
                    0.0, # dr/dt (r is constant during ODE step as per problem spec)
                    0.0, # dxo/dt (xo is constant during ODE step as per problem spec)
                    0.0, # dyo/dt (yo is constant during ODE step as per problem spec)
                    env_instance._evaluate_expression('1', local_vars=_current_ode_state), # dt/dt for t
                ]
                return dydt
        '''
        # Define ODE events string, now taking 'env_instance' explicitly
        self._ode_events_str = '''
            def _raw_event_t_max(env_instance, t_val, y_array):
                # Event: t exceeds ep
                # Returns 0 when t == ep
                return env_instance.constants['ep'] - y_array[10]
            _raw_event_t_max.terminal = True
            _raw_event_t_max.direction = -1 # Trigger when ep - t goes from positive to negative (t > ep)

            def _raw_event_v_min(env_instance, t_val, y_array):
                # Event: v becomes negative
                # Returns 0 when v == 0
                return y_array[2]
            _raw_event_v_min.terminal = True
            _raw_event_v_min.direction = -1 # Trigger when v goes from positive to negative (v < 0)

            ode_events_list = [ # Renamed to avoid name collision with the compiled list
                _raw_event_t_max,
                _raw_event_v_min,
            ]
        '''

        # Compile ODE function and events dynamically
        self._ode_func_compiled = self._compile_ode_func(self._ode_func_str)
        self._ode_events_compiled = self._compile_ode_events(self._ode_events_str)

        # --- Episode Management ---
        self.current_step = 0
        self.max_episode_steps = 200 # Default max steps for an episode
        self.reward_range = (-float('inf'), float('inf'))

    def _compile_ode_func(self, func_str):
        exec_globals = {'np': np, 'math': math}
        exec_locals = {} # 'self' is not directly in locals, it's passed as an argument
        exec(textwrap.dedent(func_str), exec_globals, exec_locals)
        _raw_func = exec_locals['_raw_ode_func']
        # Return a wrapper that solve_ivp can call,
        # which then calls _raw_func with the environment instance (self)
        return lambda t_val, y_array: _raw_func(self, t_val, y_array)

    def _compile_ode_events(self, events_str):
        if not events_str.strip():
            return []
        exec_globals = {'np': np, 'math': math}
        exec_locals = {} # 'self' is not directly in locals, it's passed as an argument
        exec(textwrap.dedent(events_str), exec_globals, exec_locals)
        _raw_events_list = exec_locals['ode_events_list']
        
        # Wrap each raw event function to pass the environment instance (self)
        wrapped_events = []
        for _raw_event_func in _raw_events_list:
            # Create a lambda wrapper that explicitly passes 'self' (the environment instance)
            # as the first argument to the raw event function.
            wrapped_event = lambda t_val, y_array, raw_func=_raw_event_func: raw_func(self, t_val, y_array)
            
            # Copy attributes (terminal, direction) from the raw event function to the wrapped one
            # These attributes are essential for solve_ivp to correctly handle events.
            wrapped_event.terminal = getattr(_raw_event_func, 'terminal', False)
            wrapped_event.direction = getattr(_raw_event_func, 'direction', 0)
            wrapped_events.append(wrapped_event)
        return wrapped_events

    def _get_obs(self):
        return np.array([self.state[key] for key in self._state_keys], dtype=np.float32)

    def _get_info(self):
        return self.state.copy() # Return a copy of the current state dictionary

    def _evaluate_expression(self, expr_str, local_vars=None):
        """
        Evaluates a given expression string using the environment's state and global context.
        """
        if local_vars is None:
            local_vars = {}

        # Strip KeYmaera X constraint prefix '?' if present
        if expr_str.startswith('?'):
            expr_str = expr_str[1:].strip()

        # Start with a copy to build up the Python expression
        expr_py = expr_str

        # Replace KeYmaera X power operator '^' with Python's '**'
        expr_py = expr_py.replace('^', '**')

        # Convert 'abs(' to 'np.abs(' but be careful not to double-prefix
        # This regex replaces 'abs(' only if it's not immediately preceded by 'np.' or 'numpy.'
        # The `(?<!...)` is a negative lookbehind assertion.
        expr_py = re.sub(r'(?<!np\.)(?<!numpy\.)abs\(', 'np.abs(', expr_py)
        # Also ensure 'numpy.abs(' is consistently converted to 'np.abs('
        # This handles cases where the original string might have used 'numpy.abs('
        expr_py = expr_py.replace("numpy.abs(", "np.abs(")

        # Handle logical operators (assuming KeYmaera X uses '&' and '|' for logical AND/OR)
        expr_py = expr_py.replace('&', 'and').replace('|', 'or')

        # Replace standalone '=' with '==' for equality.
        # This regex ensures that '=' is not part of '<=', '>=', '!=' or '=='
        expr_py = re.sub(r'(?<![<>=!])=(?![=])', '==', expr_py)

        # Combine all available variables for evaluation context
        eval_globals = {
            '__builtins__': __builtins__, # Provides access to standard built-in functions like abs, max, min
            'np': np,                     # Makes numpy available as 'np'
            'math': math,                 # Makes math module available
            'numpy': np,                  # Makes 'numpy' also an alias for np, in case expressions use 'numpy.func'
            **self.constants,
            **self.functions,
            **self.predicates
        }
        # Prioritize local_vars (e.g., from ODE) over self.state for current evaluation
        eval_locals = {
            **self.state, # self.state provides default values
            **local_vars  # local_vars override self.state keys if they overlap
        }

        try:
            # print(f"DEBUG: Evaluating '{expr_str}' -> '{expr_py}' with locals: {local_vars}")
            result = eval(expr_py, eval_globals, eval_locals)
            # print(f"DEBUG: Result of '{expr_py}': {result} (type: {type(result)})")
            return result
        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR IN _evaluate_expression: '{expr_str}' (processed to '{expr_py}'): {e}")
            print(f"Eval globals: {eval_globals}")
            print(f"Eval locals: {eval_locals}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise # Re-raise the exception to see the full traceback and pinpoint the exact expression

    def _apply_control_action(self, action):
        reward = 0
        terminated = False

        # This structure mirrors the _control_func_action_config generated earlier
        # It's a ChoiceActionConfig at the top level

        # Get the chosen branch index
        choice_idx = action['choice_ChoiceLevel_0']

        # Access the configuration for the chosen branch
        chosen_branch_config = self._control_func_action_config.choices[choice_idx]

        # Process actions within the chosen branch
        for sub_action_config in chosen_branch_config.actions:
            if isinstance(sub_action_config, ActionConstraintConfig):
                # Evaluate constraint. If false, action is invalid.
                constraint_met = self._evaluate_expression(sub_action_config.formula)
                if not constraint_met:
                    # print(f"Constraint violated: {sub_action_config.formula}")
                    reward -= 100.0 # Penalty for invalid action
                    terminated = True
                    return reward, terminated, False # False means action was not fully applied
            elif isinstance(sub_action_config, DeterministicAssignmentActionConfig):
                # Apply deterministic assignment
                assigned_value = self._evaluate_expression(sub_action_config.value)
                self.state[sub_action_config.variable] = assigned_value
            elif isinstance(sub_action_config, NondeterministicAssignmentActionConfig):
                # Apply nondeterministic assignment from agent's action
                assign_key = f"assign_{sub_action_config.variable}"
                if assign_key in action:
                    self.state[sub_action_config.variable] = action[assign_key]
                else:
                    # This should not happen if action space is constructed correctly
                    # print(f"Warning: Nondeterministic assignment for {sub_action_config.variable} not found in action.")
                    reward -= 50.0 # Penalty for incomplete action
                    terminated = True
                    return reward, terminated, False
            elif isinstance(sub_action_config, CallHybridProgramConfig):
                # This environment doesn't support nested HP calls directly in controlFunc
                # other than the main controlFunc itself.
                # For this specific example, controlFunc is the top-level discrete action.
                pass # Or raise an error if unexpected
            elif isinstance(sub_action_config, SequenceActionConfig):
                # If there's a sequence within a choice, process its actions
                # This case handles the structure created by _parse_kx_hp_program_fragment_to_action_config
                # when it returns a SequenceActionConfig for a choice branch.
                for inner_action in sub_action_config.actions:
                    if isinstance(inner_action, ActionConstraintConfig):
                        constraint_met = self._evaluate_expression(inner_action.formula)
                        if not constraint_met:
                            reward -= 100.0
                            terminated = True
                            return reward, terminated, False
                    elif isinstance(inner_action, DeterministicAssignmentActionConfig):
                        assigned_value = self._evaluate_expression(inner_action.value)
                        self.state[inner_action.variable] = assigned_value
                    elif isinstance(inner_action, NondeterministicAssignmentActionConfig):
                        assign_key = f"assign_{inner_action.variable}"
                        if assign_key in action:
                            self.state[inner_action.variable] = action[assign_key]
                        else:
                            reward -= 50.0
                            terminated = True
                            return reward, terminated, False

        # Apply t := 0; after controlFunc, as specified in the problem loop body
        self.state['t'] = 0.0

        return reward, terminated, True # True means action was successfully applied

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # --- Initial State (based on Problem Precondition and defaults) ---
            # Default initial values (can be randomized or fixed based on problem)
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

        # Ensure dx, dy represent a unit vector initially
        # This is a common requirement for orientation vectors
        norm = np.sqrt(self.state['dx']**2 + self.state['dy']**2)
        if norm > 1e-6: # Avoid division by zero
            self.state['dx'] /= norm
            self.state['dy'] /= norm
        else: # Default to (1,0) if initial dx, dy are too small
            self.state['dx'] = 1.0
            self.state['dy'] = 0.0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        print(f"--- Step {self.current_step} ---")
        print(f"Action: {action}")
        print(f"State before control: {self.state}")

        # --- 1. Apply Control Function (Discrete Action) ---
        control_reward, control_terminated, control_applied = self._apply_control_action(action)
        reward += control_reward
        terminated = control_terminated
        print(f"After control: reward={control_reward}, terminated={control_terminated}, applied={control_applied}")
        print(f"State after control: {self.state}")

        if not control_applied:
            print("Control action not applied due to constraint violation or missing assignment.")
            return self._get_obs(), reward, terminated, truncated, info

        # --- 2. Simulate Differential Equations (Continuous Dynamics) ---
        # Prepare initial state for ODE solver
        y0 = np.array([self.state[var_name] for var_name in self._ode_state_map.keys()], dtype=np.float64)
        t_span = (0, self.constants['ep']) # Integrate up to time limit 'ep'
        print(f"ODE y0: {y0}")
        print(f"ODE t_span: {t_span}")

        try:
            solution = solve_ivp(
                self._ode_func_compiled,
                t_span,
                y0,
                events=self._ode_events_compiled,
                dense_output=True,
                rtol=1e-6, atol=1e-8
            )
            # print(f"ODE solution status: {solution.status}")
            # print(f"ODE solution message: {solution.message}")
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
                    # Fallback if event data is missing, use last point of integration
                    # This might happen if an event is detected exactly at t_span[1]
                    final_y = solution.y[:, -1] 

                for i, var_name in enumerate(self._ode_state_map.keys()):
                    self.state[var_name] = final_y[i]

                # Check which event terminated it
                # For this problem, events are domain constraints (t <= ep, v >= 0)
                # If v < 0, it's a domain violation -> termination
                # We need to re-evaluate after updating self.state
                if self.state['v'] < 0:
                    reward -= 100.0 # Penalty for v < 0
                    terminated = True
                # If t > ep, it means the time limit was reached, which is expected behavior
                # so no additional penalty for that specific event, as it's part of the evolution domain.
            else: # Integration failed (status 2)
                print(f"ODE integration failed with status {solution.status}: {solution.message}")
                reward -= 150.0 # Severe penalty for integration failure
                terminated = True

        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR: Unexpected exception during ODE integration: {e}") 
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            reward -= 200.0 # Very severe penalty for unexpected errors
            terminated = True

        # --- 3. Check Problem Postcondition ---
        # (x - xo)^2 + (y - yo)^2 > 0
        # This means the robot must NOT be at the obstacle's position.
        # If it becomes <= 0, safety is violated.
        postcondition_satisfied = self._evaluate_expression('(x - xo)**2 + (y - yo)**2 > 0')
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
        print(f"Step {self.current_step}: Obs={observation}, Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # In a real environment, this would draw the robot, obstacle, etc.
            # For now, just print state for demonstration.
            print(f"Render: Step={self.current_step}, State={self.state}")

    def close(self):
        # No specific resources to close for this environment
        pass