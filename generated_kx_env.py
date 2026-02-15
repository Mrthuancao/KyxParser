
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import math # For math functions like sqrt, sin, cos, etc.
import textwrap
import re # For regex replacements in _evaluate_expression
from ast_config import (ChoiceActionConfig, SequenceActionConfig, DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig, ActionConstraintConfig, CallHybridProgramConfig, HybridProgramStepConfig) # Import AST classes

class IJRR17_Corollary_1_1_Velocity_controlled_passive_safety(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- KeYmaera X Definitions ---
        self.constants = {}
        self.functions = {}
        self.predicates = {}

        # Constants
        self.constants['ep'] = 1.0
        self.constants['b'] = 0.5
        self.constants['A'] = 1.0
        self.constants['W'] = 1.0
        self.constants['V'] = 1.0

        # Functions (lambdas for easy evaluation)
        self.functions['stopDist'] = lambda v: v**2 / (2*self.constants['b'])
        self.functions['accelComp'] = lambda v: (self.constants['A']/self.constants['b'] + 1) * (self.constants['A']/2 * self.constants['ep']**2 + self.constants['ep']*v)
        self.functions['admissibleSeparation'] = lambda v: self.functions['stopDist'](v) + self.functions['accelComp'](v)

        # Predicates (lambdas for easy evaluation)
        self.predicates['isWellformedDir'] = lambda dx, dy: dx**2 + dy**2 == 1
        self.predicates['bounds'] = lambda : self.constants['A'] >= 0 and self.constants['b'] > 0 and self.constants['ep'] > 0 and self.constants['V'] >= 0
        self.predicates['initialState'] = lambda x, y, v, dx, dy, xo, yo: v == 0 and (x-xo)**2 + (y-yo)**2 > 0 and self.predicates['isWellformedDir'](dx,dy)
        self.predicates['loopinv'] = lambda x, y, v, dx, dy, xo, yo: v >= 0 and self.predicates['isWellformedDir']((not dx,dy) and (v>0) or np.abs(x-xo) > self.functions['stopDist'](v) or np.abs(y-yo) > self.functions['stopDist'](v))
        self.predicates['assumptions'] = lambda x, y, v, dx, dy, xo, yo: self.predicates['bounds']() and self.predicates['initialState'](x,y,v,dx,dy,xo,yo)

        # --- State Variables ---
        # Mapping of state variable names to their index in the observation array
        self._state_keys = ['x', 'y', 'v', 'a', 'dx', 'dy', 'w', 'r', 'xo', 'yo', 'vxo', 'vyo', 't']
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
                'choice_ChoiceLevel_0': spaces.Discrete(2),
                'assign_vxo': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'assign_vyo': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'choice_ChoiceLevel_1': spaces.Discrete(2),
                'assign_v': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'assign_w': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'assign_r': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'assign_xo': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'assign_yo': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                    })
        self._action_space_keys = ['choice_ChoiceLevel_0', 'assign_vxo', 'assign_vyo', 'choice_ChoiceLevel_1', 'assign_v', 'assign_w', 'assign_r', 'assign_xo', 'assign_yo']
        self._control_func_action_config = ChoiceActionConfig(level_name='ChoiceLevel_0', choices=[
            SequenceActionConfig(actions=[
                NondeterministicAssignmentActionConfig(variable='vxo'),
                NondeterministicAssignmentActionConfig(variable='vyo'),
                ActionConstraintConfig(formula='vxo**2+vyo**2<=V**2')
            ]),
            ChoiceActionConfig(level_name='ChoiceLevel_1', choices=[
                SequenceActionConfig(actions=[
                    DeterministicAssignmentActionConfig(variable='v', value='0'),
                    DeterministicAssignmentActionConfig(variable='w', value='0')
                ]),
                SequenceActionConfig(actions=[
                    NondeterministicAssignmentActionConfig(variable='v'),
                    ActionConstraintConfig(formula='0<=v'),
                    NondeterministicAssignmentActionConfig(variable='w'),
                    ActionConstraintConfig(formula='-W<=w and w<=W'),
                    NondeterministicAssignmentActionConfig(variable='r'),
                    NondeterministicAssignmentActionConfig(variable='xo'),
                    NondeterministicAssignmentActionConfig(variable='yo'),
                    ActionConstraintConfig(formula='r!=0 and r*w == v'),
                    ActionConstraintConfig(formula='np.abs(x-xo) > admissibleSeparation(v) or np.abs(y-yo) > admissibleSeparation(v)')
                ])
            ])
        ]) # Stored for step logic

        # --- ODE Solver Setup (from stepFunc) ---
        # Mapping from state variable names to their index in the ODE solver's y array
        self._ode_state_map = {'x': 0, 'y': 1, 'v': 2, 'a': 3, 'dx': 4, 'dy': 5, 'w': 6, 'r': 7, 'xo': 8, 'yo': 9, 'vxo': 10, 'vyo': 11, 't': 12}
        self._ode_state_idx_map = {0: 'x', 1: 'y', 2: 'v', 3: 'a', 4: 'dx', 5: 'dy', 6: 'w', 7: 'r', 8: 'xo', 9: 'yo', 10: 'vxo', 11: 'vyo', 12: 't'} # Map index to name

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
                vxo = y_array[10]
                vyo = y_array[11]
                t = y_array[12]

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
                    0, # dw/dt
                    0, # dr/dt
                    vxo, # dxo/dt
                    vyo, # dyo/dt
                    0, # dvxo/dt
                    0, # dvyo/dt
                    1, # dt/dt
                ]
                return dydt
        '''
        self._ode_events_str = '''
            def event_t_max(t_val, y_array):
                # Event: t exceeds ep
                # Returns 0 when t == ep
                return self.constants['ep'] - y_array[12]
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

    def _evaluate_expression(self, expr_str, local_vars=None, is_constraint=False):

        # Safe evaluation of expressions using environment's context.
        # If is_constraint is True, returns False on evaluation error.

        if local_vars is None:
            local_vars = {}

        # Strip KeYmaera X constraint prefix '?' if present
        if expr_str.startswith('?'):
            expr_str = expr_str[1:].strip()

        # Combine all available variables for evaluation context
        eval_globals = {
            'np': np, 'math': math,
            **self.constants,
            **self.functions, # Include functions for general expressions
            **self.predicates  # Include predicates for general expressions
        }
        eval_locals = {
            **self.state,
            **local_vars
        }

        # Apply the same transformations as in _format_kx_expr_for_python_lambda_body
        # to ensure consistency when evaluating expressions at runtime.
        expr_py = expr_str.strip()

        # 1. Basic syntax replacements (power, abs)
        expr_py = expr_py.replace('^', '**').replace('abs(', 'np.abs(')

        # 2. Replace single '=' with '==' for equality comparisons.
        expr_py = re.sub(r'(?<![<>=!])=(?![=])', '==', expr_py)

        # 3. Handle logical operators (&, |)
        expr_py = expr_py.replace('&', 'and').replace('|', 'or')

        # 4. Handle KeYmaera X implication '->' (A -> B becomes (not (A) or (B)))
        while True:
            # Backreferences \1 and \2 are double-escaped because this string
            # is itself a string literal within the generated Python code.
            new_expr_py = re.sub(r'\((.*?)\s*->\s*(.*?)\)', r'((not \1) or \2)', expr_py)
            if new_expr_py == expr_py:
                break
            expr_py = new_expr_py

        try:
            return eval(expr_py, eval_globals, eval_locals)
        except Exception as e:
            print(f"Error evaluating expression '{expr_str}': {e}")
            print(f"Eval globals: {eval_globals}")
            print(f"Eval locals: {eval_locals}")
            if is_constraint:
                return False # For constraints, return False on error
            raise # Re-raise for non-constraint expressions

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
                constraint_met = self._evaluate_expression(sub_action_config.formula, is_constraint=True)
                if not constraint_met:
                    # print(f"Constraint violated: {sub_action_config.formula}")
                    reward -= 100.0 # Penalty for invalid action
                    terminated = True
                    return reward, terminated, False # False means action was not fully applied
            elif isinstance(sub_action_config, DeterministicAssignmentActionConfig):
                # Apply deterministic assignment
                # The value string is already pre-formatted to be Python-compatible
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
                        constraint_met = self._evaluate_expression(inner_action.formula, is_constraint=True)
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
        self.state['vxo'] = 0.0
        self.state['vyo'] = 0.0
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
        t_span = (0, self.constants['ep']) # Integrate up to time limit 'ep'

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
                if self.state['v'] < 0:
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
        postcondition_satisfied = self._evaluate_expression('((not \1) or \2)**2 + (y - yo)**2 > 0)', is_constraint=True)
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
