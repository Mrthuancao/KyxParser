import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import math # For math functions like sqrt, sin, cos, etc.
import textwrap
import re # For regex replacements in _evaluate_expression
import os # For sys.path debugging

# Assuming ast_config is in the same directory or accessible via PYTHONPATH
# If ast_config.py is in a different directory, you might need sys.path modifications
# as shown in the previous response. For this example, we assume it's directly importable.
from ast_config import (
    ChoiceActionConfig,
    SequenceActionConfig,
    DeterministicAssignmentActionConfig,
    NondeterministicAssignmentActionConfig,
    ActionConstraintConfig,
    CallHybridProgramConfig
)

class IJRR17_Theorem_1__Static_safety(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- KeYmaera X Definitions ---
        self.constants = {}
        self.functions = {}
        self.predicates = {}

        # Constants (will be set to reasonable defaults in reset if not changed externally)
        # FIX: Initialize W to a non-zero value directly in __init__
        self.constants['ep'] = 0.1 # Time horizon for ODE integration, default to 0.1
        self.constants['b'] = 1.0 # Deceleration constant, default to 1.0
        self.constants['A'] = 0.5 # Acceleration constant, default to 0.5
        self.constants['W'] = 0.5 # Max angular velocity for 'w', default to 0.5
        print(f"DEBUG: __init__ called. Initial self.constants['ep'] = {self.constants['ep']}")

        # Functions (lambdas for easy evaluation)
        # FIX: Make functions directly access self.constants and local_vars without recursive _evaluate_expression calls
        self.functions['stopDist'] = lambda v_val: (v_val**2 / (2 * self.constants['b']))
        self.functions['accelComp'] = lambda v_val: ((self.constants['A']/self.constants['b'] + 1) * (self.constants['A']/2 * self.constants['ep']**2 + self.constants['ep']*v_val))
        self.functions['admissibleSeparation'] = lambda v_val: self.functions['stopDist'](v_val) + self.functions['accelComp'](v_val)

        # Predicates (lambdas for easy evaluation)
        self.predicates['isWellformedDir'] = lambda dx, dy: (dx**2 + dy**2 == 1)
        self.predicates['bounds'] = lambda : (self.constants['A'] >= 0 and self.constants['b'] > 0 and self.constants['ep'] > 0)

        # --- State Variables ---
        # Mapping of state variable names to their index in the observation array
        self._state_keys = ['x', 'y', 'v', 'a', 'dx', 'dy', 'w', 'r', 'xo', 'yo', 't']
        # Current state values (dictionary for easy access by name)
        self.state = {key: 0.0 for key in self._state_keys} 
        # Initial state values are set in reset()
        print(f"DEBUG: __init__ called. Initial self.state = {self.state}")


        # --- Observation Space ---
        # All state variables are Real, so Box space is appropriate.
        # Bounds are infinite by default, as not specified in KeYmaera X input.
        low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)

        # --- Action Space (from controlFunc) ---
        # The action space is a Dict to handle choices and nondeterministic assignments.
        # The Box spaces for assign_w, assign_r, assign_xo, assign_yo represent the *raw*
        # inputs from the agent, which will then be transformed/clipped by the agent
        # or environment logic. Keeping them wide for now, as the agent's `choose_action`
        # handles the specific clipping/scaling.
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
                # FIX: Use approximate equality for floating point comparisons
                ActionConstraintConfig(formula='np.isclose(v, 0, atol=1e-6)'), # Modified to 'np.isclose'
                DeterministicAssignmentActionConfig(variable='a', value='0'),
                DeterministicAssignmentActionConfig(variable='w', value='0')
            ]),
            SequenceActionConfig(actions=[
                DeterministicAssignmentActionConfig(variable='a', value='A'),
                NondeterministicAssignmentActionConfig(variable='w'),
                ActionConstraintConfig(formula='-W<=w and w<=W'), # Modified to 'and'
                NondeterministicAssignmentActionConfig(variable='r'),
                NondeterministicAssignmentActionConfig(variable='xo'),
                NondeterministicAssignmentActionConfig(variable='yo'),
                # FIX: Use approximate equality for floating point comparisons
                ActionConstraintConfig(formula='r!=0 and np.isclose(r*w, v, atol=1e-6)'), # Modified to 'np.isclose'
                ActionConstraintConfig(formula='np.abs(x-xo) > admissibleSeparation(v) or np.abs(y-yo) > admissibleSeparation(v)') # Modified to 'or', 'np.abs'
            ])
        ]) # Stored for step logic

        # --- ODE Solver Setup (from stepFunc) ---
        # Mapping from state variable names to their index in the ODE solver's y array
        self._ode_state_map = {'x': 0, 'y': 1, 'v': 2, 'a': 3, 'dx': 4, 'dy': 5, 'w': 6, 'r': 7, 'xo': 8, 'yo': 9, 't': 10}
        self._ode_state_idx_map = {0: 'x', 1: 'y', 2: 'v', 3: 'a', 4: 'dx', 5: 'dy', 6: 'w', 7: 'r', 8: 'xo', 9: 'yo', 10: 't'} # Map index to name

        # Corrected _ode_func_str: Removed redundant 'self.' prefix
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
                # This is crucial for functions/predicates called within _evaluate_expression
                # that rely on the current state (e.g., admissibleSeparation)
                # This needs to be carefully managed. If functions directly use self.constants,
                # then this update is less critical for them. But if predicates use self.state, it's needed.
                # For now, keep it, but be aware of its implications.
                for i, var_name in self._ode_state_idx_map.items():
                    self.state[var_name] = y_array[i]

                # Calculate derivatives
                dydt = [
                    self._evaluate_expression('v * dx'), # dx/dt
                    self._evaluate_expression('v * dy'), # dy/dt
                    self._evaluate_expression('a'), # dv/dt
                    0.0, # da/dt (a is constant during ODE step)
                    self._evaluate_expression('-w * dy'), # ddx/dt
                    self._evaluate_expression('w * dx'), # ddy/dt
                    self._evaluate_expression('a/r'), # dw/dt (r is constant during ODE step)
                    0.0, # dr/dt (r is constant during ODE step)
                    0.0, # dxo/dt (xo is constant during ODE step)
                    0.0, # dyo/dt (yo is constant during ODE step)
                    self._evaluate_expression('1'), # dt/dt
                ]
                return dydt
        '''
        # Corrected _ode_events_str: Removed redundant 'self.' prefix
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
        self._ode_func_compiled = self._compile_ode_func(self._ode_func_str)
        self._ode_events_compiled = self._compile_ode_events(self._ode_events_str)

        # --- Episode Management ---
        self.current_step = 0
        self.max_episode_steps = 200 # Default max steps for an episode
        self.reward_range = (-float('inf'), float('inf'))

        # Agent reference, will be set by training script for constant updates
        self.agent = None 

    def _compile_ode_func(self, func_str):
        # Make the instance 'self' available in the global scope for the executed code
        exec_globals = {'np': np, 'math': math, 'self': self} # Pass self directly into globals
        # Use exec_globals for locals as well to ensure definitions are in the correct scope
        dedented_func_str = textwrap.dedent(func_str)
        # For debugging, you can uncomment the next line to see the string being executed:
        # print("DEBUG: Dedented ODE func string:\n", dedented_func_str)
        exec(dedented_func_str, exec_globals, exec_globals) 
        return exec_globals['ode_func'] # Retrieve from globals

    def _compile_ode_events(self, events_str):
        if not events_str.strip():
            return []
        # Make the instance 'self' available in the global scope for the executed code
        exec_globals = {'np': np, 'math': math, 'self': self} # Pass self directly into globals
        # Use exec_globals for locals as well
        dedented_events_str = textwrap.dedent(events_str)
        # For debugging, you can uncomment the next line to see the string being executed:
        # print("DEBUG: Dedented ODE events string:\n", dedented_events_str)
        exec(dedented_events_str, exec_globals, exec_globals)
        # Events are usually a list, so retrieve them from globals
        return exec_globals['ode_events']

    def _get_obs(self):
        return np.array([self.state[key] for key in self._state_keys], dtype=np.float32)

    def _get_info(self):
        return self.state.copy() # Return a copy of the current state dictionary

    def _evaluate_expression(self, expr_str, local_vars=None):
        if local_vars is None:
            local_vars = {}

        if expr_str.startswith('?'):
            expr_str = expr_str[1:].strip()

        eval_globals = {
            'np': np, 'math': math,
            **self.constants,
            **self.functions,
            **self.predicates
        }
        eval_locals = {
            **self.state,
            **local_vars
        }

        expr_py = expr_str.replace('^', '**').replace('abs(', 'np.abs(')
        # Logical operators are already handled by regex in the original code, but explicit replacement is safer
        expr_py = expr_py.replace('&', 'and').replace('|', 'or')
        # The following line is commented out because we now explicitly use np.isclose in the formula strings
        # expr_py = re.sub(r'(?<![<>=!])=(?![=])', '==', expr_py) # Replace standalone '=' with '=='

        try:
            return eval(expr_py, eval_globals, eval_locals)
        except Exception as e:
            # If evaluation fails here, it often means an invalid state or division by zero.
            # For predicates, returning False is a safe default for constraint checks.
            # For assignments, re-raise the error as it indicates a problem with the expression itself.
            if any(op in expr_str for op in ['<', '>', '==', '!=', 'isclose']): # Check if it's a comparison/predicate (added 'isclose')
                print(f"DEBUG: Predicate evaluation failed for '{expr_str}': {e}. Returning False.")
                return False
            print(f"ERROR: Expression evaluation failed for '{expr_str}': {e}")
            print(f"Eval globals: {eval_globals}")
            print(f"Eval locals: {eval_locals}")
            raise # Re-raise for other errors (e.g., in assignments)

    def _apply_control_action(self, action):
        reward = 0
        terminated = False
        choice_idx = action['choice_ChoiceLevel_0']
        chosen_branch_config = self._control_func_action_config.choices[choice_idx]

        for sub_action_config in chosen_branch_config.actions:
            if isinstance(sub_action_config, ActionConstraintConfig):
                constraint_met = self._evaluate_expression(sub_action_config.formula)
                if not constraint_met:
                    print(f"DEBUG: Control constraint violated: '{sub_action_config.formula}' at state {self.state}")
                    reward -= 100.0 
                    terminated = True
                    return reward, terminated, False 
            elif isinstance(sub_action_config, DeterministicAssignmentActionConfig):
                assigned_value = self._evaluate_expression(sub_action_config.value)
                self.state[sub_action_config.variable] = assigned_value
            elif isinstance(sub_action_config, NondeterministicAssignmentActionConfig):
                assign_key = f"assign_{sub_action_config.variable}"
                if assign_key in action:
                    self.state[sub_action_config.variable] = action[assign_key]
                else:
                    print(f"DEBUG: Missing nondeterministic assignment for '{sub_action_config.variable}' in action. State: {self.state}")
                    reward -= 50.0 
                    terminated = True
                    return reward, terminated, False
            elif isinstance(sub_action_config, SequenceActionConfig):
                for inner_action in sub_action_config.actions:
                    if isinstance(inner_action, ActionConstraintConfig):
                        constraint_met = self._evaluate_expression(inner_action.formula)
                        if not constraint_met:
                            print(f"DEBUG: Inner control constraint violated: '{inner_action.formula}' at state {self.state}")
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
                            print(f"DEBUG: Missing inner nondeterministic assignment for '{inner_action.variable}' in action. State: {self.state}")
                            reward -= 50.0
                            terminated = True
                            return reward, terminated, False

        self.state['t'] = 0.0 # Reset time to 0 after discrete action
        return reward, terminated, True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Initializes self.np_random

        self.current_step = 0

        # --- Set reasonable values for constants if they are still 0.0 ---
        # Ensure constants are positive and valid
        # Using .get() and explicit type/value checks for robustness against external modification
        # These are now initialized to non-zero defaults in __init__
        if not (isinstance(self.constants.get('ep'), (int, float)) and self.constants['ep'] > 0.0):
            self.constants['ep'] = 0.1 
        if not (isinstance(self.constants.get('b'), (int, float)) and self.constants['b'] > 0.0):
            self.constants['b'] = 1.0 
        if not (isinstance(self.constants.get('A'), (int, float)) and self.constants['A'] >= 0.0):
            self.constants['A'] = 0.5 
        if not (isinstance(self.constants.get('W'), (int, float)) and self.constants['W'] > 0.0):
            self.constants['W'] = 0.5 
        
        print(f"DEBUG: Reset called. self.constants['ep'] after checks = {self.constants['ep']}")

        # Update agent's W_val if agent is initialized and attached
        if hasattr(self, 'agent') and self.agent is not None:
            self.agent.W_val = self.constants['W']
            print(f"DEBUG: Agent's W_val updated to {self.agent.W_val} in env.reset().")


        # --- Initial State (randomized for better generalization) ---
        # Randomize x, y slightly around 0
        self.state['x'] = self.np_random.uniform(-0.5, 0.5)
        self.state['y'] = self.np_random.uniform(-0.5, 0.5)
        
        # Randomize initial velocity (v > 0)
        self.state['v'] = self.np_random.uniform(0.5, 2.0) # Ensure v > 0
        
        # Randomize initial direction (dx, dy) - unit vector
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.state['dx'] = np.cos(angle)
        self.state['dy'] = np.sin(angle)
        
        self.state['a'] = 0.0 # Initial acceleration often 0
        self.state['w'] = 0.0 # Initial angular velocity often 0
        self.state['r'] = 1.0 # Initial radius of curvature (arbitrary, will be set by agent or derived)
        
        # Randomize obstacle position (xo, yo) within a reasonable range
        # Let's use a range that makes sense for the environment's scale.
        # Ensure it's not too close initially, but also not too far.
        self.state['xo'] = self.np_random.uniform(3.0, 7.0)
        self.state['yo'] = self.np_random.uniform(-2.0, 2.0)
        
        self.state['t'] = 0.0

        # Ensure dx, dy represent a unit vector (robustness)
        norm = np.sqrt(self.state['dx']**2 + self.state['dy']**2)
        if norm > 1e-6: # Avoid division by zero
            self.state['dx'] /= norm
            self.state['dy'] /= norm
        else: # Default to (1,0) if initial dx, dy are too small
            self.state['dx'] = 1.0
            self.state['dy'] = 0.0

        observation = self._get_obs()
        info = self._get_info()
        print(f"DEBUG: Reset completed. self.state = {self.state}")
        return observation, info

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        print(f"DEBUG: Step called. State at start of step(): {self.state}")
        print(f"DEBUG: Step called. self.constants['ep'] at start of step(): {self.constants['ep']}")

        control_reward, control_terminated, control_applied = self._apply_control_action(action)
        reward += control_reward
        terminated = control_terminated
        print(f"DEBUG: State after _apply_control_action(): {self.state}")

        if not control_applied:
            print(f"DEBUG: Control action not fully applied. Terminated: {terminated}, Reward: {reward}")
            return self._get_obs(), reward, terminated, truncated, info

        y0 = np.array([self.state[var_name] for var_name in self._ode_state_map.keys()], dtype=np.float64)
        ep_val = self.constants['ep']

        print(f"DEBUG: In step(), ep_val for ODE integration is: {ep_val}")
        
        # Ensure ep_val is strictly positive for ODE integration
        # If ep_val is <= 0, we skip ODE integration as per KeYmaera X semantics for time horizon.
        if ep_val <= 0: 
            self.state['t'] = ep_val # Update t to the (possibly invalid or zero) ep_val
            print(f"DEBUG: ep_val ({ep_val}) <= 0. Skipping solve_ivp.")
            # No ODE integration, proceed directly to postcondition check.
            # The state remains as set by the discrete control action.
        else: # ep_val is strictly positive, proceed with ODE integration
            t_span = (0, ep_val) 
            print(f"DEBUG: In step(), t_span for solve_ivp: {t_span}") 
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
                    final_y = solution.y[:, -1]
                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]
                elif solution.status == 1: # An event occurred
                    if solution.y_events and solution.y_events[0].size > 0:
                        # Take the state right at the event
                        final_y = solution.y_events[0][0] 
                    else:
                        # Fallback to last step if event data is empty (shouldn't happen with terminal events)
                        final_y = solution.y[:, -1] 

                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]

                    # Check which event terminated it
                    if self.state['v'] < 0:
                        print(f"DEBUG: ODE event: v became negative. State: {self.state}")
                        reward -= 100.0 
                        terminated = True
                    # Check if t reached ep (this is not a penalty, just an event)
                    if self.state['t'] >= ep_val - 1e-9 and self.state['t'] < ep_val + 1e-9:
                        # print(f"DEBUG: ODE event: Time limit (ep) reached. State: {self.state}")
                        pass
                else: # Integration failed (status 2)
                    print(f"DEBUG: ODE integration failed: {solution.message}. State: {self.state}")
                    reward -= 150.0 
                    terminated = True

            except Exception as e:
                print(f"DEBUG: Error during ODE integration: {e}. State: {self.state}")
                reward -= 200.0 
                terminated = True

        # --- 3. Check Problem Postcondition ---
        try:
            postcondition_satisfied = self._evaluate_expression('(x - xo)**2 + (y - yo)**2 > 0') # Modified to '**'
            if not postcondition_satisfied:
                print(f"DEBUG: Postcondition violated: (x - xo)**2 + (y - yo)**2 <= 0. State: {self.state}")
                reward -= 100.0 
                terminated = True
            else:
                # SIGNIFICANT CHANGE: Increased survival reward to make it more impactful
                # Also add a reward for distance to obstacle to encourage moving away
                dist_sq = (self.state['x'] - self.state['xo'])**2 + (self.state['y'] - self.state['yo'])**2
                # Reward for safety: 1.0 for surviving, plus a small bonus for being further away
                reward += 1.0 + np.sqrt(dist_sq) * 0.01 # Small bonus for distance
        except Exception as e:
            print(f"DEBUG: Error evaluating postcondition: {e}. State: {self.state}")
            reward -= 100.0 # Penalty for error in postcondition evaluation
            terminated = True

        # --- 4. Check Truncation ---
        if self.current_step >= self.max_episode_steps:
            truncated = True
            print(f"DEBUG: Episode truncated due to max steps. State: {self.state}")
            reward -= 10.0 # Small penalty for truncating without reaching goal

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Render: Step={self.current_step}, State={self.state}")

    def close(self):
        pass