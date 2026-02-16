import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import math # For math functions like sqrt, sin, cos, etc.
import textwrap
import re # For regex replacements in _evaluate_expression
from typing import (Dict, Any, Tuple) # Import typing hints
from pyast_config import (ActionType, ChoiceActionConfig, SequenceActionConfig, DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig, ActionConstraintConfig, CallHybridProgramConfig, HybridProgramStepConfig) # Import AST classes

class IJRR17_Corollary_1_2_Static_safety_with_margin_for_imperfect_trajectories(gym.Env):
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
        self.constants['b'] = 1.0
        self.constants['A'] = 1.0
        self.constants['W'] = 1.0
        self.constants['Ud'] = 0.1

        # Functions (lambdas for easy evaluation)
        self.functions['stopDist'] = lambda v: ((1+self.constants['Ud']) * v**2 / (2*self.constants['b']))
        self.functions['accelComp'] = lambda v: ((1+self.constants['Ud']) * (self.constants['A']/self.constants['b'] + 1) * (self.constants['A']/2 * self.constants['ep']**2 + self.constants['ep']*v))
        self.functions['admissibleSeparation'] = lambda v: self.functions['stopDist'](v) + self.functions['accelComp'](v)

        # Predicates (lambdas for easy evaluation)
        self.predicates['isWellformedDir'] = lambda dx, dy: 1-self.constants['Ud'] <= dx**2+dy**2 and dx**2 + dy**2 <= 1+self.constants['Ud']
        self.predicates['bounds'] = lambda : self.constants['A'] >= 0                                                                                                                     and self.constants['b'] > 0                                                                                                                      and self.constants['ep'] > 0                                                                                                                     and 0<=self.constants['Ud'] and self.constants['Ud'] <= 1
        self.predicates['initialState'] = lambda x, y, v, dx, dy, xo, yo: v == 0         and (x-xo)**2 + (y-yo)**2 > 0         and self.predicates['isWellformedDir'](dx,dy)
        self.predicates['assumptions'] = lambda x, y, v, dx, dy, xo, yo: self.predicates['bounds']() and self.predicates['initialState'](x, y, v, dx, dy, xo, yo)
        self.predicates['loopinv'] = lambda x, y, v, dx, dy, xo, yo: v >= 0         and self.predicates['isWellformedDir'](dx,dy)         and (np.abs(x-xo) > self.functions['stopDist'](v) or np.abs(y-yo) > self.functions['stopDist'](v))

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
                    ActionConstraintConfig(formula='v = 0'),
                    DeterministicAssignmentActionConfig(variable='a', value='0'),
                    DeterministicAssignmentActionConfig(variable='w', value='0')
                ]),
                SequenceActionConfig(actions=[
                    DeterministicAssignmentActionConfig(variable='a', value='A'),
                    NondeterministicAssignmentActionConfig(variable='w'),
                    ActionConstraintConfig(formula='-W<=w & w<=W'),
                    NondeterministicAssignmentActionConfig(variable='r'),
                    NondeterministicAssignmentActionConfig(variable='xo'),
                    NondeterministicAssignmentActionConfig(variable='yo'),
                    ActionConstraintConfig(formula='r!=0 & r*w = v'),
                    ActionConstraintConfig(formula='np.abs(x-xo) > admissibleSeparation(v)         | np.abs(y-yo) > admissibleSeparation(v)')
                ])
            ]) # Stored for step logic

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
            # FIX: Corrected regex backreferences from ,  to , 
            new_expr_py = re.sub(r'\((.*?)\s*->\s*(.*?)\)', r'((not ) or )', expr_py)
            if new_expr_py == expr_py:
                break
            expr_py = new_expr_py

        try:
            return eval(expr_py, eval_globals, eval_locals)
        except Exception as e:
            print(f"Error evaluating expression '{expr_str}': {e}")
            # print(f"Eval globals: {eval_globals}") # Too verbose
            # print(f"Eval locals: {eval_locals}") # Too verbose
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

        # Z3 could not find a satisfying initial state or encountered an error.
        # Falling back to heuristic defaults and adjustments.
        # Apply explicit assignments from the precondition (fallback)
        self.state['v'] = 0

        # Ensure dx, dy satisfy isWellformedDir (approximate unit vector) (fallback)
        norm_sq = self.state['dx']**2 + self.state['dy']**2
        ud_val = self.constants.get('Ud', 0.0) # Get Ud from constants, default to 0.0 if not found
        min_norm_sq_target = (1.0 - ud_val)**2
        max_norm_sq_target = (1.0 + ud_val)**2
        if not (min_norm_sq_target <= norm_sq <= max_norm_sq_target):
            norm = np.sqrt(norm_sq)
            if norm > 1e-6: # Avoid division by zero
                self.state['dx'] /= norm
                self.state['dy'] /= norm
            else: # Default to (1,0) if too small
                self.state['dx'] = 1.0
                self.state['dy'] = 0.0

        # Ensure robot and obstacle are not at the same position (fallback)
        if self.state['x'] == self.state['xo'] and self.state['y'] == self.state['yo']:
            self.state['x'] += 0.1 # Slightly perturb x to ensure separation

        # Verify the full precondition after setting initial states
        precondition_satisfied = self._evaluate_expression('((A >= 0                                                                                                                     & b > 0                                                                                                                      & ep > 0                                                                                                                     & 0<=Ud & Ud <= 1) & (v = 0         & (x-xo)**2 + (y-yo)**2 > 0         & (1-Ud <= dx**2+dy**2 & dx**2 + dy**2 <= 1+Ud)))', is_constraint=True)
        if not precondition_satisfied:
            print(f"Warning: Initial state does not satisfy precondition: 'assumptions(x, y, v, dx, dy, xo, yo)'. Expanded: '((A >= 0                                                                                                                     & b > 0                                                                                                                      & ep > 0                                                                                                                     & 0<=Ud & Ud <= 1) & (v = 0         & (x-xo)**2 + (y-yo)**2 > 0         & (1-Ud <= dx**2+dy**2 & dx**2 + dy**2 <= 1+Ud)))'.")
            # If Z3 failed or heuristics didn't fully satisfy, this warning is crucial.


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
                    if self.state.get('v', 0.0) < 0: # Use .get with default for safety
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
        postcondition_satisfied = self._evaluate_expression('(x - xo)**2 + (y - yo)**2 > 0', is_constraint=True)
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
