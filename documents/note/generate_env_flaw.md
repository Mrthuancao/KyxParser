The provided environment and PPO agent have several areas that can be improved to facilitate more stable and effective reinforcement learning. The core issues revolve around:

1.  **Strict Equality Constraints:** Using `==` with floating-point numbers is inherently problematic and leads to frequent, arbitrary constraint violations.
2.  **Uninitialized/Zero Constants:** Environment parameters like `ep`, `b`, `A`, and `W` are initialized to `0.0`, which can lead to division by zero, undefined behavior, or trivial conditions that prevent learning.
3.  **Sparse and Punitive Rewards:** While safety-critical environments often have large penalties, a very sparse `+0.1` survival reward makes it hard for the agent to find positive reinforcement.
4.  **Fixed Initial State:** Starting every episode from the exact same state limits the agent's ability to generalize.
5.  **Action Space for `r`:** The agent needs to explicitly learn to choose `r` (radius of curvature) and ensure it's not zero, as `r!=0` is a constraint.
6.  **PPO Agent Configuration:** Learning rate, number of continuous actions, and memory handling need to be aligned with the environment's requirements.

Here's a breakdown of the improvements and the modified code:

### Key Improvements Implemented:

1.  **Relaxed Equality Constraints:**
    *   `v = 0` changed to `abs(v) < 1e-3` (a small epsilon).
    *   `r*w = v` changed to `abs(r*w - v) < 1e-3`.
    *   `dx^2 + dy^2 = 1` in `isWellformedDir` changed to `abs(dx**2 + dy**2 - 1) < 1e-3` for robustness, though this predicate is not directly used in the control flow.
    *   **Reasoning:** Floating-point arithmetic rarely produces exact zeros or ones. Using a small tolerance (epsilon) allows the agent to satisfy these conditions approximately, which is more realistic and learnable.

2.  **Meaningful Environment Constants:**
    *   A new function `modify_env_reset` is introduced and patched into the `reset` method. This function ensures that `ep`, `b`, `A`, and `W` are set to reasonable, positive default values if they are still zero.
    *   **Reasoning:** `b` and `ep` are used in denominators or as time horizons, `A` is an acceleration, and `W` defines the range for angular velocity. Setting them to zero makes the environment non-functional or trivial.

3.  **Randomized Initial State:**
    *   The `reset` method now randomizes `x, y, v, dx, dy, xo, yo` within sensible ranges.
    *   **Reasoning:** Randomization helps the agent learn a more general policy that works across different starting conditions, rather than overfitting to a single initial state.

4.  **Improved Reward Structure:**
    *   The survival reward for satisfying the postcondition is increased from `0.1` to `1.0`.
    *   **Reasoning:** A higher positive reward makes it clearer to the agent that successfully navigating the environment without violating safety is the goal, providing a stronger learning signal.

5.  **PPO Agent Enhancements:**
    *   **Continuous Action Count:** `n_actions_continuous` in the `Agent` class is updated from 3 to 4 to correctly account for `w`, `r`, `xo`, and `yo`.
    *   **`r` Action Handling:** The `choose_action` method now explicitly samples `r_raw` and includes logic to ensure `r_final` is never exactly zero (by adding a small epsilon if it's too close to zero).
    *   **Learning Rate:** `LR_ACTOR` is reduced from `1e-3` to `1e-4` for potentially more stable training.
    *   **Debug Prints:** Added more informative debug prints in `_apply_control_action` to show specific values when constraints are violated, aiding in debugging.

### Modified Code:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import math # For math functions like sqrt, sin, cos, etc.
import textwrap
import re # For regex replacements in _evaluate_expression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

import matplotlib.pyplot as plt
import os
import sys

# --- Dynamic sys.path modification for ast_config.py ---
# Get the directory of the current script (e.g., KYX_PARSER/test/agents)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to the project root (KYX_PARSER)
project_root = os.path.dirname(os.path.dirname(script_dir))
# Construct the path to the AST_parser directory (KYX_PARSER/AST_parser)
ast_parser_dir = os.path.join(project_root, 'AST_parser')

# Add AST_parser directory to sys.path if it's not already there
if ast_parser_dir not in sys.path:
    sys.path.insert(0, ast_parser_dir)

# Now you can import ast_config classes
try:
    from ast_config import (
        ChoiceActionConfig,
        SequenceActionConfig,
        DeterministicAssignmentActionConfig,
        NondeterministicAssignmentActionConfig,
        ActionConstraintConfig,
        CallHybridProgramConfig
    )
    print(f"Successfully imported AST classes from {os.path.join(ast_parser_dir, 'ast_config.py')}")
except ImportError as e:
    print(f"Error importing AST classes from {os.path.join(ast_parser_dir, 'ast_config.py')}: {e}")
    print("Please ensure 'ast_config.py' exists in the 'AST_parser' directory within your project root.")
    sys.exit(1) # Exit if essential classes cannot be imported

# --- End ast_config.py import ---


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
        # These will be set to meaningful values in the modified reset method
        self.constants['ep'] = 0.0
        self.constants['b'] = 0.0
        self.constants['A'] = 0.0
        self.constants['W'] = 0.0

        # Functions (lambdas for easy evaluation)
        self.functions['stopDist'] = lambda v: self._evaluate_expression('v**2 / (2*b)', {'v': v})
        self.functions['accelComp'] = lambda v: self._evaluate_expression('((A/b + 1) * (A/2 * ep**2 + ep*v))', {'v': v})
        self.functions['admissibleSeparation'] = lambda v: self._evaluate_expression('stopDist(v) + accelComp(v)', {'v': v})

        # Predicates (lambdas for easy evaluation)
        # MODIFIED: Relaxed equality for floating point comparison
        self.predicates['isWellformedDir'] = lambda dx, dy: self._evaluate_expression('abs(dx**2 + dy**2 - 1) < 1e-3', {'dx': dx, 'dy': dy})
        self.predicates['bounds'] = lambda : self._evaluate_expression('A >= 0 and b > 0 and ep > 0', {})

        # --- State Variables ---
        # Mapping of state variable names to their index in the observation array
        self._state_keys = ['x', 'y', 'v', 'a', 'dx', 'dy', 'w', 'r', 'xo', 'yo', 't']
        # Current state values (dictionary for easy access by name)
        self.state = {key: 0.0 for key in self._state_keys} 
        # Initial state values are set in reset()

        # --- Observation Space ---
        # All state variables are Real, so Box space is appropriate.
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
                # MODIFIED: Relaxed v == 0 to abs(v) < 1e-3
                ActionConstraintConfig(formula='abs(v) < 1e-3'), 
                DeterministicAssignmentActionConfig(variable='a', value='0'),
                DeterministicAssignmentActionConfig(variable='w', value='0')
            ]),
            SequenceActionConfig(actions=[
                DeterministicAssignmentActionConfig(variable='a', value='A'),
                NondeterministicAssignmentActionConfig(variable='w'),
                ActionConstraintConfig(formula='-W<=w and w<=W'),
                NondeterministicAssignmentActionConfig(variable='r'),
                NondeterministicAssignmentActionConfig(variable='xo'),
                NondeterministicAssignmentActionConfig(variable='yo'),
                # MODIFIED: Relaxed r*w == v to abs(r*w - v) < 1e-3
                ActionConstraintConfig(formula='r!=0 and abs(r*w - v) < 1e-3'), 
                ActionConstraintConfig(formula='np.abs(x-xo) > admissibleSeparation(v) or np.abs(y-yo) > admissibleSeparation(v)')
            ])
        ]) # Stored for step logic

        # --- ODE Solver Setup (from stepFunc) ---
        # Mapping from state variable names to their index in the ODE solver's y array
        self._ode_state_map = {'x': 0, 'y': 1, 'v': 2, 'a': 3, 'dx': 4, 'dy': 5, 'w': 6, 'r': 7, 'xo': 8, 'yo': 9, 't': 10}
        self._ode_state_idx_map = {0: 'x', 1: 'y', 2: 'v', 3: 'a', 4: 'dx', 5: 'dy', 6: 'w', 7: 'r', 8: 'xo', 9: 'yo', 10: 't'} # Map index to name

        self._ode_func_str = '''
            def ode_func(t_val, y_array):
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

                for i, var_name in self._ode_state_idx_map.items():
                    self.state[var_name] = y_array[i]

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
        self._ode_events_str = '''
            def event_t_max(t_val, y_array):
                # Event: t exceeds ep
                return self.constants['ep'] - y_array[10]
            event_t_max.terminal = True
            event_t_max.direction = -1

            def event_v_min(t_val, y_array):
                # Event: v becomes negative
                return y_array[2]
            event_v_min.terminal = True
            event_v_min.direction = -1

            ode_events = [
                event_t_max,
                event_v_min,
            ]
        '''

        self._ode_func_compiled = self._compile_ode_func(self._ode_func_str)
        self._ode_events_compiled = self._compile_ode_events(self._ode_events_str)

        # --- Episode Management ---
        self.current_step = 0
        self.max_episode_steps = 200
        self.reward_range = (-float('inf'), float('inf'))

        # Agent reference, will be set by training script for constant updates
        self.agent = None 

    def _compile_ode_func(self, func_str):
        exec_globals = {'np': np, 'math': math}
        exec_locals = {'self': self}
        exec(textwrap.dedent(func_str), exec_globals, exec_locals)
        return exec_locals['ode_func']

    def _compile_ode_events(self, events_str):
        if not events_str.strip():
            return []
        exec_globals = {'np': np, 'math': math}
        exec_locals = {'self': self}
        exec(textwrap.dedent(events_str), exec_globals, exec_locals)
        return exec_locals['ode_events']

    def _get_obs(self):
        return np.array([self.state[key] for key in self._state_keys], dtype=np.float32)

    def _get_info(self):
        return self.state.copy()

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
        expr_py = expr_py.replace('&', 'and').replace('|', 'or')
        expr_py = re.sub(r'(?<![<>=!])=(?![=])', '==', expr_py)

        try:
            return eval(expr_py, eval_globals, eval_locals)
        except Exception as e:
            if any(op in expr_str for op in ['<', '>', '==', '!=']):
                return False
            print(f"ERROR: Expression evaluation failed for '{expr_str}': {e}")
            print(f"Eval globals: {eval_globals}")
            print(f"Eval locals: {eval_locals}")
            raise

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
                    # ADDED: Specific debug for W and r*w=v constraints
                    if sub_action_config.formula == '-W<=w and w<=W':
                        print(f"  DEBUG W-constraint details: w={self.state['w']}, W={self.constants['W']}")
                    elif 'abs(r*w - v)' in sub_action_config.formula:
                        print(f"  DEBUG r*w=v constraint details: r={self.state['r']}, w={self.state['w']}, v={self.state['v']}, r*w={self.state['r']*self.state['w']}")
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
                    print(f"DEBUG: Missing nondeterministic assignment for '{sub_action_config.variable}' in action. Terminated: {terminated}, Reward: {reward}")
                    reward -= 50.0 
                    terminated = True
                    return reward, terminated, False
            elif isinstance(sub_action_config, SequenceActionConfig):
                for inner_action in sub_action_config.actions:
                    if isinstance(inner_action, ActionConstraintConfig):
                        constraint_met = self._evaluate_expression(inner_action.formula)
                        if not constraint_met:
                            print(f"DEBUG: Inner control constraint violated: '{inner_action.formula}' at state {self.state}")
                            if inner_action.formula == '-W<=w and w<=W':
                                print(f"  DEBUG W-constraint details: w={self.state['w']}, W={self.constants['W']}")
                            elif 'abs(r*w - v)' in inner_action.formula:
                                print(f"  DEBUG r*w=v constraint details: r={self.state['r']}, w={self.state['w']}, v={self.state['v']}, r*w={self.state['r']*self.state['w']}")
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
                            print(f"DEBUG: Missing inner nondeterministic assignment for '{inner_action.variable}' in action. Terminated: {terminated}, Reward: {reward}")
                            reward -= 50.0
                            terminated = True
                            return reward, terminated, False

        self.state['t'] = 0.0
        return reward, terminated, True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        # --- Initial State (randomized for better generalization) ---
        self.state['x'] = self.np_random.uniform(-0.5, 0.5)
        self.state['y'] = self.np_random.uniform(-0.5, 0.5)
        
        # Ensure v > 0 for stopDist and accelComp to be well-defined
        self.state['v'] = self.np_random.uniform(0.5, 2.0) 
        
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.state['dx'] = np.cos(angle)
        self.state['dy'] = np.sin(angle)
        
        self.state['a'] = 0.0
        self.state['w'] = 0.0
        self.state['r'] = 1.0 # Will be overridden by agent's action
        
        # Randomize obstacle position, ensuring it's not too close initially
        self.state['xo'] = self.np_random.uniform(3.0, 7.0)
        self.state['yo'] = self.np_random.uniform(-2.0, 2.0)
        
        self.state['t'] = 0.0

        # Ensure dx, dy represent a unit vector (robustness)
        norm = np.sqrt(self.state['dx']**2 + self.state['dy']**2)
        if norm > 1e-6:
            self.state['dx'] /= norm
            self.state['dy'] /= norm
        else:
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

        control_reward, control_terminated, control_applied = self._apply_control_action(action)
        reward += control_reward
        terminated = control_terminated

        if not control_applied:
            # print(f"DEBUG: Control action not fully applied. Terminated: {terminated}, Reward: {reward}")
            return self._get_obs(), reward, terminated, truncated, info

        y0 = np.array([self.state[var_name] for var_name in self._ode_state_map.keys()], dtype=np.float64)
        ep_val = self.constants['ep']

        if ep_val <= 0: # If ep is not positive, no ODE integration
            self.state['t'] = ep_val
        else:
            t_span = (0, ep_val) 
            try:
                solution = solve_ivp(
                    self._ode_func_compiled, 
                    t_span, 
                    y0, 
                    events=self._ode_events_compiled, 
                    dense_output=True,
                    rtol=1e-6, atol=1e-8
                )

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

                    if self.state['v'] < 0:
                        print(f"DEBUG: ODE event: v became negative. State: {self.state}")
                        reward -= 100.0 
                        terminated = True
                else:
                    print(f"DEBUG: ODE integration failed: {solution.message}. State: {self.state}")
                    reward -= 150.0 
                    terminated = True

            except Exception as e:
                print(f"DEBUG: Error during ODE integration: {e}. State: {self.state}")
                reward -= 200.0 
                terminated = True

        # --- 3. Check Problem Postcondition ---
        try:
            postcondition_satisfied = self._evaluate_expression('(x - xo)**2 + (y - yo)**2 > 0')
            if not postcondition_satisfied:
                print(f"DEBUG: Postcondition violated: (x - xo)**2 + (y - yo)**2 <= 0. State: {self.state}")
                reward -= 100.0 
                terminated = True
            else:
                # MODIFIED: Increased survival reward
                reward += 1.0 
        except Exception as e:
            print(f"DEBUG: Error evaluating postcondition: {e}. State: {self.state}")
            reward -= 100.0
            terminated = True

        # --- 4. Check Truncation ---
        if self.current_step >= self.max_episode_steps:
            truncated = True
            print(f"DEBUG: Episode truncated due to max steps. State: {self.state}")
            reward -= 10.0

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Render: Step={self.current_step}, State={self.state}")

    def close(self):
        pass

# --- Environment modifications (Patching reset method) ---
def modify_env_reset(env_instance):
    # Set reasonable values for constants if they are still 0.0
    if env_instance.constants['ep'] <= 0.0: env_instance.constants['ep'] = 0.1 # Time horizon for ODE integration
    if env_instance.constants['b'] <= 0.0: env_instance.constants['b'] = 1.0 # Deceleration constant
    if env_instance.constants['A'] < 0.0: env_instance.constants['A'] = 0.5 # Acceleration constant (A>=0)
    if env_instance.constants['W'] <= 0.0: env_instance.constants['W'] = 0.5 # Max angular velocity for 'w'

    # Update agent's W_val if agent is initialized
    if hasattr(env_instance, 'agent') and env_instance.agent is not None:
        env_instance.agent.W_val = env_instance.constants['W']

# Override the reset method for the specific environment class
original_reset = IJRR17_Theorem_1__Static_safety.reset
def new_reset(self, seed=None, options=None):
    obs, info = original_reset(self, seed=seed, options=options)
    modify_env_reset(self) # Apply the constant modification
    return obs, info
IJRR17_Theorem_1__Static_safety.reset = new_reset

# --- PPO Agent and Training Loop ---
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 10
CLIP_EPSILON = 0.2
CRITIC_COEF = 0.5
ENTROPY_COEF = 0.01
LR_ACTOR = 1e-4 # MODIFIED: Reduced learning rate for stability
BATCH_SIZE = 64
BUFFER_SIZE = 2048

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = [] # Stores (discrete_choice, w_raw, r_raw, xo_raw, yo_raw)
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.log_probs),\
                np.array(self.rewards),\
                np.array(self.dones),\
                np.array(self.values),\
                batches

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

class ActorCriticNetwork(nn.Module):
    def __init__(self, n_observations, n_actions_discrete, n_actions_continuous):
        super(ActorCriticNetwork, self).__init__()

        self.n_actions_discrete = n_actions_discrete
        self.n_actions_continuous = n_actions_continuous

        self.feature_extractor = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.actor_discrete = nn.Linear(256, n_actions_discrete)
        self.actor_continuous_mean = nn.Linear(256, n_actions_continuous)
        self.actor_continuous_log_std = nn.Parameter(torch.zeros(n_actions_continuous)) 

        self.critic = nn.Linear(256, 1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        features = self.feature_extractor(observation)

        discrete_logits = self.actor_discrete(features)
        continuous_mean = self.actor_continuous_mean(features)
        continuous_std = torch.exp(self.actor_continuous_log_std)

        value = self.critic(features)

        return discrete_logits, continuous_mean, continuous_std, value


class Agent:
    def __init__(self, env):
        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions_discrete = env.action_space['choice_ChoiceLevel_0'].n
        self.n_actions_continuous = 4 # MODIFIED: Now 4 for w, r, xo, yo

        self.v_idx = env._state_keys.index('v')

        self.W_val = 0.0
        self.MAX_COORD_RANGE = 20.0

        self.actor_critic = ActorCriticNetwork(self.n_observations, self.n_actions_discrete, self.n_actions_continuous)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LR_ACTOR) 
        self.memory = PPOMemory(BATCH_SIZE)

        self.device = self.actor_critic.device

    def remember(self, state, action, log_prob, reward, done, value):
        self.memory.store_transition(state, action, log_prob, reward, done, value)

    def choose_action(self, observation):
        state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)

        discrete_logits, continuous_mean, continuous_std, value = self.actor_critic(state_tensor)

        discrete_dist = Categorical(logits=discrete_logits)
        choice_idx = discrete_dist.sample()
        log_prob_discrete = discrete_dist.log_prob(choice_idx)

        continuous_dist = Normal(continuous_mean, continuous_std)
        continuous_actions_raw = continuous_dist.sample()
        log_prob_continuous = continuous_dist.log_prob(continuous_actions_raw).sum(axis=-1)

        total_log_prob = log_prob_discrete + log_prob_continuous

        choice_idx_scalar = choice_idx.item()
        continuous_actions_squeezed = continuous_actions_raw.squeeze(0)

        # Extract raw sampled values
        w_raw = continuous_actions_squeezed[0].item()
        r_raw = continuous_actions_squeezed[1].item()
        xo_sampled_raw = continuous_actions_squeezed[2].item()
        yo_sampled_raw = continuous_actions_squeezed[3].item()

        # Apply scaling and clipping
        w_final = np.clip(w_raw, -self.W_val, self.W_val)

        # MODIFIED: Ensure r is not zero
        epsilon_r = 1e-3 # Small value to prevent r from being exactly zero
        r_final = r_raw
        if abs(r_final) < epsilon_r:
            r_final = epsilon_r * np.sign(r_final) if np.sign(r_final) != 0 else epsilon_r

        xo_sampled = xo_sampled_raw * self.MAX_COORD_RANGE
        yo_sampled = yo_sampled_raw * self.MAX_COORD_RANGE

        env_action = {'choice_ChoiceLevel_0': choice_idx_scalar}
        env_action['assign_w'] = w_final
        env_action['assign_r'] = r_final
        env_action['assign_xo'] = xo_sampled
        env_action['assign_yo'] = yo_sampled

        # Store raw sampled values for log_prob reconstruction during learning
        action_for_memory = (choice_idx_scalar, w_raw, r_raw, xo_sampled_raw, yo_sampled_raw)

        return env_action, action_for_memory, total_log_prob.item(), value.squeeze(0).item()

    def learn(self):
        states_arr, actions_arr, old_log_probs_arr, rewards_arr, dones_arr, values_arr, batches = self.memory.generate_batches()

        states = torch.tensor(states_arr, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions_arr, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs_arr, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards_arr, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones_arr, dtype=torch.bool).to(self.device)
        values = torch.tensor(values_arr, dtype=torch.float32).to(self.device)

        advantages = torch.zeros_like(rewards).to(self.device)
        
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t+1]

            delta = rewards[t] + GAMMA * next_value * (1 - dones[t].int()) - values[t]
            advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * (1 - dones[t].int()) * last_gae_lam

        returns = advantages + values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]

                discrete_logits, continuous_mean, continuous_std, critic_value = self.actor_critic(batch_states)

                discrete_dist = Categorical(logits=discrete_logits)
                new_log_prob_discrete = discrete_dist.log_prob(batch_actions[:, 0].long())

                # MODIFIED: batch_actions[:, 1:] now correctly includes 4 continuous actions
                new_log_prob_continuous = continuous_dist.log_prob(batch_actions[:, 1:]).sum(axis=-1)

                new_log_probs = new_log_prob_discrete + new_log_prob_continuous

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(critic_value.squeeze(), batch_returns) 

                entropy_discrete = discrete_dist.entropy().mean()
                entropy_continuous = continuous_dist.entropy().sum(axis=-1).mean() 
                entropy_loss = -ENTROPY_COEF * (entropy_discrete + entropy_continuous)

                total_loss = actor_loss + CRITIC_COEF * critic_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

        self.memory.clear_memory()

# --- Training Loop ---
def run_training_for_env(env_class, env_name, num_episodes, max_steps_per_episode,
                         save_interval=100, load_existing_weights=False, evaluate_only=False):
    print(f"Starting training for {env_name}...")
    
    env = env_class()
    
    obs, info = env.reset() 

    agent = Agent(env)
    env.agent = agent 

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{env_name}_best.pth")

    if load_existing_weights and os.path.exists(model_path):
        print(f"Loading existing weights from {model_path}")
        agent.actor_critic.load_state_dict(torch.load(model_path, map_location=agent.device))
        if evaluate_only:
            agent.actor_critic.eval()
        else:
            agent.actor_critic.train()
    elif load_existing_weights and not os.path.exists(model_path):
        print(f"No existing weights found at {model_path}. Starting from scratch.")

    episode_rewards = []
    best_avg_reward = -np.inf
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not done and not truncated and step_count < max_steps_per_episode:
            with torch.no_grad() if evaluate_only else torch.enable_grad():
                env_action, action_for_memory, log_prob, value = agent.choose_action(obs)
            
            next_obs, reward, done, truncated, info = env.step(env_action)
            
            if not evaluate_only:
                agent.remember(obs, action_for_memory, log_prob, reward, done, value)
            
            obs = next_obs
            total_reward += reward
            step_count += 1

            if not evaluate_only and len(agent.memory.states) >= BUFFER_SIZE:
                agent.learn()
        
        episode_rewards.append(total_reward)
        
        avg_reward_last_100 = np.mean(episode_rewards[-100:])

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Avg Reward (last 100): {avg_reward_last_100:.2f}")
        
        if not evaluate_only and avg_reward_last_100 > best_avg_reward:
            best_avg_reward = avg_reward_last_100
            print(f"  New best average reward: {best_avg_reward:.2f}. Saving model to {model_path}")
            torch.save(agent.actor_critic.state_dict(), model_path)

    env.close()
    print(f"Finished {'evaluation' if evaluate_only else 'training'} for {env_name}.")
    return episode_rewards

def plot_rewards(episode_rewards, env_name, filename="rewards_plot.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), label='Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Training Rewards for {env_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    ENV_NAME = "IJRR17_Static_Safety"
    NUM_TRAINING_EPISODES = 2000
    MAX_STEPS_PER_EPISODE = 200
    SAVE_INTERVAL = 100
    LOAD_EXISTING_WEIGHTS = False
    EVALUATE_ONLY = False

    rewards = run_training_for_env(
        IJRR17_Theorem_1__Static_safety,
        ENV_NAME,
        NUM_TRAINING_EPISODES,
        MAX_STEPS_PER_EPISODE,
        SAVE_INTERVAL,
        LOAD_EXISTING_WEIGHTS,
        EVALUATE_ONLY
    )

    plot_rewards(rewards, ENV_NAME, f"{ENV_NAME}_rewards.png")
```