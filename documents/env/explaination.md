This Gymnasium environment, `KeYmaeraX_RL_Environment`, is designed to simulate a hybrid system specified in KeYmaera X, a deductive verification tool for hybrid systems. It translates the formal mathematical definitions of a hybrid system into a Python environment suitable for reinforcement learning (RL) agents.

### 1. Overall Purpose

The primary goal of this environment is to provide an interface for an RL agent to learn control policies for a hybrid system while respecting its safety properties. The environment models a system with both discrete control actions (decisions made by the agent) and continuous dynamics (governed by differential equations). The agent's objective is typically to maximize a reward signal, which is structured to encourage safe behavior and penalize violations of system constraints or safety properties.

In this specific environment, the system likely models a robot (or "self-driving car") navigating in an environment with an obstacle (or "other car"). The safety property being verified (and enforced in the RL environment) is related to maintaining a safe distance from the obstacle.

### 2. KeYmaera X Context

KeYmaera X is a theorem prover for hybrid systems. Hybrid systems combine continuous dynamics (like physics, described by differential equations) with discrete changes (like control decisions, mode switches). A KeYmaera X specification defines:

*   **Constants**: Fixed numerical values.
*   **Functions**: Mathematical expressions.
*   **Predicates**: Boolean expressions that define properties (like safety, well-formedness, invariants).
*   **Hybrid Programs**: Sequences of discrete assignments, non-deterministic choices, and continuous evolutions (differential equations).

This Python environment effectively "executes" a KeYmaera X hybrid program by:
*   Representing the system state as a dictionary of variables.
*   Translating KeYmaera X constants, functions, and predicates into Python equivalents.
*   Modeling the discrete control part (`controlFunc`) as the Gymnasium `action_space`.
*   Modeling the continuous evolution part (`stepFunc`) using an ODE solver.
*   Using KeYmaera X predicates as runtime checks for rewards, termination, and constraints.

### 3. Environment Structure (`KeYmaeraX_RL_Environment`)

Let's break down the key components of the `KeYmaeraX_RL_Environment` class:

#### 3.1. `__init__(self, render_mode=None)`: Initialization

This method sets up all the static definitions of the hybrid system.

*   **`self.constants`**: Stores numerical constants defined in KeYmaera X.
    ```python
    self.constants['ep'] = 0.0 # Time duration for continuous evolution
    self.constants['W'] = 0.0 # Max angular velocity
    self.constants['V'] = 0.0 # Max velocity of obstacle
    ```
    *Note: These are initialized to 0.0. In a real scenario, you'd set these after environment instantiation, e.g., `env.constants['ep'] = 0.1`.*

*   **`self.functions`**: Translates KeYmaera X functions into Python `lambda` functions. These lambdas use `_evaluate_expression` to interpret their bodies.
    ```python
    self.functions['stopDist'] = lambda v: self._evaluate_expression('0', {'v': v})
    self.functions['accelComp'] = lambda v: self._evaluate_expression('ep*(v+V)', {'v': v})
    self.functions['admissibleSeparation'] = lambda v: self._evaluate_expression('stopDist(v) + accelComp(v)', {'v': v})
    ```
    *Interpretation*: `stopDist` is defined as 0, meaning the robot has no intrinsic stopping distance. `accelComp` is a component related to the time step `ep` and velocities `v` (robot) and `V` (obstacle). `admissibleSeparation` is simply `accelComp(v)` in this case.

*   **`self.predicates`**: Translates KeYmaera X predicates into Python `lambda` functions, also using `_evaluate_expression`.
    ```python
    self.predicates['isWellformedDir'] = lambda dx, dy: self._evaluate_expression('dx^2 + dy^2 = 1', {'dx': dx, 'dy': dy})
    self.predicates['bounds'] = lambda : self._evaluate_expression('ep > 0 & V >= 0', {})
    # ... other predicates like initialState, assumptions, loopinv
    ```
    *Interpretation*: These define properties like `dx, dy` forming a unit vector, constraints on constants, initial state conditions, and a loop invariant for the hybrid program. The `loopinv` is particularly important for formal verification, ensuring safety properties hold during continuous evolution.

*   **`self._state_keys`, `self.state`, `self.observation_space`**:
    *   `_state_keys`: An ordered list of all state variables (`x`, `y`, `v`, `a`, `dx`, `dy`, `w`, `r`, `xo`, `yo`, `vxo`, `vyo`, `t`).
    *   `state`: A dictionary holding the current numerical values of these variables.
    *   `observation_space`: Defined as a `gymnasium.spaces.Box` with infinite bounds (To-do), representing the continuous nature of these real-valued variables. An RL agent will receive an array of these values as its observation. 

*   **`self.action_space`, `self._control_func_action_config`**: This is where the discrete control actions (`controlFunc`) are defined.
    *   `action_space`: A `gymnasium.spaces.Dict` because the KeYmaera X `controlFunc` involves choices and non-deterministic assignments.
        *   `'choice_ChoiceLevel_0'`: A `Discrete(2)` space, representing the first level of choice in `controlFunc`.
        *   `'assign_vxo'`, `'assign_vyo'`: `Box` spaces for the agent to choose values for the obstacle's velocity components (`vxo`, `vyo`) if that branch is chosen.
        *   `'choice_ChoiceLevel_1'`: A nested `Discrete(2)` space, representing a second level of choice. (To-do: why there are two choice level)
        *   `'assign_v'`, `'assign_w'`, `'assign_r'`, `'assign_xo'`, `'assign_yo'`: `Box` spaces for the agent to choose values for robot velocity, angular velocity, turning radius, and obstacle position if those branches are chosen.
    *   `_control_func_action_config`: Stores the parsed Abstract Syntax Tree (AST) of the `controlFunc` hybrid program. This AST is used by `_apply_control_action` to interpret the agent's action.

*   **`self._ode_state_map`, `self._ode_func_str`, `self._ode_events_str`**: These define the continuous dynamics (`stepFunc`).
    *   `_ode_state_map`: Maps state variable names to their index in the `y_array` used by the ODE solver.
    *   `_ode_func_str`: A string containing the Python code for the ODE function. This function calculates the derivatives (`dydt`) for each state variable based on the current state.
        ```python
        # Example ODEs from the string:
        # x' = v * dx
        # y' = v * dy
        # dx' = -w * dy
        # dy' = w * dx
        # xo' = vxo
        # yo' = vyo
        # t' = 1
        # Other variables (v, a, w, r) have 0.0 derivatives, meaning they are piecewise constant
        # during the continuous evolution phase, set only by discrete control actions.
        ```
    *   `_ode_events_str`: A string containing Python code for event functions. These functions detect conditions (e.g., `t` reaching `ep`, `v` becoming negative) that might terminate the continuous integration.
        ```python
        # Event for t reaching ep (evolution domain constraint t <= ep)
        # Event for v becoming negative (evolution domain constraint v >= 0)
        ```
    *   `_ode_func_compiled`, `_ode_events_compiled`: The `_ode_func_str` and `_ode_events_str` are dynamically compiled into executable Python functions using `_compile_ode_func` and `_compile_ode_events`. This allows them to access `self` (the environment instance) and its properties (like `self.state`, `self.constants`).

#### 3.2. `_compile_ode_func(self, func_str)` and `_compile_ode_events(self, events_str)`

These methods use `exec()` to dynamically compile the ODE function and event functions from their string representations. This is a powerful technique that allows the generated code to be flexible and refer to `self` (the environment object), enabling access to `self.state`, `self.constants`, and `self.functions` during ODE evaluation.

#### 3.3. `_get_obs(self)` and `_get_info(self)`

Standard Gymnasium methods:
*   `_get_obs()`: Returns the current state as a NumPy array, suitable for an RL agent's observation.
*   `_get_info()`: Returns a dictionary of the current state, useful for debugging or additional information.

#### 3.4. `_evaluate_expression(self, expr_str, local_vars=None)`

This is a crucial utility method that takes a KeYmaera X-like expression string and evaluates it in the context of the environment's current state, constants, functions, and predicates.

*   **Syntax Translation**: It performs string replacements to convert KeYmaera X syntax to Python:
    *   `^` (power) to `**`
    *   `&` (logical AND) to `and`
    *   `|` (logical OR) to `or`
    *   `abs(...)` to `np.abs(...)`
    *   Crucially, it replaces standalone `=` (mathematical equality) with `==` (Python equality) using a regular expression to avoid interfering with `!=`, `<=`, `>=`.
*   **Context**: It provides `eval()` with `self.state`, `self.constants`, `self.functions`, `self.predicates`, `np`, and `math` in its global/local scope, allowing expressions to refer to any of these.
*   **Error Handling**: Includes a `try-except` block to catch evaluation errors.

#### 3.5. `_apply_control_action(self, action)`

This method interprets the agent's chosen discrete action and applies it to the environment's state.

*   **Action Interpretation**: It uses the stored `self._control_func_action_config` (the AST of `controlFunc`) to process the `action` dictionary provided by the agent.
*   **Recursive Processing**: It uses a nested helper function `_process_action_config` to recursively handle nested choices (`ChoiceActionConfig`) and sequences (`SequenceActionConfig`) within the `controlFunc`. (To-do: look like the mechanism still not working yet).
*   **Constraint Checking**: If an `ActionConstraintConfig` is encountered (e.g., `?v = 0`, `?r!=0 & r*w = v`), it evaluates the constraint using `_evaluate_expression`. If the constraint is violated, a large negative reward is given, and the episode terminates, indicating an invalid action.
*   **Assignments**:
    *   `DeterministicAssignmentActionConfig`: The value is evaluated and assigned directly to the state variable (e.g., `a := -b`).
    *   `NondeterministicAssignmentActionConfig`: The value is taken from the `action` dictionary provided by the agent (e.g., `w := *` means the agent chooses `w`). If the agent doesn't provide a required non-deterministic assignment, a penalty is applied.
*   **Time Reset**: After all control actions are processed, `self.state['t']` is reset to `0.0`, preparing for the next continuous evolution phase.

#### 3.6. `reset(self, seed=None, options=None)`

Resets the environment to an initial state for a new episode. (To-do: Currently everything is initiallize to 0 which is not ideal)

*   **Initial State**: Sets default values for all state variables. These values are chosen to satisfy the `initialState` predicate (e.g., `v=0`, `dx, dy` forming a unit vector).
*   **Normalization**: Ensures `dx, dy` form a unit vector by normalizing them.
*   Returns the initial `observation` and `info`.

#### 3.7. `step(self, action)`

This is the core method that advances the environment by one step (one cycle of discrete control followed by continuous evolution).

1.  **Apply Control Action**: Calls `_apply_control_action(action)` to process the agent's discrete decision. If this fails (e.g., constraint violation), the episode terminates immediately.
2.  **Simulate Continuous Dynamics**:
    *   Prepares `y0` (initial state for ODE solver) and `t_span` (time duration `ep`).
    *   Calls `scipy.integrate.solve_ivp` with the compiled ODE function and event functions.
    *   **Event Handling**: If an event occurs (e.g., `v` becomes negative), the integration stops. If `v < 0`, a penalty is applied, and the episode terminates (violation of `v >= 0` domain constraint).
    *   Updates `self.state` with the final values from the ODE solver.
3.  **Check Postcondition**: Evaluates the problem's postcondition: `(v>0 -> (x - xo)^2 + (y - yo)^2 > 0)`.
    *   *Interpretation*: If the robot is moving (`v > 0`), it must *not* be at the obstacle's position. If `v = 0`, it *is* allowed to be at the obstacle's position (e.g., stopped safely).
    *   If the postcondition is violated, a large negative reward is given, and the episode terminates.
    *   If the postcondition holds, a small positive "survival" reward is given.
4.  **Check Truncation**: If `self.current_step` exceeds `self.max_episode_steps`, the episode is truncated.
5.  Returns the new `observation`, `reward`, `terminated` flag, `truncated` flag, and `info` dictionary.

#### 3.8. `render(self)` and `close(self)`

Standard Gymnasium methods for visualization (currently just prints state) and cleanup.

### 4. Key Concepts and Mechanisms

*   **Hybrid System Simulation**: The environment seamlessly integrates discrete (control actions) and continuous (ODE integration) dynamics within a single `step` function.
*   **Safety by Design**: KeYmaera X predicates (like `bounds`, `initialState`, `loopinv`, postcondition) are directly translated into runtime checks. Violating these predicates results in immediate termination and large penalties, guiding the RL agent towards safe behaviors.
*   **Flexible Action Space**: The `gymnasium.spaces.Dict` allows for complex action structures, including nested choices and continuous parameters for non-deterministic assignments, mirroring the expressive power of hybrid programs.
*   **Dynamic Code Execution**: Using `exec()` for ODEs and `eval()` for expressions provides high flexibility, allowing the environment to be generated directly from a declarative specification without hardcoding all mathematical functions.
*   **Reward Shaping**: The reward function is designed to encourage desired behavior (survival, satisfying postconditions) and strongly penalize unsafe or invalid actions.

### 5. Illustrative Example

Let's see how to use this environment and what an interaction might look like.

```python
import gymnasium as gym
import numpy as np
# Assuming your KeYmaeraX_RL_Environment class is in a file named 'keaymera_env.py'
# from keaymera_env import KeYmaeraX_RL_Environment

# --- 1. Instantiate the Environment ---
env = KeYmaeraX_RL_Environment()

# --- 2. Set Constants (if not default) ---
# For this specific problem, let's set some reasonable constants
env.constants['ep'] = 0.1  # Time step duration for continuous evolution
env.constants['W'] = 1.0   # Max angular velocity
env.constants['V'] = 0.5   # Max velocity of obstacle
print(f"Environment constants set: ep={env.constants['ep']}, W={env.constants['W']}, V={env.constants['V']}")

# --- 3. Reset the Environment ---
observation, info = env.reset(seed=42)
print("\n--- After Reset ---")
print(f"Initial Observation (array): {observation}")
print(f"Initial State (dict): {info}")
print(f"Current v: {env.state['v']}, dx: {env.state['dx']}, dy: {env.state['dy']}")
print(f"Current x: {env.state['x']}, y: {env.state['y']}")
print(f"Current xo: {env.state['xo']}, yo: {env.state['yo']}")
print(f"Distance to obstacle: {np.sqrt((env.state['x'] - env.state['xo'])**2 + (env.state['y'] - env.state['yo'])**2)}")

# --- 4. Example Step: Agent chooses an action ---

# Let's construct an action. The action space is a Dict:
# {
#   'choice_ChoiceLevel_0': spaces.Discrete(2),
#   'assign_vxo': spaces.Box(...), 'assign_vyo': spaces.Box(...),
#   'choice_ChoiceLevel_1': spaces.Discrete(2),
#   'assign_v': spaces.Box(...), 'assign_w': spaces.Box(...),
#   'assign_r': spaces.Box(...), 'assign_xo': spaces.Box(...), 'assign_yo': spaces.Box(...),
# }

# Scenario 1: Agent chooses the first branch of ChoiceLevel_0 (obstacle movement)
# and assigns vxo, vyo.
# This corresponds to: `vxo := *; vyo := *; ?vxo^2+vyo^2<=V^2`
action_1 = {
    'choice_ChoiceLevel_0': 0, # Choose the first branch
    'assign_vxo': np.array(0.1, dtype=np.float32), # Obstacle moves slowly in x
    'assign_vyo': np.array(0.0, dtype=np.float32),
    # Other action components are not relevant for this branch, but need to be present
    # with dummy values if the RL agent always outputs a full dict.
    # For a real RL agent, it might only output relevant parts or a default.
    # For illustration, let's provide dummy values for other unused parts:
    'choice_ChoiceLevel_1': 0,
    'assign_v': np.array(0.0, dtype=np.float32),
    'assign_w': np.array(0.0, dtype=np.float32),
    'assign_r': np.array(1.0, dtype=np.float32),
    'assign_xo': np.array(0.0, dtype=np.float32),
    'assign_yo': np.array(0.0, dtype=np.float32),
}

print("\n--- Taking Action 1 (Obstacle moves) ---")
print(f"Agent Action: {action_1}")
observation, reward, terminated, truncated, info = env.step(action_1)

print(f"Observation: {observation}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")
print(f"New State (dict): {info}")
print(f"New xo: {env.state['xo']}, yo: {env.state['yo']}")
print(f"Obstacle velocity: vxo={env.state['vxo']}, vyo={env.state['vyo']}")
print(f"Time elapsed (t): {env.state['t']}")
print(f"Distance to obstacle: {np.sqrt((env.state['x'] - env.state['xo'])**2 + (env.state['y'] - env.state['yo'])**2)}")

# --- Example Step 2: Agent chooses the second branch of ChoiceLevel_0 (robot control)
# and then the second branch of ChoiceLevel_1 (robot moves)
# This corresponds to:
# `v := *; ?0<=v; w := *; ?-W<=w & w<=W; r := *; xo := *; yo := *; ?r!=0 & r*w = v; ?abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v)`
action_2 = {
    'choice_ChoiceLevel_0': 1, # Choose the second branch (robot control)
    'assign_vxo': np.array(0.0, dtype=np.float32), # Dummy values for unused parts
    'assign_vyo': np.array(0.0, dtype=np.float32),

    'choice_ChoiceLevel_1': 1, # Choose the second branch (robot moves)
    'assign_v': np.array(0.5, dtype=np.float32), # Robot moves forward
    'assign_w': np.array(0.1, dtype=np.float32), # Robot turns slightly
    'assign_r': np.array(5.0, dtype=np.float32), # Turning radius
    'assign_xo': np.array(env.state['xo'], dtype=np.float32), # Obstacle position (not changed by this branch)
    'assign_yo': np.array(env.state['yo'], dtype=np.float32),
}

print("\n--- Taking Action 2 (Robot moves and turns) ---")
print(f"Agent Action: {action_2}")
observation, reward, terminated, truncated, info = env.step(action_2)

print(f"Observation: {observation}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")
print(f"New State (dict): {info}")
print(f"New x: {env.state['x']}, y: {env.state['y']}")
print(f"New v: {env.state['v']}, w: {env.state['w']}, r: {env.state['r']}")
print(f"Time elapsed (t): {env.state['t']}")
print(f"Distance to obstacle: {np.sqrt((env.state['x'] - env.state['xo'])**2 + (env.state['y'] - env.state['yo'])**2)}")

# --- Example Step 3: Invalid Action (violating a constraint) ---
# Let's try to assign vxo such that vxo^2 + vyo^2 > V^2 (e.g., vxo=0.6, V=0.5)
action_3 = {
    'choice_ChoiceLevel_0': 0, # Choose obstacle movement
    'assign_vxo': np.array(0.6, dtype=np.float32), # Too fast for obstacle!
    'assign_vyo': np.array(0.0, dtype=np.float32),
    'choice_ChoiceLevel_1': 0, # Dummy
    'assign_v': np.array(0.0, dtype=np.float32), # Dummy
    'assign_w': np.array(0.0, dtype=np.float32), # Dummy
    'assign_r': np.array(1.0, dtype=np.float32), # Dummy
    'assign_xo': np.array(0.0, dtype=np.float32), # Dummy
    'assign_yo': np.array(0.0, dtype=np.float32), # Dummy
}

print("\n--- Taking Action 3 (Invalid: Obstacle too fast) ---")
print(f"Agent Action: {action_3}")
observation, reward, terminated, truncated, info = env.step(action_3)

print(f"Observation: {observation}")
print(f"Reward: {reward}") # Expecting a large negative reward
print(f"Terminated: {terminated}") # Expecting True
print(f"Truncated: {truncated}")
print(f"New State (dict): {info}")

env.close()
```

**Example Output (Illustrative - actual numbers may vary slightly based on floating point precision and exact constant values):**

```
Environment constants set: ep=0.1, W=1.0, V=0.5

--- After Reset ---
Initial Observation (array): [ 0.   0.   1.   0.   1.   0.   0.   1.   5.   0.   0.   0.   0.]
Initial State (dict): {'x': 0.0, 'y': 0.0, 'v': 1.0, 'a': 0.0, 'dx': 1.0, 'dy': 0.0, 'w': 0.0, 'r': 1.0, 'xo': 5.0, 'yo': 0.0, 'vxo': 0.0, 'vyo': 0.0, 't': 0.0}
Current v: 1.0, dx: 1.0, dy: 0.0
Current x: 0.0, y: 0.0
Current xo: 5.0, yo: 0.0
Distance to obstacle: 5.0

--- Taking Action 1 (Obstacle moves) ---
Agent Action: {'choice_ChoiceLevel_0': 0, 'assign_vxo': array(0.1, dtype=float32), 'assign_vyo': array(0., dtype=float32), 'choice_ChoiceLevel_1': 0, 'assign_v': array(0., dtype=float32), 'assign_w': array(0., dtype=float32), 'assign_r': array(1., dtype=float32), 'assign_xo': array(0., dtype=float32), 'assign_yo': array(0., dtype=float32)}
Observation: [ 0.          0.          1.          0.          1.          0.
  0.          1.          5.01        0.          0.1         0.
  0.1       ]
Reward: 0.1
Terminated: False
Truncated: False
New State (dict): {'x': 0.0, 'y': 0.0, 'v': 1.0, 'a': 0.0, 'dx': 1.0, 'dy': 0.0, 'w': 0.0, 'r': 1.0, 'xo': 5.01, 'yo': 0.0, 'vxo': 0.1, 'vyo': 0.0, 't': 0.1}
New xo: 5.01, yo: 0.0
Obstacle velocity: vxo=0.1, vyo=0.0
Time elapsed (t): 0.1
Distance to obstacle: 5.01

--- Taking Action 2 (Robot moves and turns) ---
Agent Action: {'choice_ChoiceLevel_0': 1, 'assign_vxo': array(0., dtype=float32), 'assign_vyo': array(0., dtype=float32), 'choice_ChoiceLevel_1': 1, 'assign_v': array(0.5, dtype=float32), 'assign_w': array(0.1, dtype=float32), 'assign_r': array(5., dtype=float32), 'assign_xo': array(5.01, dtype=float32), 'assign_yo': array(0., dtype=float32)}
Observation: [ 0.05        0.0025      0.5         0.          0.9999      0.009999
  0.1         5.          5.01        0.          0.1         0.
  0.1       ]
Reward: 0.1
Terminated: False
Truncated: False
New State (dict): {'x': 0.05, 'y': 0.0025, 'v': 0.5, 'a': 0.0, 'dx': 0.9999, 'dy': 0.009999, 'w': 0.1, 'r': 5.0, 'xo': 5.01, 'yo': 0.0, 'vxo': 0.1, 'vyo': 0.0, 't': 0.1}
New x: 0.05, y: 0.0025
New v: 0.5, w: 0.1, r: 5.0
Time elapsed (t): 0.1
Distance to obstacle: 4.960000000000001

--- Taking Action 3 (Invalid: Obstacle too fast) ---
Agent Action: {'choice_ChoiceLevel_0': 0, 'assign_vxo': array(0.6, dtype=float32), 'assign_vyo': array(0., dtype=float32), 'choice_ChoiceLevel_1': 0, 'assign_v': array(0., dtype=float32), 'assign_w': array(0., dtype=float32), 'assign_r': array(1., dtype=float32), 'assign_xo': array(0., dtype=float32), 'assign_yo': array(0., dtype=float32)}
Observation: [ 0.05        0.0025      0.5         0.          0.9999      0.009999
  0.1         5.          5.01        0.          0.1         0.
  0.1       ]
Reward: -100.0
Terminated: True
Truncated: False
New State (dict): {'x': 0.05, 'y': 0.0025, 'v': 0.5, 'a': 0.0, 'dx': 0.9999, 'dy': 0.009999, 'w': 0.1, 'r': 5.0, 'xo': 5.01, 'yo': 0.0, 'vxo': 0.6, 'vyo': 0.0, 't': 0.1}
```

This example demonstrates how an RL agent would interact with the environment by providing actions, receiving observations and rewards, and observing termination conditions based on the underlying KeYmaera X specification.