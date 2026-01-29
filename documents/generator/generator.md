This Python script acts as a **code generator**. Its primary function is to translate a structured representation (an Abstract Syntax Tree, or AST) of a hybrid system, typically originating from a formal verification tool like KeYmaera X, into a runnable [Gymnasium (Gym)] environment. This allows for the simulation and interaction with complex hybrid systems within a reinforcement learning framework.

---

### 1. Imports and AST Configuration

```python
import textwrap
from typing import List, Dict, Any, Optional, Tuple, Union
import re # <--- ADDED THIS IMPORT

# Assuming ast_config.py is in the same directory and contains the ASTConfigNode classes
from ast_config import (
    GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig,
    ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig,
    HybridProgramConfig,
    ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig,
    VariableDeclarationConfig, PreconditionConfig, PostconditionConfig,
    LoopExecutionConfig, HybridProgramStepConfig,
    ActionType, ChoiceActionConfig, SequenceActionConfig,
    DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig,
    ActionConstraintConfig, CallHybridProgramConfig
)
```

*   **`import textwrap`**: This module is used for formatting text, particularly for indenting blocks of generated code to maintain readability.
*   **`from typing import ...`**: Provides type hints, which improve code readability and maintainability by specifying expected types for variables, function arguments, and return values.
*   **`import re`**: The regular expression module, used for pattern matching and string manipulation, especially in the `_evaluate_expression` function to translate KeYmaera X syntax to Python.
*   **`from ast_config import ...`**: This is a crucial import. It signifies that there's an external file (`ast_config.py`) that defines the data structures (classes) representing the Abstract Syntax Tree (AST) of the hybrid system. These classes (e.g., `GymEnvironmentConfig`, `ControlFuncConfig`, `ODEConfig`) are the *input schema* for this code generator. They define how the various components of a hybrid system (constants, functions, predicates, variables, control logic, differential equations, problem statement) are structured in memory.

---

### 2. `_ast_to_python_code_string` Function

```python
def _ast_to_python_code_string(ast_node: ActionType, indent_level: int = 0) -> str:
    """
    Recursively converts an ActionType AST node into its Python constructor call string.
    This is used to embed the control function's action configuration directly into the
    generated environment's __init__ method.
    """
    indent_str = "    " * indent_level
    next_indent_str = "    " * (indent_level + 1)

    if isinstance(ast_node, ChoiceActionConfig):
        choices_str = ",\n".join([
            f"{next_indent_str}{_ast_to_python_code_string(choice, indent_level + 1)}"
            for choice in ast_node.choices
        ])
        return (
            f"ChoiceActionConfig(level_name='{ast_node.level_name}', choices=[\n"
            f"{choices_str}\n"
            f"{indent_str}])"
        )
    elif isinstance(ast_node, SequenceActionConfig):
        actions_str = ",\n".join([
            f"{next_indent_str}{_ast_to_python_code_string(action, indent_level + 1)}"
            for action in ast_node.actions
        ])
        return (
            f"SequenceActionConfig(actions=[\n"
            f"{actions_str}\n"
            f"{indent_str}])"
        )
    elif isinstance(ast_node, DeterministicAssignmentActionConfig):
        return f"DeterministicAssignmentActionConfig(variable='{ast_node.variable}', value='{ast_node.value}')"
    elif isinstance(ast_node, NondeterministicAssignmentActionConfig):
        return f"NondeterministicAssignmentActionConfig(variable='{ast_node.variable}')"
    elif isinstance(ast_node, ActionConstraintConfig):
        # Ensure formula is single-line for string literal
        formula_single_line = ast_node.formula.replace('\n', ' ').strip()
        return f"ActionConstraintConfig(formula='{formula_single_line}')"
    elif isinstance(ast_node, CallHybridProgramConfig):
        return f"CallHybridProgramConfig(hp_name='{ast_node.hp_name}')"
    elif isinstance(ast_node, HybridProgramStepConfig):
        # Ensure invariant formula is single-line for string literal
        invariant_single_line = ast_node.invariant_formula.replace('\n', ' ').strip()
        return f"HybridProgramStepConfig(hp_name='{ast_node.hp_name}', invariant_formula='{invariant_single_line}')"
    else:
        raise ValueError(f"Unknown AST node type: {type(ast_node)}")
```

*   **Purpose**: This is a recursive helper function responsible for converting an `ActionType` AST node (which defines discrete control actions) into a Python string that, when executed, would reconstruct that exact AST object.
*   **Mechanism**:
    *   It takes an `ast_node` (an instance of `ChoiceActionConfig`, `SequenceActionConfig`, etc.) and an `indent_level` for formatting.
    *   It uses `isinstance()` to check the type of the `ast_node`.
    *   For each known `ActionType`, it constructs an f-string representing the Python constructor call for that class.
    *   For composite actions like `ChoiceActionConfig` or `SequenceActionConfig`, it recursively calls itself for each nested action, ensuring proper indentation and structure.
    *   **Why it's important**: This function allows the generator to embed the entire complex structure of the `controlFunc`'s action configuration directly as a Python object initialization within the generated Gymnasium environment's `__init__` method. This makes the environment self-contained regarding its action logic.

---

### 3. `generate_gym_environment_code` Function (Main Generator)

```python
def generate_gym_environment_code(config: GymEnvironmentConfig) -> str:
    """
    Generates Python code for a Gymnasium environment based on the provided
    GymEnvironmentConfig AST.
    """
    env_name = config.env_name
    
    # Collect all variable names for observation space and ODE state vector
    state_variable_names = [v.name for v in config.variables.state_variables]

    # --- Generate Constants, Functions, Predicates ---
    constants_init_code_lines = []
    for const in config.definitions.constants:
        #Problematic line: default_value could break the env: ep = 0
        default_val = '0.0' if const.type == 'Real' else 'False'
        constants_init_code_lines.append(f"self.constants['{const.name}'] = {const.value if const.value is not None else default_val}")
    # Pre-indent this block
    indented_constants_block = textwrap.indent("\n".join(constants_init_code_lines), '            ')

    functions_init_code_lines = []
    for func in config.definitions.functions:
        params_str = ", ".join([p_name for _, p_name in func.parameters])
        params_dict_str = ", ".join([f"'{p_name}': {p_name}" for _, p_name in func.parameters])
        # Ensure function body is a single line for embedding in string literal
        body_single_line = func.body.replace('\n', ' ').strip()
        functions_init_code_lines.append(f"self.functions['{func.name}'] = lambda {params_str}: self._evaluate_expression('{body_single_line}', {{{params_dict_str}}})")
    # Pre-indent this block
    indented_functions_block = textwrap.indent("\n".join(functions_init_code_lines), '            ')

    predicates_init_code_lines = []
    for pred in config.definitions.predicates:
        params_str = ", ".join([p_name for _, p_name in pred.parameters])
        params_dict_str = ", ".join([f"'{p_name}': {p_name}" for _, p_name in pred.parameters])
        # Ensure predicate body is a single line for embedding in string literal
        body_single_line = pred.body.replace('\n', ' ').strip()
        predicates_init_code_lines.append(f"self.predicates['{pred.name}'] = lambda {params_str}: self._evaluate_expression('{body_single_line}', {{{params_dict_str}}})")
    # Pre-indent this block
    indented_predicates_block = textwrap.indent("\n".join(predicates_init_code_lines), '            ')

    # --- Generate Action Space ---
    action_space_definition, control_func_ast_object, action_space_keys = _generate_action_space_and_logic(config.definitions.hybrid_programs)
    
    # Convert the control_func_ast_object to its Python code string representation
    control_func_ast_code_str = _ast_to_python_code_string(control_func_ast_object, indent_level=3) # Indent for __init__ method

    # --- Generate ODEs and Evolution Domain ---
    ode_func_code, ode_event_code, ode_state_map, ode_state_idx_map = _generate_ode_solver_logic(config.definitions.hybrid_programs, state_variable_names)

    # --- Generate Initial State (reset logic) ---
    reset_initial_state_code_lines = _generate_reset_logic(config.variables.state_variables, config.problem.precondition)
    # Pre-indent this block
    indented_reset_block = textwrap.indent("\n".join(reset_initial_state_code_lines), '            ')

    # --- Generate Problem Logic (Postcondition, Loop Invariants) ---
    postcondition_formula_py = config.problem.postcondition.formula.replace('\n', ' ').strip() # Ensure single line
    loop_invariant_formula_py = config.problem.loop_execution.loop_invariant_formula.replace('\n', ' ').strip() # Ensure single line

    # Imports for AST classes needed for _control_func_action_config
    ast_imports = [
        "ChoiceActionConfig", "SequenceActionConfig", "DeterministicAssignmentActionConfig",
        "NondeterministicAssignmentActionConfig", "ActionConstraintConfig", "CallHybridProgramConfig"
    ]
    ast_import_statement = f"from ast_config import ({', '.join(ast_imports)})"

    # --- Main Environment Class Structure ---
    env_code = textwrap.dedent(f"""
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
    from scipy.integrate import solve_ivp
    import math # For math functions like sqrt, sin, cos, etc.
    import textwrap
    import re # For regex replacements in _evaluate_expression
    {ast_import_statement} # Import AST classes

    class {env_name}(gym.Env):
        metadata = {{"render_modes": ["human"], "render_fps": 30}}

        def __init__(self, render_mode=None):
            super().__init__()
            self.render_mode = render_mode

            # --- KeYmaera X Definitions ---
            self.constants = {{}}
            self.functions = {{}}
            self.predicates = {{}}
            
            # Constants
{indented_constants_block}

            # Functions (lambdas for easy evaluation)
{indented_functions_block}

            # Predicates (lambdas for easy evaluation)
{indented_predicates_block}

            # --- State Variables ---
            # Mapping of state variable names to their index in the observation array
            self._state_keys = {state_variable_names}
            # Current state values (dictionary for easy access by name)
            self.state = {{key: 0.0 for key in self._state_keys}} 
            # Initial state values are set in reset()

            # --- Observation Space ---
            # All state variables are Real, so Box space is appropriate.
            # Bounds are infinite by default, as not specified in KeYmaera X input.
            low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
            high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
            self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)

            # --- Action Space (from controlFunc) ---
            # The action space is a Dict to handle choices and nondeterministic assignments.
{textwrap.indent(action_space_definition, '            ')}
            self._action_space_keys = {list(action_space_keys.keys())}
            self._control_func_action_config = {control_func_ast_code_str} # Stored for step logic

            # --- ODE Solver Setup (from stepFunc) ---
            # Mapping from state variable names to their index in the ODE solver's y array
            self._ode_state_map = {ode_state_map}
            self._ode_state_idx_map = {ode_state_idx_map} # Map index to name
            
            self._ode_func_str = '''
{textwrap.indent(ode_func_code, '                ')}
            '''
            self._ode_events_str = '''
{textwrap.indent(ode_event_code, '                ')}
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
            # ... (rest of the class definition) ...
```

*   **Purpose**: This is the core function that orchestrates the entire code generation process. It takes the `GymEnvironmentConfig` AST and produces the complete Python code for a Gymnasium environment.
*   **Workflow Breakdown**:
    1.  **`env_name = config.env_name`**: Extracts the desired name for the Gymnasium environment class.
    2.  **`state_variable_names = [...]`**: Gathers all state variable names from the AST. These will define the observation space and the variables tracked by the ODE solver.
    3.  **Constants, Functions, Predicates Generation**:
        *   It iterates through `config.definitions.constants`, `config.definitions.functions`, and `config.definitions.predicates`.
        *   For **constants**, it generates lines like `self.constants['name'] = value` to be placed in the `__init__` method. If a constant has no explicit value, a default (`0.0` for Real, `False` for Boolean) is used.
        *   For **functions** and **predicates**, it generates Python `lambda` functions. These lambdas take parameters (e.g., `v` for `stopDist(v)`) and internally call `self._evaluate_expression()` with the function's body string and a dictionary of local parameters. This allows functions/predicates to be evaluated dynamically within the environment's context.
        *   `textwrap.indent()` is used to ensure these generated lines are correctly indented within the `__init__` method of the final class.
    4.  **Action Space Generation**:
        *   `_generate_action_space_and_logic()` (explained later) is called to determine the Gymnasium `action_space` definition (e.g., `spaces.Dict` with `spaces.Discrete` and `spaces.Box` components).
        *   It also returns the actual AST object for the `controlFunc`'s action configuration. This object is then converted into a Python string using `_ast_to_python_code_string()` (explained above) and stored in `control_func_ast_code_str`. This string will be embedded directly into the `__init__` method as `self._control_func_action_config`.
    5.  **ODE Solver Logic Generation**:
        *   `_generate_ode_solver_logic()` (explained later) is called to produce the Python code strings for the ODE function (`_ode_func_code`) and event functions (`_ode_event_code`) that `scipy.integrate.solve_ivp` will use.
        *   It also returns mappings (`_ode_state_map`, `_ode_state_idx_map`) between state variable names and their indices in the ODE solver's `y` array.
    6.  **Initial State (Reset Logic) Generation**:
        *   `_generate_reset_logic()` (explained later) produces the Python code lines for initializing the state variables in the `reset()` method.
    7.  **Problem Logic Extraction**:
        *   The `postcondition` and `loop_invariant_formula` are extracted from the `config.problem` and cleaned up into single-line strings.
    8.  **Main Environment Class Structure**:
        *   The generated code is assembled into a large f-string using `textwrap.dedent()`. This f-string defines the `class {env_name}(gym.Env):`.
        *   All the previously generated code blocks (constants, functions, predicates, action space, ODE setup, reset logic) are inserted into their appropriate places within the class's `__init__` method or as helper methods.
        *   It includes necessary imports (`gymnasium`, `numpy`, `scipy`, `math`, `textwrap`, `re`, and the `ast_config` classes) at the top of the generated file.

---

### 4. Helper Methods within the Generated Gymnasium Environment Class

The following methods are part of the *generated* Python class (e.g., `IJRR17_Theorem_1_Static_safety`), not the generator itself.

#### `_compile_ode_func` and `_compile_ode_events`

```python
        def _compile_ode_func(self, func_str):
            # Create a dictionary for locals that includes 'self'
            exec_globals = {{'np': np, 'math': math}}
            exec_locals = {{'self': self}}
            # Dedent the string before execution, as it was indented for the triple-quote literal
            exec(textwrap.dedent(func_str), exec_globals, exec_locals)
            return exec_locals['ode_func']

        def _compile_ode_events(self, events_str):
            if not events_str.strip():
                return []
            exec_globals = {{'np': np, 'math': math}}
            exec_locals = {{'self': self}}
            # Dedent the string before execution
            exec(textwrap.dedent(events_str), exec_globals, exec_locals)
            return exec_locals['ode_events']
```

*   **Purpose**: These methods are crucial for dynamically compiling the ODE function and event functions (which were generated as strings) into actual Python callable objects.
*   **Mechanism**:
    *   They use `exec()` to execute the provided `func_str` or `events_str`.
    *   **Dynamic Context**: The `exec()` function is given `exec_globals` and `exec_locals` dictionaries. Crucially, `exec_locals` includes `{'self': self}`. This means that the code executed by `exec()` (i.e., the generated `ode_func` or event functions) will have access to the `self` instance of the Gymnasium environment. This allows the ODEs and events to call `self._evaluate_expression()` and access `self.state`, `self.constants`, etc., during integration.
    *   `textwrap.dedent()` is used to remove any leading indentation from the multiline string literals before execution.
    *   They return the compiled function (`ode_func`) or a list of compiled event functions (`ode_events`).

#### `_evaluate_expression`

```python
        def _evaluate_expression(self, expr_str, local_vars=None):
            # Safe evaluation of expressions using environment's context
            if local_vars is None:
                local_vars = {{}}
            
            # Strip KeYmaera X constraint prefix '?' if present
            if expr_str.startswith('?'):
                expr_str = expr_str[1:].strip()

            # Combine all available variables for evaluation context
            eval_globals = {{
                'np': np, 'math': math,
                **self.constants,
                **self.functions,
                **self.predicates
            }}
            eval_locals = {{
                **self.state,
                **local_vars
            }}
            
            # Replace common KeYmaera X syntax with Python
            expr_py = expr_str.replace('^', '**').replace('abs(', 'np.abs(')
            
            # Handle logical operators (assuming KeYmaera X uses '&' and '|' for logical AND/OR)
            expr_py = expr_py.replace('&', 'and').replace('|', 'or')
            
            # Handle comparison operators:
            # These are mostly no-ops if the input already uses Python syntax, but ensures consistency.
            expr_py = expr_py.replace('==', '==') 
            expr_py = expr_py.replace('!=', '!=')
            expr_py = expr_py.replace('>=', '>=')
            expr_py = expr_py.replace('<=', '<=')
            
            # Replace standalone '=' with '==' for equality.
            # This regex ensures that '=' is not part of '<=', '>=', '!=' or '=='
            expr_py = re.sub(r'(?<![<>=!])=(?![=])', '==', expr_py) # <--- MODIFIED LINE
            
            try:
                return eval(expr_py, eval_globals, eval_locals)
            except Exception as e:
                print(f"Error evaluating expression '{{expr_str}}': {{e}}")
                print(f"Eval globals: {{eval_globals}}")
                print(f"Eval locals: {{eval_locals}}")
                raise
```

*   **Purpose**: This is a critical utility method for safely evaluating arbitrary mathematical or logical expression strings (originating from KeYmaera X formulas) within the Python environment.
*   **Mechanism**:
    *   **Syntax Translation**: It performs several string replacements to convert common KeYmaera X syntax into valid Python:
        *   `^` (power) to `**`.
        *   `abs(` to `np.abs(` (using `numpy`'s absolute function for numerical stability).
        *   `&` (logical AND) to `and`.
        *   `|` (logical OR) to `or`.
        *   A regex `re.sub(r'(?<![<>=!])=(?![=])', '==', expr_py)` is used to replace a standalone `=` (which often means equality in KeYmaera X constraints) with `==` for Python equality, *without* affecting operators like `<=`, `>=`, `!=`, or `==`.
    *   **Evaluation Context**: It constructs `eval_globals` and `eval_locals` dictionaries to provide a controlled and safe execution environment for `eval()`:
        *   `eval_globals` includes `np`, `math`, `self.constants`, `self.functions`, and `self.predicates`. This makes these available as global names during evaluation.
        *   `eval_locals` includes `self.state` (the current values of all state variables) and any `local_vars` passed to the function (e.g., specific variable values from the ODE solver at a given time point).
    *   **`eval()`**: The built-in `eval()` function executes the translated `expr_py` string within the defined `eval_globals` and `eval_locals` context.
    *   **Error Handling**: Includes a `try-except` block to catch and report errors during expression evaluation.

#### `_apply_control_action`

```python
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
                        # print(f"Constraint violated: {{sub_action_config.formula}}")
                        reward -= 100.0 # Penalty for invalid action
                        terminated = True
                        return reward, terminated, False # False means action was not fully applied
                elif isinstance(sub_action_config, DeterministicAssignmentActionConfig):
                    # Apply deterministic assignment
                    assigned_value = self._evaluate_expression(sub_action_config.value)
                    self.state[sub_action_config.variable] = assigned_value
                elif isinstance(sub_action_config, NondeterministicAssignmentActionConfig):
                    # Apply nondeterministic assignment from agent's action
                    assign_key = f"assign_{{sub_action_config.variable}}"
                    if assign_key in action:
                        self.state[sub_action_config.variable] = action[assign_key]
                    else:
                        # This should not happen if action space is constructed correctly
                        # print(f"Warning: Nondeterministic assignment for {{sub_action_config.variable}} not found in action.")
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
                            assign_key = f"assign_{{inner_action.variable}}"
                            if assign_key in action:
                                self.state[inner_action.variable] = action[assign_key]
                            else:
                                reward -= 50.0
                                terminated = True
                                return reward, terminated, False
            
            # Apply t := 0; after controlFunc, as specified in the problem loop body
            self.state['t'] = 0.0
            
            return reward, terminated, True # True means action was successfully applied
```

*   **Purpose**: This method implements the discrete control logic defined by the `controlFunc` in the KeYmaera X model. It applies the agent's chosen action to the environment's state.
*   **Mechanism**:
    *   It receives the `action` dictionary from the Gym agent.
    *   It identifies the chosen branch of the `ChoiceActionConfig` (e.g., `action['choice_ChoiceLevel_0']`).
    *   It then iterates through the `sub_action_config` objects (which are the actual AST nodes stored in `self._control_func_action_config`) within the chosen branch.
    *   **Action Types**:
        *   `ActionConstraintConfig`: Evaluates the constraint's formula using `self._evaluate_expression()`. If the constraint is false, a penalty is applied, `terminated` is set to `True`, and the function returns, indicating an invalid action.
        *   `DeterministicAssignmentActionConfig`: Evaluates the `value` expression and assigns the result to the specified `variable` in `self.state`.
        *   `NondeterministicAssignmentActionConfig`: Takes the value for the `variable` directly from the agent's `action` dictionary (e.g., `action['assign_w']`). If the agent fails to provide a value, a penalty is applied.
        *   `SequenceActionConfig`: Handles nested sequences of actions.
    *   **`t := 0`**: After all control actions are applied, `self.state['t']` is reset to `0.0`, as is typical for hybrid programs where time resets after a discrete transition.
    *   Returns the `reward`, `terminated` status, and a boolean indicating if the control action was successfully applied without constraint violations.

#### `reset`

```python
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0

            # --- Initial State (based on Problem Precondition and defaults) ---
{indented_reset_block}
            
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
```

*   **Purpose**: Resets the environment to an initial state for a new episode.
*   **Mechanism**:
    *   Calls `super().reset()` for Gymnasium boilerplate.
    *   Resets `self.current_step`.
    *   The `indented_reset_block` (generated by `_generate_reset_logic`) assigns default initial values to all state variables in `self.state`.
    *   **Direction Vector Normalization**: A specific logic is included to normalize the `(dx, dy)` vector (representing direction) to a unit vector. This is crucial for many dynamic systems where `dx` and `dy` define an orientation. If the initial `dx, dy` are too small, it defaults to `(1.0, 0.0)`.
    *   Returns the initial `observation` (a NumPy array of state values) and `info` (a dictionary of state values).

#### `step`

```python
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
                    # print(f"ODE integration failed: {{solution.message}}")
                    reward -= 150.0 # Severe penalty for integration failure
                    terminated = True

            except Exception as e:
                # print(f"Error during ODE integration: {{e}}")
                reward -= 200.0 # Very severe penalty for unexpected errors
                terminated = True

            # --- 3. Check Problem Postcondition ---
            # (x - xo)^2 + (y - yo)^2 > 0
            # This means the robot must NOT be at the obstacle's position.
            # If it becomes <= 0, safety is violated.
            postcondition_satisfied = self._evaluate_expression('{postcondition_formula_py}')
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
                print(f"Render: Step={{self.current_step}}, State={{self.state}}")

        def close(self):
            # No specific resources to close for this environment
            pass
```

*   **Purpose**: Executes one full simulation step of the hybrid system, combining discrete control actions and continuous dynamics.
*   **Mechanism**:
    1.  **Apply Control Function**: Calls `self._apply_control_action(action)`. If this discrete action is invalid (e.g., violates a constraint), the step might end early, returning a penalty.
    2.  **Simulate Differential Equations**:
        *   Prepares `y0`, the initial state vector for the ODE solver, from `self.state`.
        *   Sets `t_span` for integration (from `0` to `ep`, where `ep` is a constant).
        *   Calls `scipy.integrate.solve_ivp()`, passing the dynamically compiled `_ode_func_compiled` and `_ode_events_compiled`.
        *   **Solution Handling**:
            *   If `solution.status == 0` (successful completion without events), `self.state` is updated with the final values from the ODE solver.
            *   If `solution.status == 1` (an event occurred), `self.state` is updated to the values at the time of the event. It then checks if `v < 0`, which signifies a domain violation (velocity cannot be negative), applying a penalty and terminating the episode if so. The `t <= ep` event simply means the time limit was reached, which is expected.
            *   If `solution.status == 2` (integration failed), a severe penalty is applied, and the episode terminates.
        *   Includes `try-except` for general errors during ODE integration.
    3.  **Check Problem Postcondition**: Evaluates the `postcondition_formula_py` (e.g., `(x - xo)^2 + (y - yo)^2 > 0`) using `self._evaluate_expression()`. If the postcondition is violated (e.g., the robot hits the obstacle), a penalty is applied, and the episode terminates. A small positive reward is given for satisfying the postcondition (i.e., staying safe).
    4.  **Check Truncation**: If the `max_episode_steps` is reached, the episode is `truncated`, and a small penalty is applied.
    *   Finally, returns the new `observation`, `reward`, `terminated` status, `truncated` status, and `info`.

#### `render` and `close`

```python
        def render(self):
            if self.render_mode == "human":
                # In a real environment, this would draw the robot, obstacle, etc.
                # For now, just print state for demonstration.
                print(f"Render: Step={{self.current_step}}, State={{self.state}}")

        def close(self):
            # No specific resources to close for this environment
            pass
```

*   **`render`**: A placeholder. In a full implementation, this would visualize the environment state. Currently, it just prints the state if `render_mode` is "human".
*   **`close`**: A placeholder. Used to clean up resources if the environment uses any (e.g., GUI windows, file handles).

---

### 5. Helper Functions for Generator (`_generate_action_space_and_logic`, `_generate_ode_solver_logic`, `_generate_reset_logic`)

These functions are called by `generate_gym_environment_code` to produce specific parts of the environment's code.

#### `_generate_action_space_and_logic`

```python
def _generate_action_space_and_logic(hps: List[HybridProgramConfig]):
    """
    Generates the Gymnasium action space definition and returns the actual AST object
    for the control function's action configuration.
    """
    control_func_config = next((hp for hp in hps if isinstance(hp, ControlFuncConfig)), None)
    if not control_func_config or not control_func_config.action_space_config:
        # Return a dummy action space and a placeholder AST object if controlFunc is missing
        return "self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)", \
               ChoiceActionConfig(level_name='Default', choices=[]), {}

    action_space_dict = {}
    action_space_keys = {}

    def _recursive_build_action_space(action_config: ActionType):
        if isinstance(action_config, ChoiceActionConfig):
            choice_key = f"choice_{action_config.level_name}"
            if choice_key not in action_space_dict:
                action_space_dict[choice_key] = f"spaces.Discrete({len(action_config.choices)})"
                action_space_keys[choice_key] = "Discrete"
            for choice_branch_config in action_config.choices:
                _recursive_build_action_space(choice_branch_config)
        elif isinstance(action_config, SequenceActionConfig):
            for sub_action in action_config.actions:
                _recursive_build_action_space(sub_action)
        elif isinstance(action_config, NondeterministicAssignmentActionConfig):
            assign_key = f"assign_{action_config.variable}"
            if assign_key not in action_space_dict:
                action_space_dict[assign_key] = "spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)"
                action_space_keys[assign_key] = "Box"

    _recursive_build_action_space(control_func_config.action_space_config)

    action_space_str_parts = []
    for key, val_str in action_space_dict.items():
        action_space_str_parts.append(f"'{key}': {val_str},")
    
    if not action_space_str_parts:
        action_space_definition = "self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)"
    else:
        # Indent the dictionary items correctly
        indented_items = textwrap.indent("\n".join(action_space_str_parts), ' ' * 8)
        action_space_definition = f"self.action_space = spaces.Dict({{\n{indented_items}\n            }})"

    # Return the actual AST object for the control function
    return action_space_definition, control_func_config.action_space_config, action_space_keys
```

*   **Purpose**: To generate the Gymnasium `action_space` definition (as a string) and extract the `controlFunc`'s action AST for later embedding.
*   **Mechanism**:
    *   It first finds the `ControlFuncConfig` within the list of hybrid programs.
    *   A nested recursive function `_recursive_build_action_space` traverses the `controlFunc`'s `action_space_config` AST:
        *   If it encounters a `ChoiceActionConfig`, it adds an entry to `action_space_dict` for a `spaces.Discrete` action, representing the choice among branches. The key is `choice_{level_name}`.
        *   If it encounters a `NondeterministicAssignmentActionConfig`, it adds an entry for a `spaces.Box` action, representing a continuous value that the agent must choose for that variable. The key is `assign_{variable_name}`.
        *   `SequenceActionConfig` simply triggers recursive calls for its sub-actions.
    *   After the recursion, `action_space_dict` contains all necessary components for the `spaces.Dict` action space.
    *   It then formats this dictionary into a Python string that can be assigned to `self.action_space` in the generated environment.
    *   It returns the formatted string, the original `control_func_ast_object`, and a dictionary of action space keys.

#### `_generate_ode_solver_logic`

```python
def _generate_ode_solver_logic(hps: List[HybridProgramConfig], state_variable_names: List[str]):
    """
    Generates the Python code for the ODE function and event functions
    for scipy.integrate.solve_ivp.
    """
    step_func_config = next((hp for hp in hps if isinstance(hp, StepFuncConfig)), None)
    if not step_func_config:
        return "", "", {}, {}

    ode_state_map = {var_name: i for i, var_name in enumerate(state_variable_names)}
    ode_state_idx_map = {i: var_name for i, var_name in enumerate(state_variable_names)}

    # ODE function `f(t, y)`
    ode_func_lines = []
    ode_func_lines.append("def ode_func(t_val, y_array):")
    ode_func_lines.append("    # Map y array to named variables for easier access")
    for var_name, idx in ode_state_map.items():
        ode_func_lines.append(f"    {var_name} = y_array[{idx}]")
    ode_func_lines.append("")
    ode_func_lines.append("    # Ensure self.state is updated with current ODE values for expression evaluation")
    ode_func_lines.append("    for i, var_name in self._ode_state_idx_map.items():")
    ode_func_lines.append("        self.state[var_name] = y_array[i]")
    ode_func_lines.append("")

    ode_func_lines.append("    # Calculate derivatives")
    derivatives = ['0.0'] * len(state_variable_names)
    for ode in step_func_config.odes:
        idx = ode_state_map.get(ode.variable)
        if idx is not None:
            # Ensure ODE equation is a single line
            equation_single_line = ode.equation.replace('\n', ' ').strip()
            derivatives[idx] = f"self._evaluate_expression('{equation_single_line}')"
        else:
            print(f"Warning: ODE for variable '{ode.variable}' not found in state_variable_names. Skipping.")

    ode_func_lines.append("    dydt = [")
    for i, deriv_expr in enumerate(derivatives):
        ode_func_lines.append(f"        {deriv_expr}, # d{ode_state_idx_map[i]}/dt")
    ode_func_lines.append("    ]")
    ode_func_lines.append("    return dydt")

    ode_func_code = "\n".join(ode_func_lines)

    # ODE event functions
    ode_event_lines = []
    
    ep_val = "self.constants['ep']"
    t_idx = ode_state_map.get('t')
    if t_idx is not None:
        ode_event_lines.append(textwrap.dedent(f"""
    def event_t_max(t_val, y_array):
        # Event: t exceeds ep
        # Returns 0 when t == ep
        return {ep_val} - y_array[{t_idx}]
    event_t_max.terminal = True
    event_t_max.direction = -1 # Trigger when ep - t goes from positive to negative (t > ep)
    """).strip())

    v_idx = ode_state_map.get('v')
    if v_idx is not None:
        if ode_event_lines: ode_event_lines.append("") # Add a blank line between events
        ode_event_lines.append(textwrap.dedent(f"""
    def event_v_min(t_val, y_array):
        # Event: v becomes negative
        # Returns 0 when v == 0
        return y_array[{v_idx}]
    event_v_min.terminal = True
    event_v_min.direction = -1 # Trigger when v goes from positive to negative (v < 0)
    """).strip())

    if ode_event_lines: ode_event_lines.append("") # Add a blank line before ode_events list
    ode_event_lines.append("ode_events = [")
    if t_idx is not None:
        ode_event_lines.append("    event_t_max,")
    if v_idx is not None:
        ode_event_lines.append("    event_v_min,")
    ode_event_lines.append("]")

    ode_event_code = "\n".join(ode_event_lines)

    return ode_func_code, ode_event_code, ode_state_map, ode_state_idx_map
```

*   **Purpose**: To generate the Python code strings for the ODE function (`f(t, y)`) and event functions required by `scipy.integrate.solve_ivp`.
*   **Mechanism**:
    *   It finds the `StepFuncConfig` which contains the ODE definitions.
    *   It creates `ode_state_map` (variable name to index) and `ode_state_idx_map` (index to variable name) to manage the mapping between named variables and their positions in the `y_array` used by the ODE solver.
    *   **ODE Function (`ode_func`)**:
        *   It generates a Python function `def ode_func(t_val, y_array):`.
        *   Inside this function, it maps the `y_array` elements back to named variables (e.g., `x = y_array[0]`, `v = y_array[2]`).
        *   **Crucially**, it updates `self.state` with the current values from `y_array`. This ensures that any expressions evaluated by `self._evaluate_expression` within the ODEs use the *current* state of the ODE solver, not the state *before* integration began.
        *   It initializes a `derivatives` list with `'0.0'` for all state variables.
        *   Then, for each `ODEConfig` in `step_func_config.odes`, it replaces the default `'0.0'` with `self._evaluate_expression('{equation}')` for the corresponding variable.
        *   Finally, it constructs the `dydt` list and returns it.
    *   **ODE Event Functions (`event_t_max`, `event_v_min`)**:
        *   It generates functions for specific events defined in the `EvolutionDomainConfig` (e.g., `t <= ep`, `v >= 0`).
        *   `event_t_max`: Returns `ep - t`. It's `terminal=True` (stops integration) and `direction=-1` (triggers when `ep - t` goes from positive to negative, meaning `t` exceeds `ep`).
        *   `event_v_min`: Returns `v`. It's `terminal=True` and `direction=-1` (triggers when `v` goes from positive to negative, meaning `v` drops below zero).
        *   These events are collected into an `ode_events` list.
    *   It returns the `ode_func_code` string, `ode_event_code` string, and the state mappings.

#### `_generate_reset_logic`

```python
def _generate_reset_logic(variables: List[VariableDeclarationConfig], precondition: PreconditionConfig):
    """
    Generates the initial state assignment logic for the reset method.
    It uses default values and then checks the precondition.
    """
    reset_code_lines = []
    reset_code_lines.append("    # Default initial values (can be randomized or fixed based on problem)")
    
    defaults = {
        'x': '0.0', 'y': '0.0', 'v': '1.0', 'a': '0.0',
        'dx': '1.0', 'dy': '0.0', 'w': '0.0', 'r': '1.0',
        'xo': '5.0', 'yo': '0.0', 't': '0.0'
    }

    for var in variables:
        default_val = defaults.get(var.name, '0.0')
        reset_code_lines.append(f"self.state['{var.name}'] = {default_val}")
    
    return reset_code_lines
```

*   **Purpose**: To generate the Python code lines that initialize the state variables in the `reset()` method of the Gymnasium environment.
*   **Mechanism**:
    *   It iterates through the `VariableDeclarationConfig` objects.
    *   For each variable, it assigns a default value from the `defaults` dictionary (e.g., `x=0.0`, `v=1.0`, `dx=1.0`, `dy=0.0`, `xo=5.0`). If a variable is not in `defaults`, it defaults to `'0.0'`.
    *   These assignments are formatted as `self.state['variable_name'] = value` and returned as a list of strings.

---

### 6. `if __name__ == "__main__":` Block (Testing and Example)

```python
if __name__ == "__main__":
    # This part is for testing the generator directly with the provided PyAST config
    # The PyAST config is hardcoded here for demonstration, but in real use
    # it would come from keymaerax_parser.py
    
    # Create a dummy GymEnvironmentConfig based on the example PyAST structure
    # This mimics the output of convert_keymaerax_ast_to_intermediate_pyast
    
    # Constants
    ep_const = ConstantDefinitionConfig(name='ep', type='Real')
    b_const = ConstantDefinitionConfig(name='b', type='Real')
    A_const = ConstantDefinitionConfig(name='A', type='Real')
    W_const = ConstantDefinitionConfig(name='W', type='Real')

    # Functions
    stopDist_func = FunctionDefinitionConfig(name='stopDist', return_type='Real', parameters=[('Real', 'v')], body='v^2 / (2*b)')
    accelComp_func = FunctionDefinitionConfig(name='accelComp', return_type='Real', parameters=[('Real', 'v')], body='(A/b + 1) * (A/2 * ep^2 + ep*v)')
    admissibleSeparation_func = FunctionDefinitionConfig(name='admissibleSeparation', return_type='Real', parameters=[('Real', 'v')], body='stopDist(v) + accelComp(v)')

    # Predicates
    isWellformedDir_pred = PredicateDefinitionConfig(name='isWellformedDir', parameters=[('Real', 'dx'), ('Real', 'dy')], body='dx^2 + dy^2 = 1')
    bounds_pred = PredicateDefinitionConfig(name='bounds', parameters=[], body='A >= 0 & b > 0 & ep > 0')

    # controlFunc Action Config
    # Choice 1: { a := -b; }
    choice1_actions = SequenceActionConfig(actions=[
        DeterministicAssignmentActionConfig(variable='a', value='-b')
    ])
    # Choice 2: { ?v = 0; a := 0; w := 0; }
    choice2_actions = SequenceActionConfig(actions=[
        ActionConstraintConfig(formula='?v = 0'),
        DeterministicAssignmentActionConfig(variable='a', value='0'),
        DeterministicAssignmentActionConfig(variable='w', value='0')
    ])
    # Choice 3: { a := A; w := *; ?-W<=w & w<=W; r := *; xo := *; yo := *; ?r!=0 & r*w = v; ? abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v); }
    choice3_actions = SequenceActionConfig(actions=[
        DeterministicAssignmentActionConfig(variable='a', value='A'),
        NondeterministicAssignmentActionConfig(variable='w'),
        ActionConstraintConfig(formula='-W<=w & w<=W'),
        NondeterministicAssignmentActionConfig(variable='r'),
        NondeterministicAssignmentActionConfig(variable='xo'),
        NondeterministicAssignmentActionConfig(variable='yo'),
        ActionConstraintConfig(formula='r!=0 & r*w = v'),
        ActionConstraintConfig(formula='abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v)')
    ])
    control_func_action_config = ChoiceActionConfig(level_name='ChoiceLevel_0', choices=[
        choice1_actions, choice2_actions, choice3_actions
    ])
    control_func = ControlFuncConfig(name='controlFunc', action_space_config=control_func_action_config)

    # stepFunc ODEs and Evolution Domain
    step_func_odes = [
        ODEConfig(variable='x', equation='v * dx'),
        ODEConfig(variable='y', equation='v * dy'),
        ODEConfig(variable='v', equation='a'),
        ODEConfig(variable='dx', equation='-w * dy'),
        ODEConfig(variable='dy', equation='w * dx'),
        ODEConfig(variable='w', equation='a/r'),
        ODEConfig(variable='t', equation='1')
    ]
    step_func_evolution_domain = EvolutionDomainConfig(constraints=['t <= ep', 'v >= 0'])
    step_func = StepFuncConfig(name='stepFunc', odes=step_func_odes, evolution_domain=step_func_evolution_domain, duration='ep')

    # DefinitionsConfig
    definitions_config = DefinitionsConfig(
        constants=[ep_const, b_const, A_const, W_const],
        functions=[stopDist_func, accelComp_func, admissibleSeparation_func],
        predicates=[isWellformedDir_pred, bounds_pred],
        hybrid_programs=[control_func, step_func]
    )

    # VariablesConfig
    state_vars = [
        VariableDeclarationConfig(name='x', type='Real'),
        VariableDeclarationConfig(name='y', type='Real'),
        VariableDeclarationConfig(name='v', type='Real'),
        VariableDeclarationConfig(name='a', type='Real'),
        VariableDeclarationConfig(name='dx', type='Real'),
        VariableDeclarationConfig(name='dy', type='Real'),
        VariableDeclarationConfig(name='w', type='Real'),
        VariableDeclarationConfig(name='r', type='Real'),
        VariableDeclarationConfig(name='xo', type='Real'),
        VariableDeclarationConfig(name='yo', type='Real'),
        VariableDeclarationConfig(name='t', type='Real')
    ]
    variables_config = VariablesConfig(state_variables=state_vars)

    # ProblemConfig
    precondition = PreconditionConfig(formula='assumptions(x, y, v, dx, dy, xo, yo)')
    postcondition = PostconditionConfig(formula='(x - xo)^2 + (y - yo)^2 > 0')
    
    # Loop body: { { controlFunc; t := 0; } { stepFunc;@invariant(t>=0, isWellformedDir(dx, dy)) } }
    loop_body_actions = SequenceActionConfig(actions=[
        SequenceActionConfig(actions=[
            CallHybridProgramConfig(hp_name='controlFunc'),
            DeterministicAssignmentActionConfig(variable='t', value='0')
        ]),
        HybridProgramStepConfig(hp_name='stepFunc', invariant_formula='t>=0 & isWellformedDir(dx, dy)')
    ])
    loop_execution = LoopExecutionConfig(
        iterations='inf',
        loop_invariant_formula='loopinv(x, y, v, dx, dy, xo, yo)',
        loop_body=loop_body_actions
    )
    problem_config = ProblemConfig(
        precondition=precondition,
        loop_execution=loop_execution,
        postcondition=postcondition
    )

    # Full GymEnvironmentConfig
    gym_config_example = GymEnvironmentConfig(
        env_name='IJRR17_Theorem_1_Static_safety',
        definitions=definitions_config,
        variables=variables_config,
        problem=problem_config
    )

    # Generate the code
    generated_code = generate_gym_environment_code(gym_config_example)

    # Print the generated code
    print("--- Generated Gymnasium Environment Code ---")
    print(generated_code)

    # Optionally, save to a file
    output_filename = "generated_kx_env.py"
    with open(output_filename, "w") as f:
        f.write(generated_code)
    print(f"\nGenerated Gymnasium environment code saved to {output_filename}")

    print("\n--- Example Usage Instructions ---")
    print(f"""
To use the generated environment:

1.  Make sure `gymnasium`, `numpy`, `scipy`, and `re` are installed (`pip install gymnasium numpy scipy`).
2.  Save the generated code above as `{output_filename}`.
3.  Ensure `ast_config.py` is in the same directory.
4.  You can then import and use the generated environment in another Python script or interactive session:

```python
import gymnasium as gym
from generated_kx_env import IJRR17_Theorem_1_Static_safety 
import numpy as np

# Instantiate the environment
env = IJRR17_Theorem_1_Static_safety()

observation, info = env.reset()
print("Initial observation:", observation)
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

# Before stepping, ensure constants like W are set in the environment if needed
# For this example, constants like W, A, b, ep are initialized to 0.0 if not given a value
# in the KeYmaera X input. You might want to set them explicitly in your env_config.
env.constants['W'] = 1.0 # Example: Set W to 1.0
env.constants['A'] = 1.0 # Example: Set A to 1.0
env.constants['b'] = 0.5 # Example: Set b to 0.5
env.constants['ep'] = 1.0 # Example: Set ep to 1.0

# Example 1: Choose the first branch: {{ a := -b; }} (index 0)
# Note: You must provide values for all Box spaces in the Dict action space,
# even if the chosen branch doesn't use them. Dummy values (e.g., 0.0) are fine.
action_branch_0 = {{
    'choice_ChoiceLevel_0': 0, # Select the first branch
    'assign_w': 0.0,           # Dummy value for unused nondeterministic assignments
    'assign_r': 0.0,
    'assign_xo': 0.0,
    'assign_yo': 0.0,
}}

print("\\nTaking 5 steps with action_branch_0 (a := -b;):")
current_obs = observation
for i in range(5):
    print(f"Step {{i+1}} (before action): State={{env.state}}")
    observation, reward, terminated, truncated, info = env.step(action_branch_0)
    print(f"Step {{i+1}}: Obs={{observation}}, Reward={{reward:.2f}}, Terminated={{terminated}}, Truncated={{truncated}}")
    if terminated or truncated:
        print("Episode finished.")
        break
env.close()

# Example 2: Reset and choose the third branch: {{ a := A; w := *; ... }} (index 2)
# This branch requires specific values for w, r, xo, yo.
print("\\nResetting environment and taking 5 steps with action_branch_2 (a := A; w := *; etc.):")
env = IJRR17_Theorem_1_Static_safety() # Re-instantiate for a fresh start
observation, info = env.reset()

# Re-set constants after reset
env.constants['W'] = 1.0
env.constants['A'] = 1.0
env.constants['b'] = 0.5
env.constants['ep'] = 1.0

# Define action_branch_2
# It's crucial that r != 0 and r*w = v for the constraint to pass.
# Let's assume v is initially 1.0 (from reset).
# If w = 0.1, then r must be 1.0 / 0.1 = 10.0
action_branch_2 = {{
    'choice_ChoiceLevel_0': 2, # Select the third branch
    'assign_w': 0.1,           # Example value for w (within -W to W, i.e., -1.0 to 1.0)
    'assign_r': 10.0,          # Example value for r (r!=0 and r*w = v, so 10.0 * 0.1 = 1.0)
    'assign_xo': 5.0,          # Example value for xo
    'assign_yo': 0.0,          # Example value for yo
}}

for i in range(5):
    print(f"Step {{i+1}} (before action): State={{env.state}}")
    observation, reward, terminated, truncated, info = env.step(action_branch_2)
    print(f"Step {{i+1}}: Obs={{observation}}, Reward={{reward:.2f}}, Terminated={{terminated}}, Truncated={{truncated}}")
    if terminated or truncated:
        print("Episode finished.")
        break
env.close()
```

*   **Purpose**: This block serves as a demonstration and testing ground for the code generator.
*   **Mechanism**:
    *   It manually constructs a `GymEnvironmentConfig` object. This object represents a specific hybrid system model (named "IJRR17_Theorem_1_Static_safety" in this example) using the AST classes imported from `ast_config.py`.
    *   It defines constants, functions, predicates, the `controlFunc`'s action structure (a complex nested `ChoiceActionConfig` with `SequenceActionConfig`, `DeterministicAssignmentActionConfig`, `NondeterministicAssignmentActionConfig`, and `ActionConstraintConfig`), `stepFunc`'s ODEs and evolution domain, state variables, and problem details (precondition, postcondition, loop execution).
    *   It then calls `generate_gym_environment_code()` with this meticulously constructed AST.
    *   The `generated_code` string is printed to the console and saved to a file named `generated_kx_env.py`.
    *   Finally, it provides detailed instructions and Python code snippets on how to use the generated environment, including how to instantiate it, set constants, and perform example `step()` calls with different actions. This part is crucial for understanding how the generated environment is expected to be interacted with.

---

In essence, this script automates the tedious and error-prone process of manually creating a Gymnasium environment for a hybrid system. By taking a structured, formal definition (the AST), it generates a runnable simulation, enabling the application of reinforcement learning techniques to analyze and control such systems.