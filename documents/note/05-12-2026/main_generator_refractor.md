Yes, there are several "code smells" in the provided code, particularly given its purpose as a *code generator*. The primary and most significant smell revolves around **code duplication and lack of modularity** in the generated environment.

Let's break down the code smells:

1.  **Massive Code Duplication / Embedded Helper Methods (Major Smell):**
    *   **Problem:** The methods `_expand_formula`, `_get_predicate_metadata`, `_apply_meta_effect`, `_evaluate_shape_expression`, `_kx_expr_to_z3_formula`, `_solve_initial_state_with_z3`, `_evaluate_expression`, `_execute_action_config`, `_apply_control_action`, and `_split_top_level_and_local` (along with `SPLIT_TOP_LEVEL_AND_LOCAL_METHOD_CODE`, `EVALUATE_EXPRESSION_METHOD_TEMPLATE`, etc.) are all *embedded directly as string literals* into every generated Gymnasium environment.
    *   **Why it's a smell:**
        *   **Maintenance Nightmare:** If there's a bug or an improvement needed in `_evaluate_expression` (or any of these helpers), you have to fix it in the *generator's source code* and then *re-generate every single environment* that uses it. If you have many generated environments, this becomes a huge burden.
        *   **Bloated Generated Code:** Each generated environment will contain hundreds of lines of identical helper code, making the generated files very large and harder to read/understand.
        *   **Poor Modularity:** These are common utilities. They should not be part of every single instance of a generated environment.
        *   **Testing Difficulty:** You can't easily test these helper methods in isolation within a generated environment without running the whole environment.
    *   **Suggested Improvement:**
        *   **Create a Base Class:** Move all these common helper methods into a `_BaseKeYmaeraXEnv` class (or similar) in a separate Python file (e.g., `kx_env_base.py`).
        *   **Inheritance:** The generated environment class (`GeneratedKXEnv`) should then inherit from this base class.
        *   **Import:** The generated code would only need a single `from kx_env_base import _BaseKeYmaeraXEnv` and `class GeneratedKXEnv(_BaseKeYmaeraXEnv):`. This makes the generated files much smaller, easier to maintain, and more modular.

2.  **Magic String for Precondition Conjuncts in `reset` (Minor Smell):**
    *   **Problem:** In the `reset` method, the line `precondition_conjuncts = self._split_top_level_and_local('((A >= 0 & b > 0 & ep > 0) & (v = 0 & (x-xo)^2 + (y-yo)^2 > 0 & (dx^2 + dy^2 = 1) & r != 0))')` uses a hardcoded string.
    *   **Why it's a smell:** This string is a direct copy of the `assumptions` predicate's expanded body. If the `assumptions` predicate (or its dependencies like `bounds` or `initialState`) changes in the `_definitions_config`, this hardcoded string will become outdated and lead to incorrect behavior without a corresponding manual update.
    *   **Suggested Improvement:** It should dynamically retrieve and expand the precondition formula from `self._problem_config`:
        ```python
        expanded_precondition_formula = self._expand_formula(self._problem_config.precondition.formula, self._definitions_config)
        precondition_conjuncts = self._split_top_level_and_local(expanded_precondition_formula)
        ```

3.  **Simplistic Regex for `_expand_formula` Argument Parsing (Minor Smell / Known Limitation):**
    *   **Problem:** The comment in `_expand_formula` explicitly states: "This split is simplistic and might not correctly parse arguments if they contain commas within nested expressions (e.g., f(b,c)). A more robust solution would involve a mini-parser for arguments."
    *   **Why it's a smell:** This indicates a known potential point of failure or incorrect behavior for more complex KeYmaera X expressions. It's a technical debt.
    *   **Suggested Improvement:** If the KeYmaera X grammar allows complex arguments, this part should ideally be replaced with an AST-based argument parser (similar to how `pyparsing` handles `func_call_args`). For simpler cases, it might be acceptable, but the warning should be heeded.

4.  **Verbose `envName` Sanitization (Minor Aesthetic Smell):**
    *   **Problem:** The logic for sanitizing `envName` at the beginning of `generate_gym_environment_code` is a bit long and repetitive.
    *   **Why it's a smell:** While functional, it clutters the main generation function.
    *   **Suggested Improvement:** Extract this into a small helper function, e.g., `_sanitize_class_name(original_name)`.

5.  **`eval()` Usage (Contextual Smell):**
    *   **Problem:** `eval()` is used extensively for dynamic lambda creation and expression evaluation.
    *   **Why it's a smell:** In general, `eval()` can be a security risk if the input strings are not fully trusted, and it can sometimes have performance overhead compared to pre-compiled code.
    *   **Mitigation/Context:** In a code *generation* context where the inputs are derived from a structured AST (which `format_kx_expr_for_python_lambda_body_generator` is designed to handle), the security risks are significantly reduced, as the generated Python code is controlled. The performance impact might be acceptable for Gym environments, but it's always something to be aware of. The current approach with `format_kx_expr_for_python_lambda_body_generator` is the correct way to make `eval` safer and more robust in this scenario.

**In summary, the most critical improvement would be to refactor the common helper methods into a base class that the generated environments inherit from.** This would address the core issues of maintainability, code bloat, and modularity. The other points are more minor but would further enhance the quality of the generated code.

---

You're right, there are definite code smells, primarily the extensive duplication of helper methods within each generated environment. Let's clean this up by extracting all common utility methods into a separate base class.

This approach will:
1.  **Improve Modularity:** The generated environment will be much smaller and focused on its specific configuration.
2.  **Enhance Maintainability:** Fixes or improvements to the helper methods only need to be made in one place (the base class).
3.  **Reduce Code Bloat:** Each generated environment file will be significantly smaller.

Here's how we'll refactor it:

**1. Create a Base Class File (`kx_env_base.py`)**

This file will contain the `_KeYmaeraXEnvBase` class, which all generated environments will inherit from. It will house all the shared logic, including:
*   Imports for `numpy`, `scipy.integrate`, `math`, `re`, `z3`, and `pyast_config` types.
*   The `__init__` method to set up common attributes like `equality_tolerance`, and to deserialize the `GymEnvironmentConfig` object.
*   All the helper methods: `_compile_ode_func`, `_compile_ode_events`, `_get_obs`, `_get_info`, `_approx_eq`, `_expand_formula`, `_get_predicate_metadata`, `_apply_meta_effect`, `_evaluate_shape_expression`, `_kx_expr_to_z3_formula`, `_solve_initial_state_with_z3`, `_evaluate_expression`, `_execute_action_config`, `_apply_control_action`, and `_split_top_level_and_local`.

**2. Modify the `generate_gym_environment_code` Function**

This function will be updated to:
*   Import `_KeYmaeraXEnvBase` from `kx_env_base.py`.
*   Make the generated environment class inherit from `_KeYmaeraXEnvBase`.
*   Serialize the entire `GymEnvironmentConfig` object into a string literal, which will be passed to the base class's `__init__` for deserialization at runtime.
*   Remove all the embedded string literals for helper methods, as they are now inherited.
*   Extract the `envName` sanitization logic into a separate helper function.
*   Dynamically expand the precondition formula in `reset` instead of using a hardcoded string.

---

### `kx_env_base.py` (New File)

```python
import gymnasium as gym
import numpy as np
from scipy.integrate import solve_ivp
import math
import textwrap
import re
import z3

from typing import List, Dict, Any, Optional, Tuple, Union

# Import AST definitions - these are crucial for the base class to deserialize the config
from pyast_config import (
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

class _KeYmaeraXEnvBase(gym.Env):
    """
    Base class for Gymnasium environments generated from KeYmaera X configurations.
    Contains common helper methods for expression evaluation, ODE solving,
    action execution, and Z3 integration.
    """
    def __init__(self, config_serialized_str: str, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Reconstruct the config object from its serialized string representation
        # We pass a restricted global scope to eval for safety.
        # The local scope will contain the AST classes.
        ast_classes_for_eval = {
            cls.__name__: cls for cls in [
                GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig,
                ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig,
                HybridProgramConfig, ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig,
                VariableDeclarationConfig, PreconditionConfig, PostconditionConfig, LoopExecutionConfig,
                HybridProgramStepConfig, ActionType, ChoiceActionConfig, SequenceActionConfig,
                DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig,
                ActionConstraintConfig, CallHybridProgramConfig
            ]
        }
        # Add common types/literals for eval to work with serialized structures
        ast_classes_for_eval.update({
            'list': list, 'dict': dict, 'tuple': tuple, 'None': None, 'True': True, 'False': False
        })

        try:
            config: GymEnvironmentConfig = eval(config_serialized_str, {'__builtins__': None}, ast_classes_for_eval)
        except Exception as e:
            raise ValueError(f"Failed to deserialize GymEnvironmentConfig from string: {e}\\nString: {config_serialized_str}")

        # --- Store original AST config objects for Z3 and other runtime needs ---
        self._definitions_config = config.definitions
        self._variables_config = config.variables
        self._problem_config = config.problem

        # --- Semantic Gap Bridging: Floating Point Equality Tolerance ---
        self.equality_tolerance = 1e-9 # Default absolute tolerance for np.isclose

        # --- KeYmaera X Definitions ---
        # These will be populated by the derived class's __init__
        self.constants: Dict[str, float] = {}
        self.functions: Dict[str, Any] = {} # Storing lambdas
        self.predicates: Dict[str, Any] = {} # Storing lambdas

        # State variables and their mapping (populated by derived class)
        self._state_keys: List[str] = []
        self.state: Dict[str, float] = {}
        self.last_failing_precondition_formula: Optional[str] = None

        # ODE solver setup (populated by derived class)
        self._ode_state_map: Dict[str, int] = {}
        self._ode_state_idx_map: Dict[int, str] = {}
        self._ode_func_compiled = None
        self._ode_events_compiled = None

        # Action space (populated by derived class)
        self._action_space_keys: List[str] = []
        self._control_func_action_config: Optional[ActionType] = None

        # Episode Management (can be set in base or derived)
        self.current_step = 0
        self.max_episode_steps = 200 # Default max steps for an episode
        self.reward_range = (-float('inf'), float('inf'))

    def _compile_ode_func(self, func_str):
        """Compiles the ODE function string into a callable Python function."""
        exec_globals = {'np': np, 'math': math, 'self': self}
        exec_locals = {}
        exec(textwrap.dedent(func_str), exec_globals, exec_locals)
        return exec_locals['ode_func']

    def _compile_ode_events(self, events_str):
        """Compiles the ODE event function strings into callable Python functions."""
        if not events_str.strip():
            return []
        exec_globals = {'np': np, 'math': math, 'self': self}
        exec_locals = {}
        exec(textwrap.dedent(events_str), exec_globals, exec_locals)
        return exec_locals['ode_events']

    def _get_obs(self):
        """Returns the current observation array from the state dictionary."""
        return np.array([self.state[key] for key in self._state_keys], dtype=np.float32)

    def _get_info(self):
        """Returns a copy of the current state dictionary as info."""
        return self.state.copy()

    def _approx_eq(self, a, b):
        """Performs an approximate equality check using numpy.isclose."""
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

    def _get_predicate_metadata(self, predicate_name: str) -> Dict[str, Any]:
        """Retrieves metadata for a given predicate name from the definitions."""
        for pred_config in self._definitions_config.predicates:
            if pred_config.name == predicate_name:
                return pred_config.metadata
        return {} # Return empty dict if not found

    def _apply_meta_effect(self, metadata: Dict[str, Any], current_reward: float, current_terminated: bool, current_truncated: bool, is_failure_case: bool) -> Tuple[float, bool, bool]:
        """
        Applies reward/penalty/termination/truncation effects based on metadata.
        'on_success': 'action:status=value' (e.g., 'terminate:status=success', 'continue:status=none')
        'on_failure': 'action:status=value' (e.g., 'truncate:status=failure', 'terminate:status=failure')

        action can be 'continue', 'terminate', 'truncate'.
        status can be 'success', 'failure', 'none'.

        is_failure_case: True if this call is due to a condition failing, False if it's a success path.
        """
        reward_change = 0.0
        new_terminated = current_terminated
        new_truncated = current_truncated

        if metadata:
            event_key = 'on_failure' if is_failure_case else 'on_success'
            event_config_str = metadata.get(event_key)

            if is_failure_case:
                penalty = float(metadata.get('penalty', 0.0))
                reward_change += penalty
            else: # Success case
                success_reward = float(metadata.get('reward', 0.0))
                reward_change += success_reward

            if event_config_str:
                parts = event_config_str.split(':')
                action = parts[0]
                # status_part = parts[1] if len(parts) > 1 else 'status=none'
                # status = status_part.split('=')[1] # Not directly used for logic, but could be for info

                if action == 'terminate':
                    new_terminated = True
                elif action == 'truncate':
                    new_truncated = True
                # 'continue' means no change to terminated/truncated state, so no 'else' needed here.

        return current_reward + reward_change, new_terminated, new_truncated

    def _evaluate_shape_expression(self, shape_expr: str) -> float:
        """
        Evaluates a shape expression from metadata.
        Shape expressions might use ternary operators (e.g., 'v > 0 ? 1/v : v').
        """
        if not shape_expr:
            return 0.0

        # Handle ternary operator 'condition ? value_if_true : value_if_false'
        # This is a simple regex-based parser, might need more robustness for complex cases.
        match = re.match(r'(.+)\s*\?\s*(.+)\s*:\s*(.+)', shape_expr)
        if match:
            condition_str, true_val_str, false_val_str = match.groups()
            try:
                condition_result = self._evaluate_expression(condition_str)
                if condition_result:
                    return float(self._evaluate_expression(true_val_str))
                else:
                    return float(self._evaluate_expression(false_val_str))
            except Exception as e:
                print(f"Warning: Error evaluating shape ternary expression '{shape_expr}': {e}")
                return 0.0
        else:
            # If not a ternary, just evaluate as a regular expression
            try:
                return float(self._evaluate_expression(shape_expr))
            except Exception as e:
                print(f"Warning: Error evaluating shape expression '{shape_expr}': {e}")
                return 0.0

    def _kx_expr_to_z3_formula(self, expr_str: str, solver_vars: Dict[str, Any]) -> Any:
        """
        Converts a KeYmaera X expression string into a Z3 formula.
        It handles basic arithmetic, comparisons, logical operators, and variable substitution.
        It assumes that solver_vars contains Z3 symbols for all relevant variables.
        """
        # Replace KeYmaera X operators with Python/Z3 equivalents
        expr_str = expr_str.replace('==', '=') # Z3 uses '=' for equality
        expr_str = expr_str.replace('&&', 'and')
        expr_str = expr_str.replace('||', 'or')
        expr_str = expr_str.replace('!', 'not ')
        expr_str = expr_str.replace('^', '**') # Python power operator

        # Regex to find identifiers (variable names, function names, constants)
        # and replace them with their Z3 symbol or evaluated constant/function call
        def replace_identifier(match):
            ident = match.group(0)
            if ident in solver_vars:
                return f"solver_vars['{ident}']"
            elif ident in self.constants:
                return str(self.constants[ident]) # Z3 needs numeric literals
            elif ident in self.functions:
                # This case is tricky for Z3. Functions need to be re-represented as Z3 functions
                # or their bodies need to be expanded. For now, we assume functions are evaluated
                # in Python before being passed to Z3, or are simple enough to be part of the expression.
                print(f"Warning: KeYmaera X function '{ident}' encountered in Z3 conversion. Treating as variable.")
                return f"solver_vars['{ident}']" # Fallback
            else:
                return ident # Keep as is, might be a number or Z3 built-in

        # Use a regex to find standalone identifiers (not part of numbers or operators)
        expr_str = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', replace_identifier, expr_str)

        # Evaluate the string as a Z3 formula
        try:
            return eval(expr_str, {'__builtins__': None, 'z3': z3}, solver_vars)
        except Exception as e:
            print(f"Error converting KeYmaera X expression '{expr_str}' to Z3 formula: {e}")
            raise

    def _solve_initial_state_with_z3(self, state_variables: List[VariableDeclarationConfig], precondition_formula: str) -> Optional[Dict[str, float]]:
        """
        Uses Z3 to find a satisfying assignment for initial state variables
        given a precondition formula.
        """
        s = z3.Solver()
        solver_vars = {}
        for var_decl in state_variables:
            solver_vars[var_decl.name] = z3.Real(var_decl.name)

        # Convert the precondition formula to Z3
        try:
            z3_precondition = self._kx_expr_to_z3_formula(precondition_formula, solver_vars)
            s.add(z3_precondition)
        except Exception as e:
            print(f"Failed to convert precondition to Z3: {e}")
            self.last_failing_precondition_formula = precondition_formula
            return None

        if s.check() == z3.sat:
            m = s.model()
            initial_state = {}
            for var_decl in state_variables:
                val = m.evaluate(solver_vars[var_decl.name])
                initial_state[var_decl.name] = float(val.as_decimal(10)) # Convert Z3 rational to float
            return initial_state
        else:
            self.last_failing_precondition_formula = precondition_formula
            return None

    def _evaluate_expression(self, expr_str: str, is_constraint: bool = False) -> Union[float, bool]:
        """
        Evaluates a KeYmaera X expression string in the current environment's context.
        Handles state variables, constants, functions, and predicates.
        """
        # Local state and constants/functions for evaluation context
        eval_context = {
            **self.state,
            **self.constants,
            **self.functions,
            **self.predicates,
            'np': np,
            'math': math,
            'self': self, # Allow access to self for methods like _approx_eq
            '_approx_eq': self._approx_eq # Explicitly expose helper
        }

        # Replace KeYmaera X specific operators with Python equivalents
        # Note: Direct string replacement for '==' to '_approx_eq' is complex
        # and prone to errors without full AST parsing. For now, standard Python
        # operators are used, and '_approx_eq' must be explicitly called if needed.
        expr_str = expr_str.replace('&&', 'and')
        expr_str = expr_str.replace('||', 'or')
        expr_str = expr_str.replace('!', 'not ')
        expr_str = expr_str.replace('^', '**') # Python power operator

        try:
            # Evaluate the expression using the current state and definitions
            result = eval(expr_str, {'__builtins__': None}, eval_context)
            return result
        except NameError as ne:
            # This could happen if a variable or function is not defined in the context
            print(f"NameError: {ne} in expression '{expr_str}'. Available context keys: {list(eval_context.keys())}")
            raise
        except Exception as e:
            print(f"Error evaluating expression '{expr_str}': {e}")
            raise

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
                print(f"Error: Choice key '{choice_key}' missing from agent action.")
                reward -= 200.0
                terminated = True
                applied_successfully = False
                return reward, terminated, applied_successfully

            choice_idx = agent_action[choice_key]
            if not (0 <= choice_idx < len(action_config.choices)):
                print(f"Error: Invalid choice index {choice_idx} for '{choice_key}'.")
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
                print(f"Error evaluating deterministic assignment for {action_config.variable}='{action_config.value}': {e}")
                reward -= 100.0
                terminated = True
                applied_successfully = False
            return reward, terminated, applied_successfully

        elif isinstance(action_config, NondeterministicAssignmentActionConfig):
            assign_key = f"assign_{action_config.variable}"
            if assign_key in agent_action:
                self.state[action_config.variable] = agent_action[assign_key]
            else:
                print(f"Warning: Nondeterministic assignment for '{action_config.variable}' (key '{assign_key}') not found in agent action. Defaulting to current value.")
                # This might indicate an incomplete action by the agent, penalize.
                reward -= 50.0
                # Do not terminate here, as it might be a valid choice to not assign.
                # However, if the problem requires it, this might need to be `terminated = True`.
                # For now, we assume it's okay to skip, but with a penalty.
            return reward, terminated, applied_successfully

        elif isinstance(action_config, ActionConstraintConfig):
            constraint_met = self._evaluate_expression(action_config.formula, is_constraint=True)

            # Try to find metadata for this constraint if it's a predicate call
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', action_config.formula)
            if predicate_name_match:
                pred_name = predicate_name_match.group(1)
                metadata = self._get_predicate_metadata(pred_name)
            else:
                # Default metadata for generic constraints
                metadata = {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'}

            if not constraint_met:
                reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, False, is_failure_case=True)
                # If metadata didn't specify termination/truncation, and it's not 'continue', then it's a failure to apply
                if not terminated and not truncated and metadata.get('on_failure', '').split(':')[0] == 'continue':
                    applied_successfully = True # Constraint failed but we are told to continue
                else:
                    applied_successfully = False # Constraint failed and we should terminate/truncate
            else:
                # If constraint is met, apply 'on_success' effects if any
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, False, is_failure_case=False)
                applied_successfully = True # Constraint met, so action was successfully applied

            return reward, terminated, applied_successfully

        elif isinstance(action_config, CallHybridProgramConfig):
            # This generator assumes controlFunc is the primary action selection.
            # If CallHybridProgramConfig appears here, it's an unsupported nested HP call within controlFunc.
            print(f"Warning: CallHybridProgramConfig '{action_config.hp_name}' encountered in _execute_action_config. Not fully supported yet.")
            reward -= 50.0
            applied_successfully = False
            return reward, terminated, applied_successfully

        else:
            print(f"Error: Unknown action config type: {type(action_config)}")
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

    def _split_top_level_and_local(self, formula: str) -> List[str]:
        """
        This function aims to split a KeYmaera X formula into top-level conjuncts (A && B && C)
        while respecting parentheses. It does not parse the full grammar, but handles
        the common case of logical ANDs at the top level.
        """
        conjuncts = []
        balance = 0
        last_split_idx = 0

        i = 0
        while i < len(formula):
            if formula[i] == '(':
                balance += 1
            elif formula[i] == ')':
                balance -= 1
            elif formula[i:i+2] == '&&' and balance == 0:
                # Found a top-level '&&'
                conjunct = formula[last_split_idx:i].strip()
                if conjunct:
                    conjuncts.append(conjunct)
                last_split_idx = i + 2
                i += 1 # Skip the second '&'
            i += 1

        # Add the last conjunct
        final_conjunct = formula[last_split_idx:].strip()
        if final_conjunct:
            conjuncts.append(final_conjunct)

        # Clean up any outer parentheses that might have been added by _expand_formula
        cleaned_conjuncts = []
        for conj in conjuncts:
            if conj.startswith('(') and conj.endswith(')'):
                # Check if it's truly an outer parenthesis or part of the expression
                try:
                    temp_balance = 0
                    is_outer = True
                    for char_idx in range(1, len(conj) - 1):
                        if conj[char_idx] == '(':
                            temp_balance += 1
                        elif conj[char_idx] == ')':
                            temp_balance -= 1
                        if temp_balance < 0: # Unbalanced inside, so outer parens are not just for grouping
                            is_outer = False
                            break
                    if is_outer and temp_balance == 0:
                        cleaned_conjuncts.append(conj[1:-1].strip())
                    else:
                        cleaned_conjuncts.append(conj)
                except Exception:
                    cleaned_conjuncts.append(conj) # Fallback
            else:
                cleaned_conjuncts.append(conj)

        return [c for c in cleaned_conjuncts if c] # Filter out empty strings
```

---

### `your_generator_file.py` (Modified Original File)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import textwrap
import re # For regex replacements

from typing import List, Dict, Any, Optional, Tuple, Union

# Import AST definitions
from pyast_config import (
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

# Import helper functions from other generator modules
from gym_generator.ast_serialization import serialize_ast_node
from gym_generator.expression_helpers import format_kx_expr_for_python_lambda_body_generator
from gym_generator.gym_logic_sub_generators import generate_action_space_and_logic, generate_ode_solver_logic, generate_reset_logic

# --- NEW: Import the base class ---
from kx_env_base import _KeYmaeraXEnvBase # Assuming kx_env_base.py is in the same directory or importable path

# The embedded_methods are no longer needed as string literals,
# they are now actual methods in _KeYmaeraXEnvBase.
# from gym_generator.embedded_methods import (
#     SPLIT_TOP_LEVEL_AND_LOCAL_METHOD_CODE,
#     EVALUATE_EXPRESSION_METHOD_TEMPLATE,
#     KX_EXPR_TO_Z3_FORMULA_METHOD_CODE,
#     SOLVE_INITIAL_STATE_WITH_Z3_METHOD_CODE
# )


def _sanitize_class_name(original_name: str) -> str:
    """
    Sanitizes a string to be a valid Python class identifier.
    """
    sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', original_name)
    sanitized_name = re.sub(r'_+', '_', sanitized_name).strip('_')
    if sanitized_name and sanitized_name[0].isdigit():
        sanitized_name = '_' + sanitized_name
    if not sanitized_name:
        return "GeneratedEnv"
    return sanitized_name


def generate_gym_environment_code(config: GymEnvironmentConfig, envName: str = "") -> str:
    """
    Generates Python code for a Gymnasium environment based on the provided
    GymEnvironmentConfig AST.
    """
    if not envName:
        env_name_for_class = _sanitize_class_name(config.env_name)
    else:
        env_name_for_class = envName

    # Collect all variable names for observation space and ODE state vector
    state_variable_names = [v.name for v in config.variables.state_variables]
    all_constant_names = [c.name for c in config.definitions.constants]
    all_function_names = [f.name for f in config.definitions.functions]
    all_predicate_names = [p.name for p in config.definitions.predicates]

    # --- Generate Constants, Functions, Predicates ---
    constants_init_code_lines = []
    for const in config.definitions.constants:
        # Handle cases where value is None or an empty string, defaulting to '0.0'
        if const.value is None or const.value == '':
            formatted_value = '0.0' # Default value if not specified, Z3 will override in reset().
        # Check if the value is a simple numeric literal (integer or float) or boolean
        elif re.fullmatch(r'^-?\d+(\.\d+)?$', const.value) or re.fullmatch(r'^(True|False)$', const.value):
            formatted_value = const.value
        else: # It's an expression involving other constants/variables, so format it for lambda body
            formatted_value = format_kx_expr_for_python_lambda_body_generator(
                const.value, [], all_constant_names, state_variable_names, all_function_names, all_predicate_names
            )
        constants_init_code_lines.append(f"self.constants['{const.name}'] = {formatted_value}")
    # Pre-indent this block
    indented_constants_block = textwrap.indent("\n".join(constants_init_code_lines), '        ')

    functions_init_code_lines = []
    for func in config.definitions.functions:
        params_str = ", ".join([p_name for _, p_name in func.parameters])
        param_names_list = [p_name for _, p_name in func.parameters]
        # Ensure function body is a single line for embedding in string literal
        body_single_line = func.body.replace('\n', ' ').strip()

        # Format the body for direct Python lambda evaluation (generator-side parsing)
        formatted_body = format_kx_expr_for_python_lambda_body_generator(
            body_single_line, param_names_list, all_constant_names, state_variable_names, all_function_names, all_predicate_names
        )
        functions_init_code_lines.append(f"self.functions['{func.name}'] = lambda {params_str}: {formatted_body}")
    # Pre-indent this block
    indented_functions_block = textwrap.indent("\n".join(functions_init_code_lines), '        ')

    predicates_init_code_lines = []
    for pred in config.definitions.predicates:
        params_str = ", ".join([p_name for _, p_name in pred.parameters])
        param_names_list = [p_name for _, p_name in pred.parameters]
        # Ensure predicate body is a single line for embedding in string literal
        body_single_line = pred.body.replace('\n', ' ').strip()

        # Format the body for direct Python lambda evaluation (generator-side parsing)
        formatted_body = format_kx_expr_for_python_lambda_body_generator(
            body_single_line, param_names_list, all_constant_names, state_variable_names, all_function_names, all_predicate_names
        )
        predicates_init_code_lines.append(f"self.predicates['{pred.name}'] = lambda {params_str}: {formatted_body}")
    # Pre-indent this block
    indented_predicates_block = textwrap.indent("\n".join(predicates_init_code_lines), '        ')

    # --- Generate Action Space ---
    action_space_definition, control_func_ast_object, action_space_keys = generate_action_space_and_logic(config.definitions.hybrid_programs)

    # Convert the control_func_ast_object to its Python code string representation
    # This will embed raw KeYmaera X strings for action values/formulas
    control_func_ast_code_str = serialize_ast_node(
        control_func_ast_object, indent_level=3
    ) # Indent for __init__ method

    # --- Generate ODEs and Evolution Domain ---
    ode_func_code, ode_event_code, ode_state_map, ode_state_idx_map = generate_ode_solver_logic(
        config.definitions.hybrid_programs, state_variable_names,
        all_constant_names, all_function_names, all_predicate_names
    )

    # --- Generate Initial State (reset logic) ---
    # Pass definitions for predicate expansion
    reset_initial_state_code_lines = generate_reset_logic(config.variables.state_variables, config.problem.precondition, config.definitions)
    # Pre-indent this block
    indented_reset_block = textwrap.indent("\n".join(reset_initial_state_code_lines), '        ') # Adjusted indentation for reset body in derived class

    # Serialize the entire config object to pass to the base class's __init__
    # This string will be `eval`ed by the base class to reconstruct the config object.
    full_config_serialized_str = serialize_ast_node(config)

    # Imports for typing hints (now mostly in base class, but keep for clarity if needed)
    typing_imports = ["Dict", "Any", "Tuple", "List", "Optional", "Union"]
    typing_import_statement = f"from typing import ({', '.join(typing_imports)})"


    # --- Main Environment Class Structure ---
    env_code = textwrap.dedent(f"""import gymnasium as gym
from gymnasium import spaces
import numpy as np
# scipy.integrate, math, textwrap, re, z3 are now imported by the base class
# pyast_config is now imported by the base class
{typing_import_statement} # Import typing hints
from kx_env_base import _KeYmaeraXEnvBase # Import the base class

class {env_name_for_class}(_KeYmaeraXEnvBase): # Inherit from the base class
    metadata = {{"render_modes": ["human"], "render_fps": 30}}

    def __init__(self, render_mode=None):
        # Pass the serialized config string to the base class for initialization
        super().__init__(config_serialized_str='''{full_config_serialized_str}''', render_mode=render_mode)

        # --- KeYmaera X Definitions (Populated in derived class) ---
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
        # Initialize to 0.0 as placeholders; these will be overwritten by reset()
        self.state = {{key: 0.0 for key in self._state_keys}}
        # Initial state values are set in reset()
        # self.last_failing_precondition_formula is initialized in base class

        # --- Observation Space ---
        # All state variables are Real, so Box space is appropriate.
        # Bounds are infinite by default, as not specified in KeYmaera X input.
        low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)

        # --- Action Space (from controlFunc) ---
        # The action space is a Dict to handle choices and nondeterministic assignments.
{textwrap.indent(action_space_definition, '        ')}
        self._action_space_keys = {list(action_space_keys.keys())}
        self._control_func_action_config = {control_func_ast_code_str} # Stored for step logic

        # --- ODE Solver Setup (from stepFunc) ---
        # Mapping from state variable names to their index in the ODE solver's y array
        self._ode_state_map = {ode_state_map}
        self._ode_state_idx_map = {ode_state_idx_map} # Map index to name

        # The ODE function strings are now passed to the base class's compiler
        _ode_func_str_local = '''
{textwrap.indent(ode_func_code, '            ')}
        '''
        _ode_events_str_local = '''
{textwrap.indent(ode_event_code, '            ')}
        '''
        self._ode_func_compiled = self._compile_ode_func(_ode_func_str_local)
        self._ode_events_compiled = self._compile_ode_events(_ode_events_str_local)

        # Initialize the environment to a valid state using the reset method
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Call base class reset for common setup
        self.current_step = 0

        # --- Initial State (based on Problem Precondition and defaults) ---
        # Dynamically expand the precondition formula from the stored config
        expanded_precondition_formula = self._expand_formula(self._problem_config.precondition.formula, self._definitions_config)
{indented_reset_block}

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        # --- Initial Loop Invariant Check (before control/dynamics) ---
        loop_invariant_formula_kx = self._problem_config.loop_execution.loop_invariant_formula
        expanded_loop_inv = self._expand_formula(loop_invariant_formula_kx, self._definitions_config)
        loop_inv_conjuncts = self._split_top_level_and_local(expanded_loop_inv)

        loop_invariant_satisfied_pre_step = True
        for conj_str in loop_inv_conjuncts:
            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)
            predicate_name_match = re.match(r'^\\s*(\\w+)\\s*\\(', conj_str)
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {{}}

            if not conjunct_result:
                loop_invariant_satisfied_pre_step = False
                # Apply failure effects
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {{'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'}},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                # Apply success effects if any (only if action is not 'continue')
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                    if terminated or truncated:
                        return self._get_obs(), reward, terminated, truncated, info

        if not loop_invariant_satisfied_pre_step:
            # Fallback for general loopinv failure if not caught by specific conjuncts
            print(f"Warning: Loop invariant violated BEFORE step execution (general failure): {{loop_invariant_formula_kx}}")
            reward, terminated, truncated = self._apply_meta_effect(
                {{'type': 'safety', 'penalty': '-500.0', 'on_failure': 'terminate:status=failure'}},
                reward, terminated, truncated, is_failure_case=True
            )
            return self._get_obs(), reward, terminated, truncated, info # Terminate early


        # --- 1. Apply Control Function (Discrete Action) ---
        control_reward, control_terminated, control_applied = self._apply_control_action(action)
        reward += control_reward
        terminated = terminated or control_terminated

        if not control_applied:
            # Invalid action chosen or constraint violated during control phase
            return self._get_obs(), reward, terminated, truncated, info

        # --- 2. Simulate Differential Equations (Continuous Dynamics) ---
        # Prepare initial state for ODE solver
        y0 = np.array([self.state[var_name] for var_name in self._ode_state_map.keys()], dtype=np.float64)

        ep_val = self.constants['ep'] # Get the 'ep' constant value
        t_span = (0, ep_val) # Integrate up to time limit 'ep'

        ode_terminated_by_event = False

        if ep_val <= 0:
            # If ep is non-positive, no meaningful time evolution occurs.
            self.state['t'] = 0.0 # Time does not advance or is reset
            if ep_val < 0:
                print(f"Warning: 'ep' constant is negative ({{ep_val}}). Skipping ODE integration and penalizing.")
                # Link to isPhysicallyValid metadata if applicable, or a default
                phys_valid_meta = self._get_predicate_metadata('isPhysicallyValid')
                reward, terminated, truncated = self._apply_meta_effect(
                    phys_valid_meta or {{'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'}},
                    reward, terminated, truncated, is_failure_case=True
                )
                ode_terminated_by_event = True # Treat as terminated by event
        else:
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
                    if solution.y_events and solution.y_events[0].size > 0:
                        final_y = solution.y_events[0][0]
                    else:
                        final_y = solution.y[:, -1] # Fallback if event data is missing

                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]

                    # Check which event terminated it
                    # If v < 0, it's a domain violation -> termination
                    if self.state.get('v', 0.0) < -self.equality_tolerance: # Use tolerance for v < 0 check
                        phys_valid_meta = self._get_predicate_metadata('isPhysicallyValid')
                        reward, terminated, truncated = self._apply_meta_effect(
                            phys_valid_meta or {{'type': 'safety', 'penalty': '-100.0', 'on_failure': 'truncate:status=failure'}}, # Default to truncate as per problem description example
                            reward, terminated, truncated, is_failure_case=True
                        )
                        ode_terminated_by_event = True
                    # If t > ep, it means the time limit was reached, which is expected behavior.
                    # No additional penalty for that specific event, as it's part of the evolution domain.
                else: # Integration failed (status 2)
                    reward, terminated, truncated = self._apply_meta_effect(
                        {{'type': 'safety', 'penalty': '-150.0', 'on_failure': 'terminate:status=failure'}},
                        reward, terminated, truncated, is_failure_case=True
                    )
                    ode_terminated_by_event = True

            except Exception as e:
                reward, terminated, truncated = self._apply_meta_effect(
                    {{'type': 'safety', 'penalty': '-200.0', 'on_failure': 'terminate:status=failure'}},
                    reward, terminated, truncated, is_failure_case=True
                )
                ode_terminated_by_event = True

        terminated = terminated or ode_terminated_by_event
        truncated = truncated or ode_terminated_by_event # If ODE event truncates, propagate

        # If ODE integration or control action led to termination/truncation, return early
        if terminated or truncated:
            return self._get_obs(), reward, terminated, truncated, info

        # --- 3. Evaluate Loop Invariant and Apply Rewards/Penalties/Shaping ---
        # Re-evaluate loop invariant after ODEs
        loop_invariant_satisfied_post_ode = True
        for conj_str in loop_inv_conjuncts: # Reuse loop_inv_conjuncts from earlier
            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)
            predicate_name_match = re.match(r'^\\s*(\\w+)\\s*\\(', conj_str)
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {{}}

            if not conjunct_result:
                loop_invariant_satisfied_post_ode = False
                # Apply failure effects
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {{'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'}},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                # Apply success effects if any (only if action is not 'continue')
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                    if terminated or truncated:
                        return self._get_obs(), reward, terminated, truncated, info

        if not loop_invariant_satisfied_post_ode:
            # Fallback for general loopinv failure if not caught by specific conjuncts
            print(f"Warning: Loop invariant violated AFTER ODEs (general failure): {{loop_invariant_formula_kx}}")
            reward, terminated, truncated = self._apply_meta_effect(
                {{'type': 'safety', 'penalty': '-300.0', 'on_failure': 'terminate:status=failure'}},
                reward, terminated, truncated, is_failure_case=True
            )
            if terminated or truncated:
                return self._get_obs(), reward, terminated, truncated, info

        # --- Reward Shaping (from progress predicates) ---
        # isWellformedDir (progress type, shape: -((dx^2 + dy^2) - 1)^2)
        wellformed_meta = self._get_predicate_metadata('isWellformedDir')
        if wellformed_meta.get('type') == 'progress' and 'shape' in wellformed_meta:
            shape_value = self._evaluate_shape_expression(wellformed_meta['shape'])
            reward += shape_value # Add to reward

        # isEfficient (progress type, shape: v > 0 ? 1/v : v)
        efficient_meta = self._get_predicate_metadata('isEfficient')
        if efficient_meta.get('type') == 'progress' and 'shape' in efficient_meta:
            shape_value = self._evaluate_shape_expression(efficient_meta['shape'])
            reward += shape_value # Add to reward


        # --- 4. Check Problem Postcondition (ensure) ---
        # This is the goal of the problem.
        postcondition_formula_kx = self._problem_config.postcondition.formula
        postcondition_satisfied = self._evaluate_expression(postcondition_formula_kx, is_constraint=True)
        ensure_meta = self._get_predicate_metadata('ensure') # Get metadata for 'ensure' predicate

        if postcondition_satisfied:
            reward, terminated, truncated = self._apply_meta_effect(
                ensure_meta, reward, terminated, truncated, is_failure_case=False
            )
        else:
            reward, terminated, truncated = self._apply_meta_effect(
                ensure_meta, reward, terminated, truncated, is_failure_case=True
            )

            # Also apply shape if available for the postcondition (ensure)
            if ensure_meta.get('type') == 'progress' and 'shape' in ensure_meta:
                shape_value = self._evaluate_shape_expression(ensure_meta['shape'])
                reward += shape_value # Add to reward

        # --- 5. Check Truncation ---
        if self.current_step >= self.max_episode_steps:
            truncated = True
            # Small penalty for truncating without reaching goal, unless already terminated by ensure failure
            if not terminated and not postcondition_satisfied:
                reward -= 10.0

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
""")
    return env_code
```