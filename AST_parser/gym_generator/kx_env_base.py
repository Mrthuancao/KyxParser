import gymnasium as gym
import numpy as np
import math
import textwrap
import re
import z3
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod

# The import for ast_expand_formula has been removed as the expansion logic is now
# handled directly within _expand_formula in this file.
from gym_generator.expression_helpers import format_kx_expr_for_z3_formula_generator, expand_formula # Import expand_formula here

# NEW: Import cisb_analysis functions
# Assuming cisb_analysis.py is in the same directory or an accessible package path
from gym_generator.cisb_analysis import analyze_cisb_diversity, get_dynamic_bounding_box_defs, is_value_in_bounding_box, _approx_eq_helper
# this is filed AST_parser/gym_generator/kx_env_base.py

# NEW: Safe division helper
def safe_div(a, b):
    """Prevents ZeroDivisionError by returning a very large number or 0."""
    try:
        if b == 0 or b == 0.0:
            return a * 1e18 if a != 0 else 0.0
        return a / b
    except ZeroDivisionError:
        return a * 1e18 if a != 0 else 0.0

class _KeYmaeraXEnvBase(gym.Env, ABC):
    def __init__(self, config_serialized_str: str, render_mode=None, cisb_size: int = 1000, cisb_populate_attempts: int = 1000, seed: Optional[int] = None, debug_mode: bool = False, max_episode_steps: int = 200):
        super().__init__()
        self.render_mode = render_mode
        self.np_random = np.random.default_rng(seed=seed)
        self.debug_mode = debug_mode
        self.max_episode_steps = max_episode_steps

        # 1. Setup AST Classes for deserialization
        from pyast_config import (
            GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig,
            ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig,
            HybridProgramConfig, ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig,
            VariableDeclarationConfig, PreconditionConfig, PostconditionConfig,
            LoopExecutionConfig, HybridProgramStepConfig,
            ActionType, ChoiceActionConfig, SequenceActionConfig,
            DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig,
            ActionConstraintConfig, CallHybridProgramConfig
        )
        # Also need expression_helpers for parsing constant values if they are expressions
        from gym_generator.expression_helpers import kx_expr
        from gym_generator.expression_ast_nodes import ExpressionASTNode # For _ensure_ast_node type hint

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
        ast_classes_for_eval.update({
            'list': list, 'dict': dict, 'tuple': tuple, 'None': None, 'True': True, 'False': False
        })

        if self.debug_mode:
            print(f"DEBUG_BASE_INIT: Raw config_serialized_str received:\n{config_serialized_str}") # Print full string
            print(f"DEBUG_BASE_INIT: Hash of config_serialized_str: {hash(config_serialized_str)}") # Add hash to detect changes

        try:
            self.config: GymEnvironmentConfig = eval(config_serialized_str, {'__builtins__': None}, ast_classes_for_eval)
            if self.debug_mode:
                print(f"DEBUG_BASE_INIT: Deserialization successful.")
                # ADDED: Print the actual content of state_variables after deserialization
                print(f"DEBUG_BASE_INIT: Content of self.config.variables.state_variables names (after eval): {[v.name for v in self.config.variables.state_variables]}")
                print(f"DEBUG_BASE_INIT: Length of self.config.variables.state_variables: {len(self.config.variables.state_variables)}")
                print(f"DEBUG_BASE_INIT: id(self.config.variables) after eval: {id(self.config.variables)}")
        except Exception as e:
            raise ValueError(f"Failed to deserialize GymEnvironmentConfig: {e}")

        self._definitions_config = self.config.definitions
        self._variables_config = self.config.variables
        self._problem_config = self.config.problem
        if self.debug_mode:
            print(f"DEBUG_BASE_INIT: id(self._variables_config) after assignment: {id(self._variables_config)}")
            print(f"DEBUG_BASE_INIT: Content of self._variables_config.state_variables names (after assignment): {[v.name for v in self._variables_config.state_variables]}")


        self.equality_tolerance = 1e-7
        self.z3_equality_tolerance = 0.05

        # 2. Initialize Constants (placeholders, actual values set by generated class or CISB)
        self.constants: Dict[str, float] = {}
        for c in self._definitions_config.constants:
            self.constants[c.name] = 0.0 # Placeholder. Will be overwritten by fixed values in child class, or sampled by CISB.

        # 3. Initialize Functions and Predicates (placeholders, actual lambdas set by generated class)
        self.functions: Dict[str, Any] = {}
        self.predicates: Dict[str, Any] = {}

        self._state_keys: List[str] = [v.name for v in self._variables_config.state_variables]
        if self.debug_mode:
            print(f"DEBUG_BASE_INIT: _state_keys after parsing config: {self._state_keys}, length: {len(self._state_keys)}")
            print(f"DEBUG_BASE_INIT: id(self._state_keys) after parsing config: {id(self._state_keys)}")

        self.state: Dict[str, float] = {key: 0.0 for key in self._state_keys}
        
        self.last_failing_precondition_formula = None
        self._ode_func_compiled = None
        self._ode_events_compiled = None
        self.current_step = 0

        self.state_history: List[Dict[str, Any]] = []
        self.last_episode_history: List[Dict[str, Any]] = []
        self.cumulative_episode_time: float = 0.0

        self._cisb: List[Dict[str, float]] = []
        self._cisb_size = cisb_size
        self._cisb_populate_attempts = cisb_populate_attempts
        self.sample_ranges: Dict[str, Tuple[float, float]] = {}
        
    def _evaluate_expression(self, expr_str: str, local_vars: Optional[Dict[str, Any]] = None, is_constraint: bool = False) -> Any:
        """
        Safe evaluation of expressions using environment's context.
        This method parses KeYmaera X expressions at runtime.
        If is_constraint is True, returns False on evaluation error.
        """

        if local_vars is None:
            local_vars = {}

        # Strip KeYmaera X constraint prefix '?' if present
        if expr_str.startswith('?'):
            expr_str = expr_str[1:].strip()

        processed_expr = expr_str.strip()

        # Combine all known identifiers from the environment that need prefixing
        all_env_identifiers = list(self.state.keys()) + \
                               list(self.constants.keys()) + \
                               list(self.functions.keys()) + \
                               list(self.predicates.keys())

        all_env_identifiers_sorted = sorted(all_env_identifiers, key=len, reverse=True)

        local_identifiers_in_scope = list(local_vars.keys()) if local_vars else []

        for ident_name in all_env_identifiers_sorted:
            if ident_name in local_identifiers_in_scope:
                continue

            replacement = ""
            if ident_name in self.state:
                replacement = f"self.state['{ident_name}']"
            elif ident_name in self.constants:
                replacement = f"self.constants['{ident_name}']"
            elif ident_name in self.functions:
                replacement = f"self.functions['{ident_name}']"
            elif ident_name in self.predicates:
                replacement = f"self.predicates['{ident_name}']"

            if replacement:
                # Use word boundaries to avoid replacing parts of other names (e.g., 'x' in 'xr')
                processed_expr = re.sub(r'\b' + re.escape(ident_name) + r'\b', replacement, processed_expr)

        expr_py_transformed = processed_expr

        def _parse_expression_recursive_runtime(sub_expr: str) -> str:
            sub_expr = sub_expr.strip()
            if not sub_expr:
                return ""

            # Handle parentheses first
            if sub_expr.startswith('(') and sub_expr.endswith(')'):
                balance = 0
                is_fully_enclosed = True
                # Check all but the last char to see if it's fully enclosed
                for i in range(len(sub_expr) - 1):
                    char = sub_expr[i]
                    if char == '(': balance += 1
                    elif char == ')': balance -= 1
                    if balance == 0: # Found a matching parenthesis before the end
                        is_fully_enclosed = False
                        break
                if is_fully_enclosed:
                    return _parse_expression_recursive_runtime(sub_expr[1:-1])

            # Find lowest precedence logical operator (IMPLIES) - right-associative
            balance = 0
            implication_idx = -1
            for i in range(len(sub_expr) - 2, -1, -1): # Iterate right-to-left
                char_pair = sub_expr[i:i+2]
                if char_pair == '->' and balance == 0:
                    implication_idx = i
                    break
                if sub_expr[i] == ')': balance += 1
                elif sub_expr[i] == '(': balance -= 1

            if implication_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:implication_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[implication_idx+2:].strip())
                return f"((not ({left})) or ({right}))"

            # Find next lowest precedence logical operator (OR) - left-associative
            balance = 0
            disjunction_idx = -1
            for i in range(len(sub_expr)): # Iterate left-to-right
                if sub_expr[i] == '|' and balance == 0:
                    disjunction_idx = i
                    break
                if sub_expr[i] == '(': balance += 1
                elif sub_expr[i] == ')': balance -= 1

            if disjunction_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:disjunction_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[disjunction_idx+1:].strip())
                return f"({left}) or ({right})"

            # Find next lowest precedence logical operator (AND) - left-associative
            balance = 0
            conjunction_idx = -1
            for i in range(len(sub_expr)): # Iterate left-to-right
                if sub_expr[i] == '&' and balance == 0:
                    conjunction_idx = i
                    break
                if sub_expr[i] == '(': balance += 1
                elif sub_expr[i] == ')': balance -= 1

            if conjunction_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:conjunction_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[conjunction_idx+1:].strip())
                return f"({left}) and ({right})"

            # Handle NOT operator (unary, highest logical precedence)
            if sub_expr.startswith('!'):
                operand = _parse_expression_recursive_runtime(sub_expr[1:].strip())
                return f"(not ({operand}))"

            # If no logical operators, handle relational operators
            relational_ops = [('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>'), ('=', '_approx_eq')]

            balance = 0
            rel_op_found_kx = None
            rel_op_found_py = None
            rel_op_idx = -1

            # Iterate to find the first relational operator (left-to-right)
            for i in range(len(sub_expr)):
                char = sub_expr[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0:
                    # Check multi-character operators first to avoid partial matches
                    for kx_op, py_op_format in [('!=', '!='), ('<=', '<='), ('>=', '>=')]:
                        if sub_expr[i:i+len(kx_op)] == kx_op:
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break

                    # Check single-character operators and '=' carefully
                    for kx_op, py_op_format in [('<', '<'), ('>', '>'), ('=', '_approx_eq')]:
                        if sub_expr[i] == kx_op:
                            # Ensure it's not part of a multi-char operator (e.g., '<' in '<=')
                            if (kx_op == '<' and i + 1 < len(sub_expr) and sub_expr[i+1] == '=') or \
                               (kx_op == '>' and i + 1 < len(sub_expr) and sub_expr[i+1] == '=') or \
                               (kx_op == '=' and ((i > 0 and sub_expr[i-1] in '<>=!') or \
                                                 (i < len(sub_expr) - 1 and sub_expr[i+1] == '='))):
                                continue # Skip if it's part of <=, >=, !=, or ==
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break

            if rel_op_found_kx:
                left = _parse_expression_recursive_runtime(sub_expr[:rel_op_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[rel_op_idx+len(rel_op_found_kx):].strip())
                if rel_op_found_kx == '=':
                    return f"self._approx_eq({left}, {right})"
                else:
                    return f"({left}) {rel_op_found_py} ({right})"

            # If no logical operators, it's a literal or identifier or function call
            return sub_expr

        expr_py_transformed = _parse_expression_recursive_runtime(processed_expr.replace('^', '**').replace('abs(', 'np.abs('))

        try:
            eval_globals_for_eval = {'np': np, 'math': math, '_approx_eq': self._approx_eq, 'self': self, 'safe_div': safe_div}
            result = eval(expr_py_transformed, eval_globals_for_eval, local_vars)
            return result
        except Exception as e:
            if self.debug_mode:
                print(f"DEBUG_EVAL: Error evaluating expression '{expr_str}' (transformed: '{expr_py_transformed}'): {e}")
            if is_constraint:
                return False
            raise

    # Removed the _expand_formula method from kx_env_base.py
    # The expansion is now handled by expression_helpers.expand_formula

    def _split_top_level_and_local(self, formula_str: str) -> List[str]:
        """
        Splits a formula string by top-level '&' operators, respecting brace nesting.
        This is a simplified version of the parser's split_by_operator_outside_braces
        tailored for logical conjunctions.
        """
        parts = []
        brace_level = 0
        current_part_start = 0

        formula_str = re.sub(r'\s*&\s*', '&', formula_str)

        i = 0
        while i < len(formula_str):
            if formula_str[i] == '(':
                brace_level += 1
            elif formula_str[i] == ')':
                brace_level -= 1

            if formula_str[i] == '&' and brace_level == 0:
                part = formula_str[current_part_start:i].strip()
                if part:
                    parts.append(part)
                current_part_start = i + 1
            i += 1

        last_part = formula_str[current_part_start:].strip()
        if last_part:
            parts.append(last_part)
        return parts if parts else [formula_str.strip()]

    def _get_predicate_metadata(self, predicate_name: str) -> Dict[str, Any]:
        """Retrieves metadata for a given predicate name."""
        for pred_config in self._definitions_config.predicates:
            if pred_config.name == predicate_name:
                return pred_config.metadata
        return {}

    def _evaluate_shape_expression(self, shape_expr: str) -> float:
        """
        Evaluates a shape expression from metadata.
        Shape expressions might use KeYmaera X ternary operators (e.g., 'value_if_true if condition else value_if_false').
        """
        if not shape_expr:
            return 0.0

        match = re.match(r'(.+)\s+if\s+(.+)\s+else\s+(.+)', shape_expr)
        if match:
            true_val_str, condition_str, false_val_str = match.groups()
            try:
                condition_result = self._evaluate_expression(condition_str)
                if condition_result:
                    return float(self._evaluate_expression(true_val_str))
                else:
                    return float(self._evaluate_expression(false_val_str))
            except Exception as e:
                if self.debug_mode:
                    print(f"DEBUG_EVAL: Warning: Error evaluating shape ternary expression '{shape_expr}': {e}")
                return 0.0
        else:
            try:
                return float(self._evaluate_expression(shape_expr))
            except Exception as e:
                if self.debug_mode:
                    print(f"DEBUG_EVAL: Warning: Error evaluating shape expression '{shape_expr}': {e}")
                return 0.0

    def _apply_meta_effect(self, metadata: Dict[str, Any], current_reward: float, current_terminated: bool, current_truncated: bool, is_failure_case: bool) -> Tuple[float, bool, bool]:
        """
        Applys reward/penalty/termination/truncation effects based on metadata.
        """
        reward_change = 0.0
        new_terminated = current_terminated
        new_truncated = current_truncated

        if metadata:
            event_key = 'on_failure' if is_failure_case else 'on_success'
            event_config_value = metadata.get(event_key) # Renamed to avoid confusion with string

            if is_failure_case:
                penalty = float(metadata.get('penalty', 0.0))
                reward_change += penalty
            else:
                success_reward = float(metadata.get('reward', 0.0))
                reward_change += success_reward

            if event_config_value:
                action = None
                if isinstance(event_config_value, dict): # Check if it's the dict format
                    action = event_config_value.get('action')
                elif isinstance(event_config_value, str): # Fallback for string format
                    parts = event_config_value.split(':')
                    action = parts[0]
                
                if action == 'terminate':
                    new_terminated = True
                elif action == 'truncate':
                    new_truncated = True

        return current_reward + reward_change, new_terminated, new_truncated

    def _execute_action_config(self, action_config: Any, agent_action: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """
        Recursively executes a given action configuration (Choice, Sequence, Assignment, Constraint).
        Returns (reward, terminated, applied_successfully).
        """
        from pyast_config import (ChoiceActionConfig, SequenceActionConfig, DeterministicAssignmentActionConfig,
                                 NondeterministicAssignmentActionConfig, ActionConstraintConfig, CallHybridProgramConfig)

        reward = 0.0
        terminated = False
        truncated = False # Initialize truncated here
        applied_successfully = True

        if isinstance(action_config, ChoiceActionConfig):
            choice_key = f"choice_{action_config.level_name}"
            if choice_key not in agent_action:
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Error: Choice key '{choice_key}' missing from agent action.")
                reward -= 200.0
                terminated = True
                applied_successfully = False
                return reward, terminated, applied_successfully

            choice_idx = agent_action[choice_key]
            if not (0 <= choice_idx < len(action_config.choices)):
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Error: Invalid choice index {choice_idx} for '{choice_key}'.")
                reward -= 200.0
                terminated = True
                applied_successfully = False
                return reward, terminated, applied_successfully

            chosen_sub_config = action_config.choices[choice_idx]
            if self.debug_mode:
                print(f"DEBUG_ACTION: Chosen action branch: {choice_idx}")
            return self._execute_action_config(chosen_sub_config, agent_action)

        elif isinstance(action_config, SequenceActionConfig):
            for sub_action in action_config.actions:
                r, t, s = self._execute_action_config(sub_action, agent_action)
                reward += r
                if t: terminated = True
                # Note: `truncated` is not directly updated here from sub_action's return.
                # If a sub-action truncates, the `terminated` or `truncated` flag should propagate.
                # For now, we'll assume `t` (terminated) covers critical exits.
                # If a sub-action explicitly returns `truncated=True` and `t=False`,
                # this would need to be `truncated = truncated or tr` if `tr` was returned.
                # Given current `_apply_meta_effect` returns, `t` and `tr` are mutually exclusive
                # for a single meta effect, so propagating `t` is usually sufficient for early exit.
                if not s:
                    applied_successfully = False
                    break
            return reward, terminated, applied_successfully

        elif isinstance(action_config, DeterministicAssignmentActionConfig):
            try:
                assigned_value = self._evaluate_expression(action_config.value)
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Assigning {action_config.variable} = {assigned_value:.6f} (from '{action_config.value}')")
                self.state[action_config.variable] = assigned_value
            except Exception as e:
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Error evaluating deterministic assignment for {action_config.variable}='{action_config.value}': {e}")
                reward -= 100.0
                terminated = True
                applied_successfully = False
            return reward, terminated, applied_successfully

        elif isinstance(action_config, NondeterministicAssignmentActionConfig):
            assign_key = f"assign_{action_config.variable}"
            if assign_key in agent_action:
                self.state[action_config.variable] = agent_action[assign_key]
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Nondeterministic assignment for {action_config.variable} = {agent_action[assign_key]:.6f}")
            else:
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Warning: Nondeterministic assignment for '{action_config.variable}' (key '{assign_key}') not found in agent action. Defaulting to current value.")
                reward -= 50.0
            return reward, terminated, applied_successfully

        elif isinstance(action_config, ActionConstraintConfig):
            constraint_met = self._evaluate_expression(action_config.formula, is_constraint=True)
            if self.debug_mode:
                print(f"DEBUG_ACTION: Evaluating constraint '{action_config.formula}'. Result: {constraint_met}")

            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', action_config.formula)
            if predicate_name_match:
                pred_name = predicate_name_match.group(1)
                metadata = self._get_predicate_metadata(pred_name)
            else:
                # Default metadata now uses the dictionary format for on_failure
                metadata = {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}}

            if not constraint_met:
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Constraint FAILED: '{action_config.formula}'. Setting applied_successfully=False.")
                reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=True)
                
                # Check for 'continue' action in metadata, now handling both dict and string formats
                on_failure_config = metadata.get('on_failure')
                if (isinstance(on_failure_config, dict) and on_failure_config.get('action') == 'continue') or \
                   (isinstance(on_failure_config, str) and on_failure_config.split(':')[0] == 'continue'):
                    applied_successfully = True
                    if self.debug_mode:
                        print(f"DEBUG_ACTION: Constraint FAILED but metadata says 'continue'. applied_successfully=True.")
                else:
                    applied_successfully = False
            else:
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Constraint PASSED: '{action_config.formula}'. Setting applied_successfully=True.")
                
                # Check for 'continue' action in metadata, now handling both dict and string formats
                on_success_config = metadata.get('on_success')
                if not ((isinstance(on_success_config, dict) and on_success_config.get('action') == 'continue') or \
                        (isinstance(on_success_config, str) and on_success_config.split(':')[0] == 'continue')):
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                applied_successfully = True

            return reward, terminated, applied_successfully

        elif isinstance(action_config, CallHybridProgramConfig):
            if self.debug_mode:
                print(f"DEBUG_ACTION: Warning: CallHybridProgramConfig '{action_config.hp_name}' encountered in _execute_action_config. Not fully supported yet.")
            reward -= 50.0
            applied_successfully = False
            return reward, terminated, applied_successfully

        else:
            if self.debug_mode:
                print(f"DEBUG_ACTION: Error: Unknown action config type: {type(action_config)}")
            reward -= 200.0
            terminated = True
            applied_successfully = False
            return reward, terminated, applied_successfully

    def _apply_control_action(self, action: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """
        Applies the agent's chosen action by traversing the control function's AST.
        """
        # Assuming _control_func_action_config is set elsewhere, e.g., in the concrete environment class
        # For this base class, it's not defined, but the method signature implies it will be used.
        # If this method is called, _control_func_action_config must be defined in the derived class.
        if not hasattr(self, '_control_func_action_config'):
            if self.debug_mode:
                print("DEBUG_CONTROL: _control_func_action_config not defined. Cannot apply control action.")
            return -200.0, True, False # Indicate failure

        reward, terminated, applied_successfully = self._execute_action_config(self._control_func_action_config, action)
        if self.debug_mode:
            print(f"DEBUG_CONTROL: _apply_control_action final result: reward={reward:.2f}, terminated={terminated}, applied_successfully={applied_successfully}")

        if applied_successfully:
            self.state['t'] = 0.0

        return reward, terminated, applied_successfully

    def _kx_expr_to_z3_formula(self, kx_expr: str, z3_symbol_map: Dict[str, Any], fixed_constants: Dict[str, Union[int, float]]) -> Any:
        """
        Converts a KeYmaera X expression string into a Z3 formula using the AST-based parser.
        This function now relies on `expression_helpers.format_kx_expr_for_z3_formula_generator`
        to handle both parsing and predicate expansion.
        """
        if self.debug_mode:
            print(f"DEBUG_CISB: Preparing KeYmaera X expression for Z3 conversion: {kx_expr}")

        return format_kx_expr_for_z3_formula_generator(
            kx_expr, # Pass the original expression, expansion will happen inside
            self._definitions_config,
            z3_symbol_map,
            fixed_constants,
            self.z3_equality_tolerance,
            self.debug_mode
        )

    # Helper function to convert AST node to KeYmaera X string, needed for constant evaluation
    def _node_to_kx(self, node: 'ExpressionASTNode') -> str:
        """Converts an AST node back to a KeYmaera X string representation."""
        from kyx_parser.AST_parser.expression_ast_nodes import (
            ExpressionFunctionCall, ExpressionIdentifier,
            ExpressionBinaryOp, ExpressionUnaryOp, ExpressionLiteral
        )
        if isinstance(node, ExpressionIdentifier):
            return node.name
        elif isinstance(node, ExpressionLiteral):
            return str(node.value)
        elif isinstance(node, ExpressionBinaryOp):
            return f"({self._node_to_kx(node.left)} {node.to_kx_op()} {self._node_to_kx(node.right)})"
        elif isinstance(node, ExpressionUnaryOp):
            return f"({node.to_kx_op()}{self._node_to_kx(node.operand)})"
        elif isinstance(node, ExpressionFunctionCall):
            args_str = ", ".join([self._node_to_kx(arg) for arg in node.args])
            return f"{self._node_to_kx(node.name)}({args_str})"
        else:
            if self.debug_mode:
                print(f"DEBUG_NODE_TO_KX: Warning: Unhandled AST node type in _node_to_kx: {type(node)}")
            return str(node)

    # Helper function to ensure AST node, needed for constant evaluation
    def _ensure_ast_node(self, parsed_result: Any) -> Optional['ExpressionASTNode']:
        """Ensures the result is an AST node, handling pyparsing.ParseResults."""
        from pyparsing import ParseResults
        from kyx_parser.AST_parser.expression_ast_nodes import ExpressionASTNode
        if isinstance(parsed_result, ParseResults): # Corrected from parsed_parsed_result
            return parsed_result[0] if parsed_result else None
        return parsed_result if isinstance(parsed_result, ExpressionASTNode) else None


    def _populate_cisb_diverse_z3(self):
        """
        Populates the Contrastive Initial State Buffer (CISB) with diverse initial states
        that satisfy the problem's precondition, using Z3.Optimize with soft constraints
        and blocking clauses for diversity.
        """
        if self.debug_mode:
            print(f"DEBUG_CISB: Populating Contrastive Initial State Buffer (CISB) with {self._cisb_size} diverse states using Z3.Optimize...")
        current_cisb_count = 0
        attempts = 0

        z3_symbols = {}
        fixed_constants_map = {} # Stores constants that have a fixed numerical value or evaluated expression value

        # Initialize sample_ranges for state variables (if not already set)
        for var_decl in self._variables_config.state_variables:
            z3_symbols[var_decl.name] = z3.Real(var_decl.name)
            # Default bounds for state variables if not specified elsewhere
            if var_decl.name not in self.sample_ranges:
                self.sample_ranges[var_decl.name] = (-100.0, 100.0)

        # Process constants
        for const_decl in self._definitions_config.constants:
            # NEW: Check if constant is already fixed in self.constants (set by the generated env's __init__)
            # We check for `is not None` and `!= 0.0` to avoid treating the initial placeholder 0.0 as a fixed value.
            # This check is crucial here because self.constants can be pre-populated by the child class.
            if const_decl.name in self.constants and self.constants[const_decl.name] is not None and self.constants[const_decl.name] != 0.0:
                fixed_constants_map[const_decl.name] = self.constants[const_decl.name]
                if self.debug_mode:
                    print(f"DEBUG_CISB: Constant '{const_decl.name}' is pre-set in self.constants to {self.constants[const_decl.name]}. Treating as fixed.")
            elif const_decl.value is not None and re.fullmatch(r'^-?\d+(?:\.\d+)?$', const_decl.value):
                # If constant has a fixed numerical value from config, treat it as such
                fixed_constants_map[const_decl.name] = float(const_decl.value)
                if self.debug_mode:
                    print(f"DEBUG_CISB: Constant '{const_decl.name}' has fixed value '{const_decl.value}' from config. Treating as fixed.")
            else:
                # Otherwise, treat it as a variable for Z3 and apply bounds from metadata if available
                z3_symbols[const_decl.name] = z3.Real(const_decl.name)
                if 'bound' in const_decl.metadata:
                    bound_str = const_decl.metadata['bound']
                    # Corrected regex to parse bounds string (e.g., "0.1:0.5" or "[0.1:0.5]")
                    bound_part = bound_str.split(';')[0].strip()
                    # Updated regex to correctly capture numbers within optional square brackets
                    match = re.match(r'^\s*[?\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*]?\s*$', bound_part)
                    if match:
                        low = float(match.group(1))
                        high = float(match.group(2))
                        self.sample_ranges[const_decl.name] = (low, high)
                        if self.debug_mode:
                            print(f"DEBUG_CISB: Applied bound for constant '{const_decl.name}': [{low}, {high}]")
                    else:
                        if self.debug_mode:
                            print(f"DEBUG_CISB: Warning: Could not parse bound string '{bound_str}' for constant '{const_decl.name}'. Using default range (-100.0, 100.0).")
                        if const_decl.name not in self.sample_ranges:
                            self.sample_ranges[const_decl.name] = (-100.0, 100.0)
                else:
                    if const_decl.name not in self.sample_ranges:
                        self.sample_ranges[const_decl.name] = (-100.0, 100.0)


        opt = z3.Optimize()
        opt.set(timeout=10000)

        precondition_kx = self._problem_config.precondition.formula
        if self.debug_mode:
            print(f"DEBUG_CISB: Precondition formula for CISB population: {precondition_kx}")
        try:
            # The expansion is now handled inside format_kx_expr_for_z3_formula_generator
            z3_precondition_formula = self._kx_expr_to_z3_formula(precondition_kx, z3_symbols, fixed_constants_map)
            opt.add(z3_precondition_formula)
        except ValueError as e:
            print(f"CRITICAL ERROR: Failed to convert precondition to Z3 formula during CISB population: {e}")
            return

        for var_name, z3_var in z3_symbols.items():
            # Use the bounds from self.sample_ranges, which now includes constant bounds
            low, high = self.sample_ranges.get(var_name, (-100.0, 100.0))
            opt.add(z3_var >= low, z3_var <= high)

        while current_cisb_count < self._cisb_size and attempts < self._cisb_populate_attempts:
            attempts += 1
            opt.push()

            # Re-check if any variable is not fixed and has a range > 0
            any_variable_can_be_sampled = False
            for var_name, z3_var in z3_symbols.items():
                if var_name not in fixed_constants_map and var_name != 't':
                    low, high = self.sample_ranges.get(var_name, (-100.0, 100.0))
                    if not _approx_eq_helper(low, high, tolerance=1e-9):
                        any_variable_can_be_sampled = True
                        break
            
            # If no variables can be sampled, we still try to find a solution if one exists.
            # The soft constraints are primarily for diversity.
            if not any_variable_can_be_sampled:
                if self.debug_mode:
                    print("DEBUG_CISB: No variables with non-zero range found for soft constraints. Attempting to find a basic solution.")


            check_result = opt.check()

            if check_result == z3.sat:
                model = opt.model()
                if model is None:
                    if self.debug_mode:
                        print("DEBUG_CISB: Z3 model is None despite SAT result. Skipping this attempt.")
                    opt.pop()
                    continue

                solved_values = {}
                
                # Temporarily store evaluated Z3 variables
                evaluated_z3_vars = {}
                for k, v in z3_symbols.items():
                    evaluated_val = model.evaluate(v, model_completion=True)
                    if evaluated_val is None:
                        if self.debug_mode:
                            print(f"DEBUG_CISB: Variable '{k}' not assigned a value by Z3 model. Skipping for this sample.")
                        continue

                    # Robust conversion from Z3 value to Python float
                    try:
                        if hasattr(evaluated_val, 'as_fraction'):
                            # Handles rational numbers like 1/2
                            evaluated_z3_vars[k] = float(evaluated_val.as_fraction())
                        elif hasattr(evaluated_val, 'as_long'):
                            # Handles integer numbers
                            evaluated_z3_vars[k] = float(evaluated_val.as_long())
                        elif hasattr(evaluated_val, 'as_decimal'):
                            # Handles real numbers with a decimal representation
                            decimal_str = evaluated_val.as_decimal(20) # Use a high precision
                            evaluated_z3_vars[k] = float(decimal_str)
                        else:
                            # Fallback to string conversion, handling '?' if present
                            s_val = str(evaluated_val)
                            if '?' in s_val:
                                numeric_part = s_val.split('?')[0]
                                evaluated_z3_vars[k] = float(numeric_part)
                            else:
                                evaluated_z3_vars[k] = float(s_val)
                    except (z3.Z3Exception, ValueError) as e:
                        if self.debug_mode:
                            print(f"DEBUG_CISB: Error converting Z3 value '{evaluated_val}' for key '{k}' to float: {e}. Skipping this sample.")
                        continue # Skip this sample
                
                # Now, populate solved_values, prioritizing evaluated_z3_vars for Z3-bound variables
                # and fixed_constants_map for explicitly fixed constants.
                solved_values.update(evaluated_z3_vars)
                solved_values.update(fixed_constants_map) # Fixed constants override if there's overlap (e.g. if 'A' is both a Z3 var and in fixed_constants_map)

                # Now, evaluate constants that have a 'value' expression (e.g., minV = A*ep)
                # These must be evaluated using the values *just obtained* from the model or fixed_constants_map.
                for const_def in self._definitions_config.constants:
                    if const_def.value is not None and not re.fullmatch(r'^-?\d+(?:\.\d+)?$', const_def.value):
                        # This is a constant whose value is an expression
                        try:
                            # Create a temporary environment context for evaluation
                            temp_state = self.state.copy()
                            temp_constants = self.constants.copy()
                            
                            # Update temporary state/constants with current solved_values for evaluation
                            for k, v in solved_values.items():
                                if k in temp_state:
                                    temp_state[k] = v
                                elif k in temp_constants:
                                    temp_constants[k] = v

                            # Temporarily set the environment's state and constants
                            original_state = self.state
                            original_constants = self.constants
                            self.state = temp_state
                            self.constants = temp_constants

                            evaluated_const_val = float(self._evaluate_expression(const_def.value))
                            solved_values[const_def.name] = evaluated_const_val

                            # Restore original state and constants
                            self.state = original_state
                            self.constants = original_constants

                        except Exception as e:
                            if self.debug_mode:
                                print(f"DEBUG_CISB: Error evaluating constant '{const_def.name}' with expression '{const_def.value}': {e}. Skipping this sample.")
                            opt.pop()
                            continue # Skip this sample

                # Normalize dx, dy if they represent a direction vector
                if 'dx' in solved_values and 'dy' in solved_values:
                    dx_val = solved_values['dx']
                    dy_val = solved_values['dy']
                    magnitude = math.sqrt(dx_val**2 + dy_val**2)
                    if magnitude > 1e-6: # Avoid division by zero for very small magnitudes
                        solved_values['dx'] = dx_val / magnitude
                        solved_values['dy'] = dy_val / magnitude
                    else:
                        # Default to a valid direction if magnitude is too small
                        solved_values['dx'] = 1.0
                        solved_values['dy'] = 0.0

                # Ensure all state variables are present in solved_values
                all_state_vars_present = True
                for var_def in self._variables_config.state_variables:
                    if var_def.name not in solved_values:
                        if self.debug_mode:
                            print(f"DEBUG_CISB: Required state variable '{var_def.name}' not found in solved_values for this sample. Skipping.")
                        all_state_vars_present = False
                        break
                if not all_state_vars_present:
                    opt.pop()
                    continue

                self._cisb.append(solved_values)
                current_cisb_count += 1
                if self.debug_mode and current_cisb_count % 100 == 0:
                    print(f"DEBUG_CISB: ... {current_cisb_count}/{self._cisb_size} states populated.")

                blocking_conditions = []
                for var_name, z3_var in z3_symbols.items():
                    val = solved_values.get(var_name)
                    if val is not None:
                        diversity_epsilon = 0.01 # Small epsilon to ensure diversity, can be tuned
                        # Block the exact solution found for this variable
                        blocking_conditions.append(
                            z3.Or(z3_var < val - diversity_epsilon, z3_var > val + diversity_epsilon)
                        )

                if blocking_conditions:
                    opt.add(z3.Or(blocking_conditions))

                opt.pop() # Pop the soft constraints and blocking clause for the next attempt

            elif check_result == z3.unsat:
                if self.debug_mode:
                    print(f"DEBUG_CISB: Z3 UNSAT during CISB population attempt {attempts}. No more diverse solutions found within current constraints.")
                opt.pop()
                break
            else: # z3.unknown
                if self.debug_mode:
                    print(f"DEBUG_CISB: Z3 UNKNOWN during CISB population attempt {attempts}. Retrying with new soft constraints.")
                opt.pop()

        if current_cisb_count < self._cisb_size:
            if self.debug_mode:
                print(f"DEBUG_CISB: Warning: Could not populate {current_cisb_count} states for CISB after {attempts} attempts.")
        else:
            if self.debug_mode:
                print(f"DEBUG_CISB: CISB population complete with {current_cisb_count} states.")

        unique_samples = set()
        for sample in self._cisb:
            # Create a tuple of sorted (key, rounded_value) pairs for uniqueness check
            rounded_sample = tuple(sorted([(k, round(v, 6)) for k, v in sample.items()]))
            unique_samples.add(rounded_sample)

        if self.debug_mode:
            print(f"DEBUG_CISB: Number of unique initial states in CISB: {len(unique_samples)}")

            print("DEBUG_CISB: Sample initial states from CISB (first 5):")
            for i, sample in enumerate(self._cisb[:min(5, len(self._cisb))]):
                print(f"  Sample {i+1}: {sample}")
                # Ensure these keys exist before accessing
                A_val = sample.get('A', 'N/A')
                b_val = sample.get('b', 'N/A')
                ep_val = sample.get('ep', 'N/A')
                print(f"  Solved constants for CISB entry: A={A_val}, b={b_val}, ep={ep_val}")

        if not self._cisb:
            print("CRITICAL ERROR: CISB is empty! This environment cannot function correctly without initial states.")
            print("Please check precondition and sampling ranges for satisfiability.")

        # NEW: Call the diversity analysis function after CISB population
        if self.debug_mode: # Only run analysis if debug_mode is True
            # Create a wrapper for _evaluate_expression that temporarily updates state/constants
            def cisb_entry_evaluate_expression_wrapper(expr_str: str, cisb_entry_context: Dict[str, Any]) -> Any:
                """
                A wrapper function to evaluate expressions in the context of a specific CISB entry.
                It temporarily updates self.state and self.constants, then restores them.
                """
                # Store the current state/constants
                temp_original_state = self.state.copy()
                temp_original_constants = self.constants.copy()

                # Update self.state and self.constants with values from cisb_entry_context
                # Prioritize values from cisb_entry_context
                for k, v in cisb_entry_context.items():
                    if k in self.state:
                        self.state[k] = v
                    elif k in self.constants:
                        self.constants[k] = v
                
                try:
                    # _evaluate_expression will now use the temporarily updated self.state/self.constants
                    # Pass an empty local_vars dict as the context is already in self.state/self.constants
                    result = self._evaluate_expression(expr_str, local_vars={})
                finally:
                    # Restore original state and constants
                    self.state = temp_original_state
                    self.constants = temp_original_constants
                return result

            analyze_cisb_diversity(
                self._cisb,
                self.config,
                self.sample_ranges,
                cisb_entry_evaluate_expression_wrapper, # Pass the wrapper function
                self.debug_mode
            )

    def _compile_ode_func(self, func_str: str):
        exec_globals = {'np': np, 'math': math, 'self': self, 'safe_div': safe_div}
        exec_locals = {}
        exec(textwrap.dedent(func_str), exec_globals, exec_locals)
        return exec_locals['ode_func']

    def _compile_ode_events(self, events_str: str):
        if not events_str.strip(): return []
        exec_globals = {'np': np, 'math': math, 'self': self, 'safe_div': safe_div}
        exec_locals = {}
        exec(textwrap.dedent(events_str), exec_globals, exec_locals)
        return exec_locals['ode_events']

    def _approx_eq(self, a, b):
        return np.isclose(a, b, atol=self.equality_tolerance)

    def _get_obs(self):
        # This method is now primarily handled by the child class (GeneratedKXEnv)
        # However, a basic implementation is here for abstract method compliance
        # and as a fallback. The child's _get_obs will call super()._get_obs() if needed.
        return np.array([float(self.state.get(key, 0.0)) for key in self._state_keys], dtype=np.float32)

    def _get_info(self):
        return self.state.copy()

    def _get_state_for_history(self) -> Dict[str, Any]:
        """Returns the current state combined with episode_time for history recording."""
        current_state_for_history = self.state.copy()
        current_state_for_history['episode_time'] = self.cumulative_episode_time
        return current_state_for_history

    def _add_state_to_history(self):
        """Appends the current state and cumulative time to the history."""
        self.state_history.append(self._get_state_for_history())

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and prepares for a new episode.
        """
        super().reset(seed=seed)

        if self.debug_mode:
            print(f"DEBUG_BASE_RESET: _state_keys at start of base reset: {self._state_keys}, len: {len(self._state_keys)}")
            print(f"DEBUG_BASE_RESET: id(self._state_keys) at start of base reset: {id(self._state_keys)}")


        self.last_episode_history = list(self.state_history)

        self.state_history = []
        self.cumulative_episode_time = 0.0
        self.current_step = 0
        self.last_failing_precondition_formula = None

        if not self._cisb:
            if self.debug_mode:
                print("DEBUG_BASE_RESET: Error: CISB is empty. Generating a single initial state using Z3 as fallback.")
            # These fallback values should ideally match the fixed constants set in GeneratedKXEnv
            final_initial_state = {
                # Ensure these keys match the actual state variables of GeneratedKXEnv
                'xr': 0.0, 'vr': 0.0, 'ar': 0.0, 'xo': 0.0, 'vo': 0.1, 'ao': 0.0, 't': 0.0, 'T': 0.0,
                'ep': 0.05, 'b': 2.0, 'A': 5.0, # Example values for sampled constants
                'Vmin': 0.1, 'ixr': 0.0, 'ixo': 0.0, 'D': 5.0
            }
        else:
            chosen_initial_state_entry = self.np_random.choice(self._cisb)
            final_initial_state = chosen_initial_state_entry.copy()

        if self.debug_mode:
            print(f"DEBUG_BASE_RESET: Chosen final_initial_state from CISB/fallback: {final_initial_state}")
            print(f"DEBUG_BASE_RESET: Keys in final_initial_state: {final_initial_state.keys()}")

        # Re-initialize self.state to ensure it has all _state_keys before populating
        self.state: Dict[str, float] = {key: 0.0 for key in self._state_keys}
        if self.debug_mode:
            print(f"DEBUG_BASE_RESET: self.state initialized with all _state_keys (before populating from final_initial_state): {self.state.keys()}")


        for var_name, value in final_initial_state.items():
            if var_name in self.state:
                self.state[var_name] = value
            elif var_name in self.constants:
                self.constants[var_name] = value

        if self.debug_mode:
            print(f"DEBUG_BASE_RESET: self.state after populating from final_initial_state: {self.state.keys()}, length: {len(self.state.keys())}")
            print(f"DEBUG_BASE_RESET: _state_keys after populating self.state: {self._state_keys}, length: {len(self._state_keys)}")
            print(f"DEBUG_BASE_RESET: id(self._state_keys) after populating self.state: {id(self._state_keys)}")


        self._add_state_to_history()

        observation = self._get_obs() # This calls the child's _get_obs
        info = self._get_info()
        if self.debug_mode:
            print(f"DEBUG_BASE_RESET: Observation returned by base reset (from child _get_obs): {observation}, length: {len(observation)}")
        return observation, info

    @abstractmethod
    def step(self, action):
        pass

    def render(self, mode="human", plot_config: Optional[Union[List[str], Dict[str, str]]] = None):
        """
        Renders the environment.
        - mode="human": Renders to the screen (not implemented for this base class).
        - mode="plot_data": Returns the raw state history for external plotting.
        - mode="matplotlib": Generates and displays matplotlib plots of the state history.
          `plot_config` can be:
          - A list of strings: ['var1', 'var2'] - plots these variables using their DataFrame column names as labels.
          - A dictionary: {'Plot Label 1': 'df_col_name1', 'Plot Label 2': 'df_col_name2'} - plots df_col_name1 with 'Plot Label 1'.
          If `plot_config` is None, all numerical columns in the DataFrame (except 'episode_time') will be plotted.
        """
        if mode == "human":
            pass
        elif mode == "plot_data":
            return self.state_history
        elif mode == "matplotlib":
            self._plot_state_history(plot_config=plot_config, history_data=self.last_episode_history)
            return None
        return None

    def _plot_state_history(self, plot_config: Optional[Union[List[str], Dict[str, str]]] = None, history_data: Optional[List[Dict[str, Any]]] = None):
        """
        Generates and displays a matplotlib plot of the state history.
        `plot_config` can be:
        - A list of strings: ['var1', 'var2'] - plots these variables using their DataFrame column names as labels.
        - A dictionary: {'Plot Label 1': 'df_col_name1', 'Plot Label 2': 'df_col_name2'} - plots df_col_name1 with 'Plot Label 1'.
        If `plot_config` is None, all numerical columns in the DataFrame (except 'episode_time') will be plotted.
        """
        data_to_plot = history_data if history_data is not None else self.state_history

        if not data_to_plot:
            if self.debug_mode:
                print("DEBUG_PLOT: No state history available to plot.")
            return

        df = pd.DataFrame(data_to_plot)
        if self.debug_mode:
            print(f"DEBUG_PLOT: DataFrame shape: {df.shape}")
            print(f"DEBUG_PLOT: DataFrame head:\n{df.head()}")
            print(f"DEBUG_PLOT: DataFrame tail:\n{df.tail()}")
            print(f"DEBUG_PLOT: DataFrame columns: {df.columns.tolist()}")


        x_axis_key = 'episode_time'
        if x_axis_key not in df.columns:
            print(f"DEBUG_PLOT: Error: Time key '{x_axis_key}' not found in state history for plotting. This indicates an issue with state history recording.")
            return

        variables_to_plot_map: Dict[str, str] = {} # {plot_label: df_column_name}

        if plot_config is None:
            # Default behavior: plot all numerical columns except the x_axis_key
            for col_name in df.columns:
                if col_name != x_axis_key and pd.api.types.is_numeric_dtype(df[col_name]):
                    variables_to_plot_map[col_name] = col_name # Use df name as label
            if self.debug_mode and not variables_to_plot_map:
                print("DEBUG_PLOT: No numerical columns found in DataFrame for default plotting.")

        elif isinstance(plot_config, list):
            for var_name in plot_config:
                if var_name in df.columns:
                    variables_to_plot_map[var_name] = var_name
                else:
                    if self.debug_mode:
                        print(f"DEBUG_PLOT: Warning: Variable '{var_name}' requested for plotting but not found in DataFrame columns: {df.columns.tolist()}.")
        elif isinstance(plot_config, dict):
            for plot_label, df_col_name in plot_config.items():
                if df_col_name in df.columns:
                    variables_to_plot_map[plot_label] = df_col_name
                else:
                    if self.debug_mode:
                        print(f"DEBUG_PLOT: Warning: DataFrame column '{df_col_name}' (for plot label '{plot_label}') not found in DataFrame columns: {df.columns.tolist()}.")
        else:
            if self.debug_mode:
                print(f"DEBUG_PLOT: Invalid plot_config type: {type(plot_config)}. Expected list or dict.")
            return

        if not variables_to_plot_map:
            if self.debug_mode:
                print(f"DEBUG_PLOT: No valid variables to plot after processing plot_config.")
            return

        num_vars = len(variables_to_plot_map)
        cols = 2
        rows = (num_vars + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        fig.suptitle('Agent State Evolution Over Time', fontsize=16)

        min_y_range_absolute = 0.05

        # Iterate through the map to plot
        for i, (plot_label, df_col_name) in enumerate(variables_to_plot_map.items()):
            ax = axes[i]
            ax.plot(df[x_axis_key], df[df_col_name], label=plot_label)
            ax.set_xlabel(f"Time ({x_axis_key})")
            ax.set_ylabel(plot_label) # Use the plot_label here
            ax.set_title(f"Evolution of {plot_label}") # Use the plot_label here
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            min_time = df[x_axis_key].min()
            max_time = df[x_axis_key].max()
            if max_time <= min_time + 1e-9:
                ax.set_xlim(0, max_time + 0.01)
            else:
                ax.set_xlim(0, max_time)

            min_val = df[df_col_name].min()
            max_val = df[df_col_name].max()

            data_range = max_val - min_val

            if self.debug_mode:
                print(f"DEBUG_PLOT: Plotting {plot_label} (from DataFrame column '{df_col_name}'): min_val={min_val:.6f}, max_val={max_val:.6f}, data_range={data_range:.6f}")

            if data_range < min_y_range_absolute:
                avg_val = (min_val + max_val) / 2
                lower_bound = avg_val - min_y_range_absolute / 2
                upper_bound = avg_val + min_y_range_absolute / 2
                ax.set_ylim(lower_bound, upper_bound)
                if self.debug_mode:
                    print(f"  DEBUG_PLOT: Applied forced y-lim: [{lower_bound:.6f}, {upper_bound:.6f}] (range {min_y_range_absolute})")
            else:
                buffer = data_range * 0.1
                lower_bound = min_val - buffer
                upper_bound = max_val + buffer
                ax.set_ylim(min_val - buffer, max_val + buffer)
                if self.debug_mode:
                    print(f"  DEBUG_PLOT: Applied buffered y-lim: [{lower_bound:.6f}, {upper_bound:.6f}] (range {upper_bound - lower_bound:.6f})")

        # Hide any unused subplots
        for j in range(0, len(axes)):
            if j >= num_vars:
                fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def close(self):
        plt.close('all')
        pass