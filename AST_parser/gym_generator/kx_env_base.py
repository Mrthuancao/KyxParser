import gymnasium as gym
import numpy as np
import math
import textwrap
import re
import z3
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

# NEW: Import expression_helpers for AST-based Z3 conversion
from gym_generator.expression_helpers import format_kx_expr_for_z3_formula_generator
from gym_generator.expression_helpers import expand_formula as ast_expand_formula # Alias to avoid name collision with self._expand_formula

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
    def __init__(self, config_serialized_str: str, render_mode=None, cisb_size: int = 1000, cisb_populate_attempts: int = 5000, seed: Optional[int] = None, debug_mode: bool = False, max_episode_steps: int = 200):
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

        try:
            self.config: GymEnvironmentConfig = eval(config_serialized_str, {'__builtins__': None}, ast_classes_for_eval)
        except Exception as e:
            raise ValueError(f"Failed to deserialize GymEnvironmentConfig: {e}")

        self._definitions_config = self.config.definitions
        self._variables_config = self.config.variables
        self._problem_config = self.config.problem
        self.equality_tolerance = 1e-7
        self.z3_equality_tolerance = 0.05

        # 2. Initialize Constants (placeholders, actual values set by generated class or CISB)
        self.constants: Dict[str, float] = {}
        for c in self._definitions_config.constants:
            self.constants[c.name] = 0.0 # Placeholder

        # 3. Initialize Functions and Predicates (placeholders, actual lambdas set by generated class)
        self.functions: Dict[str, Any] = {}
        self.predicates: Dict[str, Any] = {}

        self._state_keys: List[str] = [v.name for v in self._variables_config.state_variables]
        self.state: Dict[str, float] = {key: 0.0 for key in self._state_keys}

        if 't' not in self.state:
            self.state['t'] = 0.0
            if 't' not in self._state_keys:
                self._state_keys.append('t')

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
        
        self._populate_cisb_diverse_z3() 

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
                processed_expr = re.sub(r'\b' + re.escape(ident_name) + r'\b', replacement, processed_expr)

        expr_py_transformed = processed_expr

        def _parse_expression_recursive_runtime(sub_expr: str) -> str:
            sub_expr = sub_expr.strip()
            if not sub_expr:
                return ""

            if sub_expr.startswith('(') and sub_expr.endswith(')'):
                balance = 0
                is_fully_enclosed = True
                for i in range(len(sub_expr) - 1):
                    char = sub_expr[i]
                    if char == '(': balance += 1
                    elif char == ')': balance -= 1
                    if balance == 0:
                        is_fully_enclosed = False
                        break
                if is_fully_enclosed:
                    return _parse_expression_recursive_runtime(sub_expr[1:-1])

            balance = 0
            implication_idx = -1
            for i in range(len(sub_expr) - 2, -1, -1):
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

            balance = 0
            disjunction_idx = -1
            for i in range(len(sub_expr)):
                if sub_expr[i] == '|' and balance == 0:
                    disjunction_idx = i
                    break
                if sub_expr[i] == '(': balance += 1
                elif sub_expr[i] == ')': balance -= 1

            if disjunction_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:disjunction_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[disjunction_idx+1:].strip())
                return f"({left}) or ({right})"

            balance = 0
            conjunction_idx = -1
            for i in range(len(sub_expr)):
                if sub_expr[i] == '&' and balance == 0:
                    conjunction_idx = i
                    break
                if sub_expr[i] == '(': balance += 1
                elif sub_expr[i] == ')': balance -= 1

            if conjunction_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:conjunction_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[conjunction_idx+1:].strip())
                return f"({left}) and ({right})"

            if sub_expr.startswith('!'):
                operand = _parse_expression_recursive_runtime(sub_expr[1:].strip())
                return f"(not ({operand}))"

            relational_ops = [('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>'), ('=', '_approx_eq')]

            balance = 0
            rel_op_found_kx = None
            rel_op_found_py = None
            rel_op_idx = -1

            for i in range(len(sub_expr)):
                char = sub_expr[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0:
                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) > 1 and sub_expr[i:i+len(kx_op)] == kx_op:
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break

                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) == 1 and sub_expr[i] == kx_op:
                            if kx_op == '=' and ((i > 0 and sub_expr[i-1] in '<>=!') or \
                                            (i < len(sub_expr) - 1 and sub_expr[i+1] == '=')):
                                continue
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

            return sub_expr

        expr_py_final = _parse_expression_recursive_runtime(expr_py_transformed.replace('^', '**').replace('abs(', 'np.abs('))

        try:
            eval_globals_for_eval = {'np': np, 'math': math, '_approx_eq': self._approx_eq, 'self': self, 'safe_div': safe_div}
            result = eval(expr_py_final, eval_globals_for_eval, local_vars)
            return result
        except Exception as e:
            if self.debug_mode:
                print(f"DEBUG_EVAL: Error evaluating expression '{expr_str}' (transformed: '{expr_py_final}'): {e}")
            if is_constraint:
                return False
            raise

    def _expand_formula(self, formula_str: str, definitions: Any) -> str:
        """
        Recursively expands predicate calls within a formula string.
        This returns the formula in KeYmaera X syntax, suitable for further parsing.
        It handles simple argument substitution.
        """
        # Now uses the AST-based expand_formula from expression_helpers
        return ast_expand_formula(formula_str, definitions)


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
            event_config_str = metadata.get(event_key)

            if is_failure_case:
                penalty = float(metadata.get('penalty', 0.0))
                reward_change += penalty
            else:
                success_reward = float(metadata.get('reward', 0.0))
                reward_change += success_reward

            if event_config_str:
                parts = event_config_str.split(':')
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
        applied_successfully = True

        if isinstance(action_config, ChoiceActionConfig):
            choice_key = f"choice_{action_config.level_name}"
            if choice_key not in agent_action:
                print(f"DEBUG_ACTION: Error: Choice key '{choice_key}' missing from agent action.")
                reward -= 200.0
                terminated = True
                applied_successfully = False
                return reward, terminated, applied_successfully

            choice_idx = agent_action[choice_key]
            if not (0 <= choice_idx < len(action_config.choices)):
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
                metadata = {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'}

            if not constraint_met:
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Constraint FAILED: '{action_config.formula}'. Setting applied_successfully=False.")
                reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, False, is_failure_case=True)
                if not terminated and not truncated and metadata.get('on_failure', '').split(':')[0] == 'continue':
                    applied_successfully = True
                    if self.debug_mode:
                        print(f"DEBUG_ACTION: Constraint FAILED but metadata says 'continue'. applied_successfully=True.")
                else:
                    applied_successfully = False
            else:
                if self.debug_mode:
                    print(f"DEBUG_ACTION: Constraint PASSED: '{action_config.formula}'. Setting applied_successfully=True.")
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                applied_successfully = True

            return reward, terminated, applied_successfully

        elif isinstance(action_config, CallHybridProgramConfig):
            print(f"DEBUG_ACTION: Warning: CallHybridProgramConfig '{action_config.hp_name}' encountered in _execute_action_config. Not fully supported yet.")
            reward -= 50.0
            applied_successfully = False
            return reward, terminated, applied_successfully

        else:
            print(f"DEBUG_ACTION: Error: Unknown action config type: {type(action_config)}")
            reward -= 200.0
            terminated = True
            applied_successfully = False
            return reward, terminated, applied_successfully

    def _apply_control_action(self, action: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """
        Applies the agent's chosen action by traversing the control function's AST.
        """
        reward, terminated, applied_successfully = self._execute_action_config(self._control_func_action_config, action)
        if self.debug_mode:
            print(f"DEBUG_CONTROL: _apply_control_action final result: reward={reward:.2f}, terminated={terminated}, applied_successfully={applied_successfully}")

        if applied_successfully:
            self.state['t'] = 0.0

        return reward, terminated, applied_successfully

    def _kx_expr_to_z3_formula(self, kx_expr: str, z3_symbol_map: Dict[str, Any], fixed_constants: Dict[str, Union[int, float]]) -> Any:
        """
        Converts a KeYmaera X expression string into a Z3 formula using the AST-based parser.
        """
        return format_kx_expr_for_z3_formula_generator(
            kx_expr,
            self._definitions_config, # Pass DefinitionsConfig for predicate expansion
            z3_symbol_map,
            fixed_constants,
            self.z3_equality_tolerance,
            self.debug_mode
        )

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
        fixed_constants_map = {}

        # Initialize sample_ranges for state variables (if not already set)
        for var_decl in self._variables_config.state_variables:
            z3_symbols[var_decl.name] = z3.Real(var_decl.name)
            # Default bounds for state variables if not specified elsewhere
            if var_decl.name not in self.sample_ranges:
                self.sample_ranges[var_decl.name] = (-100.0, 100.0)

        # Process constants, applying bounds from metadata
        for const_decl in self._definitions_config.constants:
            if const_decl.value is not None and re.fullmatch(r'^-?\d+(\.\d+)?$', const_decl.value):
                # If constant has a fixed numerical value, treat it as such
                fixed_constants_map[const_decl.name] = float(const_decl.value)
            else:
                # Otherwise, treat it as a variable for Z3 and apply bounds from metadata if available
                z3_symbols[const_decl.name] = z3.Real(const_decl.name)
                if 'bound' in const_decl.metadata:
                    bound_str = const_decl.metadata['bound']
                    match = re.match(r'\[(-?\d+(\.\d+)?):(-?\d+(\.\d+)?)\]', bound_str)
                    if match:
                        low = float(match.group(1))
                        high = float(match.group(3))
                        self.sample_ranges[const_decl.name] = (low, high)
                        if self.debug_mode:
                            print(f"DEBUG_CISB: Applied bound for constant '{const_decl.name}': [{low}, {high}]")
                    else:
                        if self.debug_mode:
                            print(f"DEBUG_CISB: Warning: Could not parse bound string '{bound_str}' for constant '{const_decl.name}'. Using default range.")
                        if const_decl.name not in self.sample_ranges:
                            self.sample_ranges[const_decl.name] = (-100.0, 100.0)
                else:
                    if const_decl.name not in self.sample_ranges:
                        self.sample_ranges[const_decl.name] = (-100.0, 100.0)


        opt = z3.Optimize()
        opt.set(timeout=10000)

        precondition_kx = self._problem_config.precondition.formula
        try:
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

            soft_constraints_added = False
            for var_name, z3_var in z3_symbols.items():
                if var_name in ['v', 't']: # 'v' is often 0 in initial state, 't' is always 0.
                    continue

                low, high = self.sample_ranges.get(var_name, (-100.0, 100.0))
                if low == high: # Skip if the range is a single point
                    continue

                target = self.np_random.uniform(low, high)
                opt.add_soft(z3_var == target, weight=1)
                soft_constraints_added = True

            if not soft_constraints_added:
                # If no soft constraints were added (e.g., all variables have fixed values or single-point ranges),
                # we still need to try and find a solution.
                pass

            check_result = opt.check()

            if check_result == z3.sat:
                model = opt.model()
                solved_values = {}
                for k, v in z3_symbols.items():
                    evaluated_val = model.evaluate(v, model_completion=True)
                    if evaluated_val is not None:
                        if hasattr(evaluated_val, 'as_fraction'):
                            solved_values[k] = float(evaluated_val.as_fraction())
                        elif hasattr(evaluated_val, 'as_long'):
                            solved_values[k] = float(evaluated_val.as_long())
                        else:
                            solved_values[k] = float(str(evaluated_val))
                solved_values.update(fixed_constants_map)

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

                self._cisb.append(solved_values)
                current_cisb_count += 1
                if self.debug_mode and current_cisb_count % 100 == 0:
                    print(f"DEBUG_CISB: ... {current_cisb_count}/{self._cisb_size} states populated.")

                blocking_conditions = []
                for var_name, z3_var in z3_symbols.items():
                    val = solved_values.get(var_name)
                    if val is not None:
                        diversity_epsilon = 0.1 # Small epsilon to ensure diversity
                        blocking_conditions.append(
                            z3.Or(z3_var < val - diversity_epsilon, z3_var > val + diversity_epsilon)
                        )

                if blocking_conditions:
                    opt.add(z3.Or(blocking_conditions))

                opt.pop() # Pop the soft constraints and blocking clause for the next attempt

            elif check_result == z3.unsat:
                print(f"DEBUG_CISB: Z3 UNSAT during CISB population attempt {attempts}. No more diverse solutions found within current constraints.")
                opt.pop()
                break
            else: # z3.unknown
                print(f"DEBUG_CISB: Z3 UNKNOWN during CISB population attempt {attempts}. Retrying with new soft constraints.")
                opt.pop()

        if current_cisb_count < self._cisb_size:
            print(f"DEBUG_CISB: Warning: Could only populate {current_cisb_count} states for CISB after {attempts} attempts.")
        else:
            if self.debug_mode:
                print(f"DEBUG_CISB: CISB population complete with {current_cisb_count} states.")

        unique_samples = set()
        for sample in self._cisb:
            rounded_sample = tuple(sorted([(k, round(v, 6)) for k, v in sample.items()]))
            unique_samples.add(rounded_sample)

        if self.debug_mode:
            print(f"DEBUG_CISB: Number of unique initial states in CISB: {len(unique_samples)}")

            print("DEBUG_CISB: Sample initial states from CISB (first 5):")
            for i, sample in enumerate(self._cisb[:min(5, len(self._cisb))]):
                print(f"  Sample {i+1}: {sample}")
                print(f"  Solved constants for CISB entry: A={sample.get('A')}, b={sample.get('b')}, ep={sample.get('ep')}")
        if not self._cisb:
            print("CRITICAL ERROR: CISB is empty! This environment cannot function correctly without initial states.")
            print("Please check precondition and sampling ranges for satisfiability.")

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

        self.last_episode_history = list(self.state_history)

        self.state_history = []
        self.cumulative_episode_time = 0.0
        self.current_step = 0
        self.last_failing_precondition_formula = None

        if not self._cisb:
            print("DEBUG_RESET: Error: CISB is empty. Generating a single initial state using Z3 as fallback.")
            final_initial_state = {
                'x': 0.0, 'y': 0.0, 'v': 0.0, 'a': 0.0, 'dx': 1.0, 'dy': 0.0,
                'w': 0.0, 'r': 1.0, 'xo': 5.0, 'yo': 0.0, 't': 0.0,
                'ep': 1.0, 'b': 1.0, 'A': 1.0, 'W': 1.0
            }
        else:
            chosen_initial_state_entry = self.np_random.choice(self._cisb)
            final_initial_state = chosen_initial_state_entry.copy()

        for var_name, value in final_initial_state.items():
            if var_name in self.state:
                self.state[var_name] = value
            elif var_name in self.constants:
                self.constants[var_name] = value

        if self.debug_mode:
            print(f"DEBUG_RESET: Initial state: x={self.state.get('x',0):.6f}, y={self.state.get('y',0):.6f}, v={self.state.get('v',0):.6f}, a={self.state.get('a',0):.6f}, w={self.state.get('w',0):.6f}")
            print(f"DEBUG_RESET: Initial constants: A={self.constants.get('A',0):.6f}, b={self.constants.get('b',0):.6f}, ep={self.constants.get('ep',0):.6f}")

        self._add_state_to_history()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    @abstractmethod
    def step(self, action):
        pass

    def render(self, mode="human", variables_to_plot: Optional[List[str]] = None):
        if mode == "human":
            pass
        elif mode == "plot_data":
            return self.state_history
        elif mode == "matplotlib":
            self._plot_state_history(variables_to_plot, history_data=self.last_episode_history)
            return None
        return None

    def _plot_state_history(self, variables_to_plot: Optional[List[str]] = None, history_data: Optional[List[Dict[str, Any]]] = None):
        """
        Generates and displays a matplotlib plot of the state history.
        """
        data_to_plot = history_data if history_data is not None else self.state_history

        if not data_to_plot:
            if self.debug_mode:
                print("DEBUG_PLOT: No state history available to plot.")
            return

        if self.debug_mode:
            print(f"DEBUG_PLOT: state_history length: {len(data_to_plot)}")
        df = pd.DataFrame(data_to_plot)
        if self.debug_mode:
            print(f"DEBUG_PLOT: DataFrame shape: {df.shape}")
            print(f"DEBUG_PLOT: DataFrame head:\n{df.head()}")
            print(f"DEBUG_PLOT: DataFrame tail:\n{df.tail()}")

        x_axis_key = 'episode_time'

        if x_axis_key not in df.columns:
            print(f"DEBUG_PLOT: Error: Time key '{x_axis_key}' not found in state history for plotting. This indicates an issue with state history recording.")
            return

        if variables_to_plot is None:
            default_plot_vars = ['x', 'y', 'v', 'a', 'w']
            variables_to_plot = [col for col in default_plot_vars if col in df.columns]
        else:
            variables_to_plot = [v for v in variables_to_plot if v in df.columns]

        if not variables_to_plot:
            if self.debug_mode:
                print(f"DEBUG_PLOT: No variables selected for plotting or found in DataFrame.")
            return

        num_vars = len(variables_to_plot)
        cols = 2
        rows = (num_vars + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        fig.suptitle('Agent State Evolution Over Time', fontsize=16)

        min_y_range_absolute = 0.05

        for i, var in enumerate(variables_to_plot):
            ax = axes[i]
            ax.plot(df[x_axis_key], df[var], label=var)
            ax.set_xlabel(f"Time ({x_axis_key})")
            ax.set_ylabel(var)
            ax.set_title(f"Evolution of {var}")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            min_time = df[x_axis_key].min()
            max_time = df[x_axis_key].max()
            if max_time <= min_time + 1e-9:
                ax.set_xlim(0, max_time + 0.01)
            else:
                ax.set_xlim(0, max_time)

            min_val = df[var].min()
            max_val = df[var].max()

            data_range = max_val - min_val

            if self.debug_mode:
                print(f"DEBUG_PLOT: Plotting {var}: min_val={min_val:.6f}, max_val={max_val:.6f}, data_range={data_range:.6f}")

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

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def close(self):
        plt.close('all')
        pass