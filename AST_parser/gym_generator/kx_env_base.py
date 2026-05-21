import gymnasium as gym
import numpy as np
import math
import textwrap
import re
import z3
from typing import List, Dict, Any, Optional, Tuple, Union

# Import your helper logic
try:
    from gym_generator.expression_helpers import expand_formula, _node_to_kx
except ImportError:
    from expression_helpers import expand_formula, _node_to_kx

def safe_div(a, b):
    """Prevents ZeroDivisionError by returning a very large number or 0."""
    try:
        if b == 0 or b == 0.0:
            return a * 1e18 if a != 0 else 0.0
        return a / b
    except ZeroDivisionError:
        return a * 1e18 if a != 0 else 0.0

class _KeYmaeraXEnvBase(gym.Env):
    def __init__(self, config_serialized_str: str, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

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

        # 2. Initialize Constants
        self.constants: Dict[str, float] = {}
        for c in self._definitions_config.constants:
            raw_val = getattr(c, 'value', 0.0)
            try:
                self.constants[c.name] = float(raw_val)
            except (ValueError, TypeError):
                self.constants[c.name] = 0.0

        # 3. Initialize Predicates and Functions
        self.definitions_map = {}
        for f in self._definitions_config.functions:
            self.definitions_map[f.name] = self._make_callable_definition(f.parameters, f.body)
        for p in self._definitions_config.predicates:
            self.definitions_map[p.name] = self._make_callable_definition(p.parameters, p.body)

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

    def _make_callable_definition(self, params: List[str], body: str):
        def wrapper(*args):
            local_context = dict(zip(params, args))
            return self._evaluate_expression_raw(body, local_context)
        return wrapper

    def _evaluate_expression_raw(self, expr_str: str, local_context: Dict[str, Any] = None) -> Any:
        proc = self._common_translate(expr_str)
        proc = proc.replace('&&', ' and ').replace('||', ' or ')
        proc = proc.replace('&', ' and ').replace('|', ' or ')
        proc = re.sub(r'!(?!=)', ' not ', proc)

        safe_builtins = {
            'abs': abs, 'min': min, 'max': max, 'math': math, 'np': np,
            'safe_div': safe_div
        }
        
        eval_context = {
            **self.definitions_map, 
            **self.constants, 
            **self.state, 
            '_approx_eq': self._approx_eq
        }
        if local_context:
            eval_context.update(local_context)

        try:
            return eval(proc, {'__builtins__': safe_builtins}, eval_context)
        except ZeroDivisionError:
            return 0.0 

    def _evaluate_expression(self, expr_str: str, is_constraint: bool = False) -> Union[float, bool]:
        try:
            return self._evaluate_expression_raw(expr_str)
        except Exception:
            if is_constraint: return False
            return 0.0

    def _expand_formula(self, formula_str: str, definitions_config=None) -> str:
        config = definitions_config if definitions_config is not None else self._definitions_config
        expanded = expand_formula(formula_str, config)
        if not isinstance(expanded, str):
            expanded = _node_to_kx(expanded)
        return expanded

    def _common_translate(self, formula_str: str) -> str:
        proc = self._expand_formula(formula_str)
        proc = proc.replace('^', '**')
        proc = re.sub(r'(\([^)]+\)|[\w\.]+)\s*/\s*(\([^)]+\)|[\w\.]+)', r'safe_div(\1, \2)', proc)
        proc = proc.replace('!=', ' __NE__ ').replace('<=', ' __LE__ ').replace('>=', ' __GE__ ')
        proc = re.sub(r'(?<![<>=!])=(?![=>])', '==', proc)
        proc = proc.replace('__NE__', '!=').replace('__LE__', '<=').replace('__GE__', '>=')
        return proc

    def _split_top_level_and_local(self, formula: str) -> List[str]:
        formula = formula.strip()
        while formula.startswith('(') and formula.endswith(')'):
            count = 0
            balanced = True
            for i in range(len(formula) - 1):
                if formula[i] == '(': count += 1
                elif formula[i] == ')': count -= 1
                if count == 0:
                    balanced = False
                    break
            if balanced: formula = formula[1:-1].strip()
            else: break

        parts, current, depth, i = [], [], 0, 0
        while i < len(formula):
            char = formula[i]
            if char == '(': depth += 1
            elif char == ')': depth -= 1
            if depth == 0:
                if formula[i:i+2] == '&&':
                    parts.append("".join(current).strip())
                    current, i = [], i + 2
                    continue
                elif char == '&':
                    parts.append("".join(current).strip())
                    current, i = [], i + 1
                    continue
            current.append(char)
            i += 1
        parts.append("".join(current).strip())
        return [p for p in parts if p]

    def _get_predicate_metadata(self, predicate_name: str) -> Dict[str, Any]:
        """Retrieves metadata for a given predicate name from the AST."""
        for pred_config in self._definitions_config.predicates:
            if pred_config.name == predicate_name:
                return pred_config.metadata
        return {}

    def _evaluate_shape_expression(self, shape_expr: str) -> float:
        """Evaluates a shape expression, supporting ternary operators."""
        if not shape_expr: return 0.0
        match = re.match(r'(.+)\s*\?\s*(.+)\s*:\s*(.+)', shape_expr)
        if match:
            cond_str, true_val, false_val = match.groups()
            try:
                return float(self._evaluate_expression(true_val)) if self._evaluate_expression(cond_str) else float(self._evaluate_expression(false_val))
            except: return 0.0
        return float(self._evaluate_expression(shape_expr))

    def _apply_meta_effect(self, metadata: Dict[str, str], reward: float, terminated: bool, truncated: bool, is_failure_case: bool) -> Tuple[float, bool, bool]:
        if is_failure_case:
            reward += float(metadata.get('penalty', 0.0))
            if 'terminate' in metadata.get('on_failure', ''): terminated = True
            if 'truncate' in metadata.get('on_failure', ''): truncated = True
        else:
            reward += float(metadata.get('reward', 0.0))
            if 'terminate' in metadata.get('on_success', ''): terminated = True
        return reward, terminated, truncated

    def _apply_control_action(self, agent_action: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """Executes the control function logic defined in the AST."""
        reward, terminated, applied = self._execute_action_config(self._control_func_action_config, agent_action)
        if applied: self.state['t'] = 0.0
        return reward, terminated, applied

    def _execute_action_config(self, config, agent_action) -> Tuple[float, bool, bool]:
        """Recursive executor for Choice, Sequence, and Assignment actions."""
        from pyast_config import (ChoiceActionConfig, SequenceActionConfig, DeterministicAssignmentActionConfig, 
                                 NondeterministicAssignmentActionConfig, ActionConstraintConfig)
        
        if isinstance(config, ChoiceActionConfig):
            choice_key = f"choice_{config.level_name}"
            idx = agent_action.get(choice_key, 0)
            return self._execute_action_config(config.choices[idx], agent_action)

        elif isinstance(config, SequenceActionConfig):
            total_r = 0.0
            for act in config.actions:
                r, t, s = self._execute_action_config(act, agent_action)
                total_r += r
                if t: return total_r, True, s
                if not s: return total_r, False, False
            return total_r, False, True

        elif isinstance(config, DeterministicAssignmentActionConfig):
            self.state[config.variable] = float(self._evaluate_expression(config.value))
            return 0.0, False, True

        elif isinstance(config, NondeterministicAssignmentActionConfig):
            val = agent_action.get(f"assign_{config.variable}", self.state[config.variable])
            self.state[config.variable] = float(val)
            return 0.0, False, True

        elif isinstance(config, ActionConstraintConfig):
            if not self._evaluate_expression(config.formula, is_constraint=True):
                return -100.0, True, False # Penalty for control violation
            return 0.0, False, True
        
        return 0.0, False, True

    def _solve_initial_state_with_z3(self, state_vars, precondition_formula: str) -> Optional[Dict[str, float]]:
        solver = z3.Solver()
        z3_vars = {v.name: z3.Real(v.name) for v in state_vars}
        for name, val in self.constants.items():
            z3_vars[name] = z3.Real(name)
            solver.add(z3_vars[name] == val)

        expanded = self._expand_formula(precondition_formula)
        proc = expanded.replace('^', '**').replace('&&', ' & ').replace('||', ' | ').replace(' and ', ' & ').replace(' or ', ' | ')
        proc = re.sub(r'!(?!=)', ' ~ ', proc)
        proc = re.sub(r'(?<![<>=!])=(?![=>])', '==', proc)
        z3_env = {'abs': lambda x: z3.If(x >= 0, x, -x), **z3_vars}

        try:
            z3_expr = eval(proc, {'__builtins__': None}, z3_env)
            solver.add(z3_expr)
            if solver.check() == z3.sat:
                model = solver.model()
                return {v.name: float(model[z3_vars[v.name]].as_decimal(10).replace('?', '')) if z3.is_algebraic_number(model[z3_vars[v.name]]) else float(model[z3_vars[v.name]].as_long()) if model[z3_vars[v.name]] is not None else 0.0 for v in state_vars}
        except Exception:
            pass
        return None

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