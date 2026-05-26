import gymnasium as gym
import numpy as np
import math
import textwrap
import re
import z3
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod # Import ABC and abstractmethod

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

class _KeYmaeraXEnvBase(gym.Env, ABC): # Inherit from ABC
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
        self.debug_mode = False # Assuming a debug mode flag exists or can be added

        # --- ADDITIONS FOR STATE HISTORY PLOTTING ---
        self.state_history: List[Dict[str, Any]] = []
        self.last_episode_history: List[Dict[str, Any]] = [] # Stores history of the *previous* episode
        self.cumulative_episode_time: float = 0.0 # Tracks time for plotting across an episode
        # --- END ADDITIONS ---

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
                return float(self._evaluate_expression(cond_str)) if self._evaluate_expression(cond_str) else float(self._evaluate_expression(false_val))
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
        # Assuming _control_func_action_config is defined elsewhere in the actual implementation
        # For demonstration, let's assume it's a dummy config for now.
        # This part of the code was not the focus of the original question, so it's kept as is.
        # If _control_func_action_config is not defined, this will raise an error.
        # Example placeholder:
        # self._control_func_action_config = SequenceActionConfig(actions=[])
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

    # --- ADDITIONS FOR STATE HISTORY MANAGEMENT ---
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
        This method handles clearing history, resetting time/step, and adding a default initial state.
        The derived class's reset method should call this first, then set its specific initial state,
        and finally update the history with the actual initial state.
        """
        super().reset(seed=seed) # Call gym.Env's reset for RNG setup

        # Store the history of the just-finished episode
        self.last_episode_history = list(self.state_history)
        # Clear history for the new episode
        self.state_history = []
        self.cumulative_episode_time = 0.0
        self.current_step = 0

        # Set initial state to defaults (e.g., all 0.0). This will be overwritten by the derived class.
        for key in self._state_keys:
            self.state[key] = 0.0 # Example: reset all to 0
        if 't' in self.state:
            self.state['t'] = 0.0

        # Add this default initial state to the history. The derived class will update this entry.
        self._add_state_to_history()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    @abstractmethod # Mark step as abstract
    def step(self, action):
        """
        Takes an action in the environment.
        This method must be implemented by concrete subclasses.
        """
        pass # No implementation needed for abstract method
    # --- END ADDITIONS ---

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
        The x-axis will always use 'episode_time' for cumulative plotting.
        If history_data is provided, it will be used instead of self.state_history.
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