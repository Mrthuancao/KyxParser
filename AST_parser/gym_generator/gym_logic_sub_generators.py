# gym_logic_sub_generators.py
# This module contains helper functions that generate specific pieces of logic for the Gym environment based on the AST configuration.
# These functions are called by the main gym generator to construct the full environment class.
import textwrap
import re
from typing import List, Dict, Any, Tuple, Optional

# Assuming pyast_config.py is in the same directory or importable
from pyast_config import (
    DefinitionsConfig, VariablesConfig, ProblemConfig,
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
from gym_generator.expression_helpers import format_kx_expr_for_python_lambda_body_generator
from gym_generator.ast_serialization import serialize_ast_node

def generate_action_space_and_logic(hps: List[HybridProgramConfig]) -> Tuple[str, ActionType, Dict[str, str]]:
    """
    Generates the Gymnasium action space definition and returns the actual AST object
    for the control function's action configuration.
    """
    control_func_config = next((hp for hp in hps if isinstance(hp, ControlFuncConfig)), None)
    if not control_func_config or not control_func_config.action_space_config:
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
        indented_items = textwrap.indent("\n".join(action_space_str_parts), ' ' * 8)
        action_space_definition = f"self.action_space = spaces.Dict({{\n{indented_items}\n        }})"

    return action_space_definition, control_func_config.action_space_config, action_space_keys


def generate_ode_solver_logic(hps: List[HybridProgramConfig], state_variable_names: List[str],
                               all_constants: List[str], all_function_names: List[str], all_predicate_names: List[str]) -> Tuple[str, str, Dict[str, int], Dict[int, str]]:
    """
    Generates the Python code for the ODE function and event functions
    for scipy.integrate.solve_ivp.
    """
    step_func_config = next((hp for hp in hps if isinstance(hp, StepFuncConfig)), None)
    if not step_func_config:
        return "", "", {}, {}

    ode_state_map = {var_name: i for i, var_name in enumerate(state_variable_names)}
    ode_state_idx_map = {i: var_name for i, var_name in enumerate(state_variable_names)}

    ode_func_lines = []
    ode_func_lines.append("def ode_func(t_val, y_array):")
    ode_func_lines.append("    # Map y array to named variables for easier access")
    for var_name, idx in ode_state_map.items():
        ode_func_lines.append(f"    {var_name} = y_array[{idx}]")
    ode_func_lines.append("")

    ode_func_lines.append("    # Calculate derivatives")
    derivatives = ['0.0'] * len(state_variable_names)
    ode_params_in_scope = list(ode_state_map.keys()) + ['t_val']

    ode_lhs_vars = {ode.variable for ode in step_func_config.odes}

    for ode in step_func_config.odes:
        idx = ode_state_map.get(ode.variable)
        if idx is not None:
            equation_single_line = ode.equation.replace('\n', ' ').strip()
            formatted_equation = format_kx_expr_for_python_lambda_body_generator(
                equation_single_line, ode_params_in_scope, all_constants, state_variable_names, all_function_names, all_predicate_names
            )
            derivatives[idx] = formatted_equation
        else:
            print(f"Warning: ODE for variable '{ode.variable}' not found in state_variable_names. Skipping.")

    for var_name in state_variable_names:
        if var_name not in ode_lhs_vars:
            idx = ode_state_map.get(var_name)
            if idx is not None:
                derivatives[idx] = '0.0'

    if 'w' in ode_lhs_vars:
        w_idx = ode_state_map['w']
        original_w_ode_equation = next((ode.equation for ode in step_func_config.odes if ode.variable == 'w'), '')
        if 'a/r' in original_w_ode_equation:
            derivatives[w_idx] = f"({derivatives[w_idx]}) if r != 0 else 0.0"

    ode_func_lines.append("    dydt = [")
    for i, deriv_expr in enumerate(derivatives):
        ode_func_lines.append(f"        {deriv_expr}, # d{ode_state_idx_map[i]}/dt")
    ode_func_lines.append("    ]")
    ode_func_lines.append("    return dydt")

    ode_func_code = "\n".join(ode_func_lines)

    ode_event_lines = []

    ep_val = "self.constants['ep']"
    t_idx = ode_state_map.get('t')
    if t_idx is not None:
        ode_event_lines.append(textwrap.dedent(f"""
    def event_t_max(t_val, y_array):
        return {ep_val} - y_array[{t_idx}]
    event_t_max.terminal = True
    event_t_max.direction = -1
    """).strip())

    v_idx = ode_state_map.get('v')
    a_idx = ode_state_map.get('a')
    if v_idx is not None and a_idx is not None:
        if ode_event_lines: ode_event_lines.append("")
        ode_event_lines.append(textwrap.dedent(f"""
    def event_v_min(t_val, y_array):
        v_val = y_array[{v_idx}]
        a_val = y_array[{a_idx}]

        if v_val == 0 and a_val > 0:
            return 1.0
        return v_val
    event_v_min.terminal = True
    event_v_min.direction = -1
    """).strip())

    if ode_event_lines: ode_event_lines.append("")
    ode_event_lines.append("ode_events = [")
    if t_idx is not None:
        ode_event_lines.append("    event_t_max,")
    if v_idx is not None:
        ode_event_lines.append("    event_v_min,")
    ode_event_lines.append("]")

    ode_event_code = "\n".join(ode_event_lines)

    return ode_func_code, ode_event_code, ode_state_map, ode_state_idx_map