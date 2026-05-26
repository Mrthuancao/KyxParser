# main_generator.py
# This module generates the main Gymnasium environment class code as a string based on the provided GymEnvironmentConfig AST.
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
from gym_generator.kx_env_base import _KeYmaeraXEnvBase # Assuming kx_env_base.py is in the same directory or importable path

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
        params_str = ", ".join([p[1] if isinstance(p, (tuple, list)) else p for p in func.parameters])
        param_names_list = [p[1] if isinstance(p, (tuple, list)) else p for p in func.parameters]
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
        params_str = ", ".join([p[1] if isinstance(p, (tuple, list)) else p for p in pred.parameters])
        param_names_list = [p[1] if isinstance(p, (tuple, list)) else p for p in pred.parameters]
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
    indented_reset_block = textwrap.indent("\n".join(reset_initial_state_code_lines), '    ') 

    # Serialize the entire config object to pass to the base class's __init__
    full_config_serialized_str = serialize_ast_node(config, indent_level=2)

    # Imports for typing hints
    typing_imports = ["Dict", "Any", "Tuple", "List", "Optional", "Union"]
    typing_import_statement = f"from typing import ({', '.join(typing_imports)})"


     # --- Main Environment Class Structure ---
    env_code = textwrap.dedent(f"""# {env_name_for_class}.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import re
{typing_import_statement}
from gym_generator.kx_env_base import _KeYmaeraXEnvBase

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

class {env_name_for_class}(_KeYmaeraXEnvBase):
    metadata = {{"render_modes": ["human"], "render_fps": 30}}

    def __init__(self, render_mode=None):
        self.constants = {{}}
        self.functions = {{}}
        self.predicates = {{}}

{indented_constants_block}
{indented_functions_block}
{indented_predicates_block}

        super().__init__(config_serialized_str='''{full_config_serialized_str}''', render_mode=render_mode)

        self._state_keys = {state_variable_names}
        self.state = {{key: 0.0 for key in self._state_keys}}

        low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)

{textwrap.indent(action_space_definition, '        ')}
        self._action_space_keys = {list(action_space_keys.keys())}
        self._control_func_action_config = {control_func_ast_code_str}

        self._ode_state_map = {ode_state_map}
        self._ode_state_idx_map = {ode_state_idx_map}

        _ode_func_str_local = '''{ode_func_code}'''
        _ode_events_str_local = '''{ode_event_code}'''
        self._ode_func_compiled = self._compile_ode_func(_ode_func_str_local)
        self._ode_events_compiled = self._compile_ode_events(_ode_events_str_local)

        # Call the derived class's reset method to set the initial state and populate history.
        self.reset() 

    def reset(self, seed=None, options=None):
        # Call the base class's reset method first. This handles:
        # 1. Storing the previous episode's history into self.last_episode_history.
        # 2. Clearing self.state_history for the new episode.
        # 3. Resetting self.cumulative_episode_time to 0.0.
        # 4. Resetting self.current_step to 0.
        # 5. Setting self.state to default values (e.g., all 0.0).
        # 6. Adding this default state to self.state_history.
        observation, info = super().reset(seed=seed, options=options) 
        
        # Now, apply the specific initial state logic from the generator.
        # This will overwrite the default state set by the base class.
{indented_reset_block}

        # After applying the specific initial state (Z3/heuristics),
        # self.state is now correctly initialized.
        # The base class's reset already added a default initial state to self.state_history.
        # We need to update that entry with the actual initial state.
        # self.state_history should never be empty after super().reset()
        self.state_history[-1] = self._get_state_for_history() # Update the last entry with the actual initial state

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1 
        reward, terminated, truncated = 0.0, False, False
        info = self._get_info()

        # --- Initial Loop Invariant Check ---
        expanded_loop_inv = self._expand_formula(self._problem_config.loop_execution.loop_invariant_formula)
        for conj in self._split_top_level_and_local(expanded_loop_inv):
            if not self._evaluate_expression(conj, is_constraint=True):
                m = re.match(r'^\\s*(\\w+)', conj)
                meta = self._get_predicate_metadata(m.group(1)) if m else {{}}
                reward, terminated, truncated = self._apply_meta_effect(meta or {{'penalty': -100}}, reward, terminated, truncated, True)
                if terminated or truncated:
                    # If termination/truncation happens here, it's based on the state *before* this step's dynamics.
                    # The history should already contain the state from the end of the previous step.
                    # No new history entry needed unless state was modified here (which it shouldn't be for invariant check).
                    return self._get_obs(), reward, terminated, truncated, info

        # --- 1. Control ---
        c_reward, c_terminated, c_applied = self._apply_control_action(action)
        reward += c_reward
        if c_terminated:
            # If terminated here, the state *might* have changed due to _apply_control_action.
            # We should record this state. Control actions are typically instantaneous.
            self._add_state_to_history() # Record state after control action, before ODE (if it led to termination)
            return self._get_obs(), reward, True, False, info

        # --- 2. Dynamics ---
        y0 = np.array([self.state[k] for k in self._ode_state_map.keys()])
        sol = solve_ivp(self._ode_func_compiled, (0, self.constants['ep']), y0, events=self._ode_events_compiled, rtol=1e-6)
        
        final_y = sol.y[:, -1]
        for i, name in self._ode_state_idx_map.items(): self.state[name] = final_y[i]

        # Update self.state['t'] with the actual time elapsed during this ODE step.
        # This is crucial for `cumulative_episode_time`.
        # The `sol.t[-1]` is the actual time reached by the solver, which might be less than 'ep' if an event occurred.
        self.state['t'] = sol.t[-1] 

        # --- History Management for step ---
        # Accumulate time *after* self.state['t'] has been updated by the ODE dynamics.
        self.cumulative_episode_time += self.state.get('t', 0.0) 
        self._add_state_to_history() # Record state after ODE dynamics (this is the primary history point for a step)

        if sol.status == 1: # Event triggered (e.g. v < 0)
            reward, terminated, truncated = self._apply_meta_effect({{'penalty': -100}}, reward, False, True, True)
            # History already added above, just after ODE dynamics.
            return self._get_obs(), reward, False, True, info

        # --- 3. Post-Dynamics Invariant & Shaping ---
        for conj in self._split_top_level_and_local(expanded_loop_inv):
            if not self._evaluate_expression(conj, is_constraint=True):
                # History already added above, after ODE dynamics.
                return self._get_obs(), reward - 100, True, False, info

        # Reward Shaping
        for p_config in self._definitions_config.predicates:
            if p_config.metadata.get('type') == 'progress' and 'shape' in p_config.metadata:
                reward += self._evaluate_shape_expression(p_config.metadata['shape'])

        # --- 4. Postcondition ---
        if self._evaluate_expression(self._problem_config.postcondition.formula, is_constraint=True):
            meta = self._get_predicate_metadata('ensure')
            reward, terminated, truncated = self._apply_meta_effect(meta, reward, terminated, truncated, False)

        if self.current_step >= self.max_episode_steps: truncated = True

        # No further state changes for history after this point, so the history added after ODE is sufficient.

        return self._get_obs(), reward, terminated, truncated, info
""")
    return env_code