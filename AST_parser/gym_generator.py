import textwrap
from typing import List, Dict, Any, Optional, Tuple, Union
import re

# Assuming ast_config.py is in the same directory and contains the ASTConfigNode classes
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

# --- Helper Functions for Expression Formatting ---

def _format_kx_expr_for_python_lambda_body(expr_str: str, lambda_param_names: List[str], all_constants: List[str], all_state_vars: List[str], all_functions: List[str], all_predicates: List[str]) -> str:
    """
    Converts a KeYmaera X expression string into a Python expression string
    suitable for direct use in a lambda body. Handles operator conversions and
    prefixes identifiers with their correct scope (self.constants, self.state, etc.).
    """
    expr_py = expr_str.strip()

    # 1. Basic syntax replacements (power, abs) - Highest precedence
    expr_py = expr_py.replace('^', '**').replace('abs(', 'np.abs(')

    # 2. Replace single '=' with '==' for equality comparisons.
    # This is done here for lambda bodies, as they are directly evaluated Python code.
    # For expressions passed to _evaluate_expression, that function handles it.
    expr_py = re.sub(r'(?<![<>=!])=(?![=])', '==', expr_py)

    # 3. Handle logical operators (&, |) - Convert to Python 'and', 'or'
    expr_py = expr_py.replace('&', 'and').replace('|', 'or')

    # 4. Handle KeYmaera X implication '->' (A -> B becomes (not (A) or (B))) - Lowest precedence
    while True:
        # FIX: Corrected regex backreferences from ,  to \1, \2
        new_expr_py = re.sub(r'\((.*?)\s*->\s*(.*?)\)', r'((not \1) or \2)', expr_py)
        if new_expr_py == expr_py: # No more (A -> B) patterns found
            break
        expr_py = new_expr_py
    
    # 5. Identify and replace identifiers with their correct scope prefix
    # Build a list of all known identifiers (excluding lambda params)
    # Order matters: longer names first, then constants, then state, then functions/preds
    all_env_identifiers = sorted(
        list(set(all_constants + all_state_vars + all_functions + all_predicates + ['np'])), # Added 'np' to ensure it's not prefixed
        key=len, reverse=True # Longest names first
    )

    # Create a regex pattern to match these identifiers as whole words
    # Ensure lambda_param_names are not replaced
    identifiers_to_replace = [
        ident for ident in all_env_identifiers if ident not in lambda_param_names
    ]
    
    if identifiers_to_replace: # Only apply if there are identifiers to replace
        # Escape identifiers for regex and join them with '|'
        # Add lookarounds to ensure whole word matching for identifiers
        pattern_str = r'\b(' + '|'.join(re.escape(ident) for ident in identifiers_to_replace) + r')\b'
        pattern = re.compile(pattern_str)

        def replacer(match):
            ident = match.group(1)
            if ident in all_constants:
                return f"self.constants['{ident}']"
            elif ident in all_state_vars:
                return f"self.state['{ident}']"
            elif ident in all_functions:
                # For functions, we assume they are called. We just prefix the name.
                # The actual arguments will be part of the original expr_str.
                return f"self.functions['{ident}']"
            elif ident in all_predicates:
                return f"self.predicates['{ident}']"
            elif ident == 'np': # 'np' should not be prefixed
                return 'np'
            return ident # Should not be reached if pattern is correct

        # Perform replacements
        expr_py = pattern.sub(replacer, expr_py)
    
    return expr_py

def _format_kx_expr_for_python_eval_context(expr_str: str) -> str:
    """
    Converts a KeYmaera X expression string into a Python expression string
    suitable for evaluation within a general context (like _evaluate_expression).
    Handles only basic operator conversions (power, abs) that are not handled
    by _evaluate_expression's regex.
    """
    expr_py = expr_str.strip()

    # 1. Basic syntax replacements (power, abs)
    expr_py = expr_py.replace('^', '**').replace('abs(', 'np.abs(')
    
    # Other conversions (e.g., '=', '&', '|', '->') are handled by _evaluate_expression
    # itself when it processes the string at runtime.
    return expr_py

def _expand_formula(formula_str: str, definitions: DefinitionsConfig) -> str:
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

# --- End Helper Functions ---


def _ast_to_python_code_string(ast_node: ActionType, indent_level: int = 0,
                                all_constants: List[str] = None, all_state_vars: List[str] = None,
                                all_functions: List[str] = None, all_predicates: List[str] = None) -> str:
    """
    Recursively converts an ActionType AST node into its Python constructor call string.
    This is used to embed the control function's action configuration directly into the
    generated environment's __init__ method.
    """
    # Initialize lists if not provided (for top-level call)
    if all_constants is None: all_constants = []
    if all_state_vars is None: all_state_vars = []
    if all_functions is None: all_functions = []
    if all_predicates is None: all_predicates = []

    indent_str = "    " * indent_level
    next_indent_str = "    " * (indent_level + 1)

    if isinstance(ast_node, ChoiceActionConfig):
        choices_str = ",\n".join([
            f"{next_indent_str}{_ast_to_python_code_string(choice, indent_level + 1, all_constants, all_state_vars, all_functions, all_predicates)}"
            for choice in ast_node.choices
        ])
        return (
            f"ChoiceActionConfig(level_name='{ast_node.level_name}', choices=[\n"
            f"{choices_str}\n"
            f"{indent_str}])"
        )
    elif isinstance(ast_node, SequenceActionConfig):
        actions_str = ",\n".join([
            f"{next_indent_str}{_ast_to_python_code_string(action, indent_level + 1, all_constants, all_state_vars, all_functions, all_predicates)}"
            for action in ast_node.actions
        ])
        return (
            f"SequenceActionConfig(actions=[\n"
            f"{actions_str}\n"
            f"{indent_str}])"
        )
    elif isinstance(ast_node, DeterministicAssignmentActionConfig):
        # Format the value string for direct evaluation by _evaluate_expression
        # _format_kx_expr_for_python_eval_context ensures it's ready for eval()
        formatted_value = _format_kx_expr_for_python_eval_context(ast_node.value)
        return f"DeterministicAssignmentActionConfig(variable='{ast_node.variable}', value='{formatted_value}')"
    elif isinstance(ast_node, NondeterministicAssignmentActionConfig):
        return f"NondeterministicAssignmentActionConfig(variable='{ast_node.variable}')"
    elif isinstance(ast_node, ActionConstraintConfig):
        # Ensure formula is single-line for string literal
        formula_single_line = ast_node.formula.replace('\n', ' ').strip()
        # Format the formula string for direct evaluation by _evaluate_expression
        formatted_formula = _format_kx_expr_for_python_eval_context(formula_single_line)
        return f"ActionConstraintConfig(formula='{formatted_formula}')"
    elif isinstance(ast_node, CallHybridProgramConfig):
        return f"CallHybridProgramConfig(hp_name='{ast_node.hp_name}')"
    elif isinstance(ast_node, HybridProgramStepConfig):
        # Ensure invariant formula is single-line for string literal
        invariant_single_line = ast_node.invariant_formula.replace('\n', ' ').strip()
        # Format the invariant formula string for direct evaluation by _evaluate_expression
        formatted_invariant = _format_kx_expr_for_python_eval_context(invariant_single_line)
        return f"HybridProgramStepConfig(hp_name='{ast_node.hp_name}', invariant_formula='{formatted_invariant}')"
    else:
        raise ValueError(f"Unknown AST node type: {type(ast_node)}")


def generate_gym_environment_code(config: GymEnvironmentConfig) -> str:
    """
    Generates Python code for a Gymnasium environment based on the provided
    GymEnvironmentConfig AST.
    """
    original_env_name = config.env_name
    
    # --- SANITIZE ENV_NAME FOR PYTHON CLASS IDENTIFIER ---
    # 1. Replace non-alphanumeric characters (excluding underscore) with underscore
    sanitized_env_name = re.sub(r'[^a-zA-Z0-9_]', '_', original_env_name)
    # 2. Replace multiple underscores with a single underscore and strip leading/trailing underscores
    sanitized_env_name = re.sub(r'_+', '_', sanitized_env_name).strip('_')
    
    # 3. Ensure it doesn't start with a digit
    if sanitized_env_name and sanitized_env_name[0].isdigit():
        sanitized_env_name = '_' + sanitized_env_name
    
    # 4. Fallback if the name becomes empty after sanitization
    if not sanitized_env_name:
        sanitized_env_name = "GeneratedEnv" 
    
    env_name_for_class = sanitized_env_name # Use this for the class definition
    # ---------------------------------------------------

    # Collect all variable names for observation space and ODE state vector
    state_variable_names = [v.name for v in config.variables.state_variables]
    all_constant_names = [c.name for c in config.definitions.constants]
    all_function_names = [f.name for f in config.definitions.functions]
    all_predicate_names = [p.name for p in config.definitions.predicates]

    # --- Generate Constants, Functions, Predicates ---
    constants_init_code_lines = []
    for const in config.definitions.constants:
        # Check if the value is a simple numeric literal (integer or float) or boolean
        # which doesn't need _format_kx_expr_for_python_lambda_body
        if const.value is not None and (re.fullmatch(r'^-?\d+(\.\d+)?$', const.value) or re.fullmatch(r'^(True|False)$', const.value)):
            formatted_value = const.value
        elif const.value is not None:
             # It's an expression involving other constants/variables, so format it
            formatted_value = _format_kx_expr_for_python_lambda_body(
                const.value, [], all_constant_names, state_variable_names, all_function_names, all_predicate_names
            )
        else: # const.value is None (not provided in AST)
            # Apply heuristic for critical constants
            if const.name in ['ep', 'W', 'V', 'A', 'b', 'Ud']:
                formatted_value = '1.0'
                if const.name == 'Ud':
                    formatted_value = '0.1'
            else:
                formatted_value = '0.0' # Default for other unspecified constants

        constants_init_code_lines.append(f"self.constants['{const.name}'] = {formatted_value}")
    # Pre-indent this block
    indented_constants_block = textwrap.indent("\n".join(constants_init_code_lines), '        ')

    functions_init_code_lines = []
    for func in config.definitions.functions:
        params_str = ", ".join([p_name for _, p_name in func.parameters])
        param_names_list = [p_name for _, p_name in func.parameters]
        # Ensure function body is a single line for embedding in string literal
        body_single_line = func.body.replace('\n', ' ').strip()
        
        # Format the body for direct Python lambda evaluation
        formatted_body = _format_kx_expr_for_python_lambda_body(
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
        
        # Format the body for direct Python lambda evaluation
        formatted_body = _format_kx_expr_for_python_lambda_body(
            body_single_line, param_names_list, all_constant_names, state_variable_names, all_function_names, all_predicate_names
        )
        predicates_init_code_lines.append(f"self.predicates['{pred.name}'] = lambda {params_str}: {formatted_body}")
    # Pre-indent this block
    indented_predicates_block = textwrap.indent("\n".join(predicates_init_code_lines), '        ')

    # --- Generate Action Space ---
    action_space_definition, control_func_ast_object, action_space_keys = _generate_action_space_and_logic(config.definitions.hybrid_programs)
    
    # Convert the control_func_ast_object to its Python code string representation
    control_func_ast_code_str = _ast_to_python_code_string(
        control_func_ast_object, indent_level=3,
        all_constants=all_constant_names, all_state_vars=state_variable_names,
        all_functions=all_function_names, all_predicates=all_predicate_names
    ) # Indent for __init__ method

    # --- Generate ODEs and Evolution Domain ---
    ode_func_code, ode_event_code, ode_state_map, ode_state_idx_map = _generate_ode_solver_logic(
        config.definitions.hybrid_programs, state_variable_names,
        all_constant_names, all_function_names, all_predicate_names
    )

    # --- Generate Initial State (reset logic) ---
    # Pass definitions for predicate expansion
    reset_initial_state_code_lines = _generate_reset_logic(config.variables.state_variables, config.problem.precondition, config.definitions)
    # Pre-indent this block
    indented_reset_block = textwrap.indent("\n".join(reset_initial_state_code_lines), '    ')

    # --- Generate Problem Logic (Postcondition, Loop Invariants) ---
    # Ensure formula is single-line for Python eval
    postcondition_formula_py_raw = config.problem.postcondition.formula.replace('\n', ' ').strip()
    postcondition_formula_py = _format_kx_expr_for_python_eval_context(postcondition_formula_py_raw)

    loop_invariant_formula_py_raw = config.problem.loop_execution.loop_invariant_formula.replace('\n', ' ').strip()
    loop_invariant_formula_py = _format_kx_expr_for_python_eval_context(loop_invariant_formula_py_raw)

    # Imports for AST classes needed for _control_func_action_config
    ast_imports = [
        "ActionType", 
        "ChoiceActionConfig", "SequenceActionConfig", "DeterministicAssignmentActionConfig",
        "NondeterministicAssignmentActionConfig", "ActionConstraintConfig", "CallHybridProgramConfig",
        "HybridProgramStepConfig"
    ]
    ast_import_statement = f"from pyast_config import ({', '.join(ast_imports)})"

    # Imports for typing hints
    typing_imports = ["Dict", "Any", "Tuple"]
    typing_import_statement = f"from typing import ({', '.join(typing_imports)})"


    # --- Main Environment Class Structure ---
    env_code = textwrap.dedent(f"""import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import math # For math functions like sqrt, sin, cos, etc.
import textwrap
import re # For regex replacements in _evaluate_expression
{typing_import_statement} # Import typing hints
{ast_import_statement} # Import AST classes

class {env_name_for_class}(gym.Env):
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
{textwrap.indent(action_space_definition, '        ')}
        self._action_space_keys = {list(action_space_keys.keys())}
        self._control_func_action_config = {control_func_ast_code_str} # Stored for step logic

        # --- ODE Solver Setup (from stepFunc) ---
        # Mapping from state variable names to their index in the ODE solver's y array
        self._ode_state_map = {ode_state_map}
        self._ode_state_idx_map = {ode_state_idx_map} # Map index to name
        
        self._ode_func_str = '''
{textwrap.indent(ode_func_code, '            ')}
        '''
        self._ode_events_str = '''
{textwrap.indent(ode_event_code, '            ')}
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

    def _compile_ode_func(self, func_str):
        # Create a dictionary for globals that includes 'self'
        exec_globals = {{'np': np, 'math': math, 'self': self}} # 'self' in globals
        exec_locals = {{}} # No special locals needed, self is in globals
        # Dedent the string before execution, as it was indented for the triple-quote literal
        exec(textwrap.dedent(func_str), exec_globals, exec_locals)
        return exec_locals['ode_func']

    def _compile_ode_events(self, events_str):
        if not events_str.strip():
            return []
        exec_globals = {{'np': np, 'math': math, 'self': self}} # 'self' in globals
        exec_locals = {{}} # No special locals needed, self is in globals
        # Dedent the string before execution
        exec(textwrap.dedent(events_str), exec_globals, exec_locals)
        return exec_locals['ode_events']

    def _get_obs(self):
        return np.array([self.state[key] for key in self._state_keys], dtype=np.float32)

    def _get_info(self):
        return self.state.copy() # Return a copy of the current state dictionary

    def _evaluate_expression(self, expr_str, local_vars=None, is_constraint=False):
        
        # Safe evaluation of expressions using environment's context.
        # If is_constraint is True, returns False on evaluation error.
        
        if local_vars is None:
            local_vars = {{}}
        
        # Strip KeYmaera X constraint prefix '?' if present
        if expr_str.startswith('?'):
            expr_str = expr_str[1:].strip()

        # Combine all available variables for evaluation context
        eval_globals = {{
            'np': np, 'math': math,
            **self.constants,
            **self.functions, # Include functions for general expressions
            **self.predicates  # Include predicates for general expressions
        }}
        eval_locals = {{
            **self.state,
            **local_vars
        }}
        
        # Apply the same transformations as in _format_kx_expr_for_python_lambda_body
        # to ensure consistency when evaluating expressions at runtime.
        expr_py = expr_str.strip()
        
        # 1. Basic syntax replacements (power, abs)
        expr_py = expr_py.replace('^', '**').replace('abs(', 'np.abs(')
        
        # 2. Replace single '=' with '==' for equality comparisons.
        expr_py = re.sub(r'(?<![<>=!])=(?![=])', '==', expr_py)

        # 3. Handle logical operators (&, |)
        expr_py = expr_py.replace('&', 'and').replace('|', 'or')
        
        # 4. Handle KeYmaera X implication '->' (A -> B becomes (not (A) or (B)))
        while True:
            # FIX: Corrected regex backreferences from ,  to \1, \2
            new_expr_py = re.sub(r'\((.*?)\s*->\s*(.*?)\)', r'((not \1) or \2)', expr_py)
            if new_expr_py == expr_py:
                break
            expr_py = new_expr_py
        
        try:
            return eval(expr_py, eval_globals, eval_locals)
        except Exception as e:
            print(f"Error evaluating expression '{{expr_str}}': {{e}}")
            # print(f"Eval globals: {{eval_globals}}") # Too verbose
            # print(f"Eval locals: {{eval_locals}}") # Too verbose
            if is_constraint:
                return False # For constraints, return False on error
            raise # Re-raise for non-constraint expressions

    def _execute_action_config(self, action_config: ActionType, agent_action: Dict[str, Any]) -> Tuple[float, bool, bool]:
        \"\"\"
        Recursively executes a given action configuration (Choice, Sequence, Assignment, Constraint).
        Returns (reward, terminated, applied_successfully).
        \"\"\"
        reward = 0.0
        terminated = False
        applied_successfully = True

        if isinstance(action_config, ChoiceActionConfig):
            choice_key = f"choice_{{action_config.level_name}}"
            if choice_key not in agent_action:
                print(f"Error: Choice key '{{{{choice_key}}}}' missing from agent action.")
                reward -= 200.0
                terminated = True
                applied_successfully = False
                return reward, terminated, applied_successfully

            choice_idx = agent_action[choice_key]
            if not (0 <= choice_idx < len(action_config.choices)):
                print(f"Error: Invalid choice index {{{{choice_idx}}}} for '{{{{choice_key}}}}'.")
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
                assigned_value = self._evaluate_expression(action_config.value)
                self.state[action_config.variable] = assigned_value
            except Exception as e:
                print(f"Error evaluating deterministic assignment for {{{{action_config.variable}}}}='{{{{action_config.value}}}}': {{{{e}}}}")
                reward -= 100.0
                terminated = True
                applied_successfully = False
            return reward, terminated, applied_successfully

        elif isinstance(action_config, NondeterministicAssignmentActionConfig):
            assign_key = f"assign_{{action_config.variable}}"
            if assign_key in agent_action:
                self.state[action_config.variable] = agent_action[assign_key]
            else:
                print(f"Warning: Nondeterministic assignment for '{{{{action_config.variable}}}}' (key '{{{{assign_key}}}}') not found in agent action. Defaulting to current value.")
                # This might indicate an incomplete action by the agent, penalize.
                reward -= 50.0 
                # Do not terminate here, as it might be a valid choice to not assign.
                # However, if the problem requires it, this might need to be `terminated = True`.
                # For now, we assume it's okay to skip, but with a penalty.
            return reward, terminated, applied_successfully

        elif isinstance(action_config, ActionConstraintConfig):
            constraint_met = self._evaluate_expression(action_config.formula, is_constraint=True)
            if not constraint_met:
                # print(f"Constraint violated: {{{{action_config.formula}}}}")
                reward -= 100.0 # Penalty for invalid action
                terminated = True
                applied_successfully = False
            return reward, terminated, applied_successfully

        elif isinstance(action_config, CallHybridProgramConfig):
            # This generator assumes controlFunc is the primary action selection.
            # If CallHybridProgramConfig appears here, it's an unsupported nested HP call within controlFunc.
            print(f"Warning: CallHybridProgramConfig '{{{{action_config.hp_name}}}}' encountered in _execute_action_config. Not fully supported yet.")
            reward -= 50.0
            applied_successfully = False
            return reward, terminated, applied_successfully

        else:
            print(f"Error: Unknown action config type: {{{{type(action_config)}}}}")
            reward -= 200.0
            terminated = True
            applied_successfully = False
            return reward, terminated, applied_successfully

    def _apply_control_action(self, action):
        \"\"\"
        Applies the agent's chosen action by traversing the control function's AST.
        \"\"\"
        reward, terminated, applied_successfully = self._execute_action_config(self._control_func_action_config, action)
        
        # Apply t := 0; after controlFunc, as specified in the problem loop body
        if applied_successfully: # Only reset t if the action was valid and applied
            self.state['t'] = 0.0
        
        return reward, terminated, applied_successfully

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # --- Initial State (based on Problem Precondition and defaults) ---
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
        
        ep_val = self.constants['ep'] # Get the 'ep' constant value
        t_span = (0, ep_val) # Integrate up to time limit 'ep'

        if ep_val <= 0:
            # If ep is non-positive, no meaningful time evolution occurs.
            # This might indicate an invalid control action or problem setup.
            self.state['t'] = 0.0 # Time does not advance or is reset
            if ep_val < 0:
                print(f"Warning: 'ep' constant is negative ({{ep_val}}). Skipping ODE integration and penalizing.")
                reward -= 100.0 # Penalty for invalid time duration
                terminated = True
            # else (ep_val == 0): No time passes, but not necessarily an error if intended.
        else:
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
                    if self.state.get('v', 0.0) < 0: # Use .get with default for safety
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
        postcondition_satisfied = self._evaluate_expression('{postcondition_formula_py}', is_constraint=True)
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
""")
    return env_code

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
        # DeterministicAssignmentActionConfig, ActionConstraintConfig, CallHybridProgramConfig
        # do not add to the action space directly, as they are not agent inputs.

    _recursive_build_action_space(control_func_config.action_space_config)

    action_space_str_parts = []
    for key, val_str in action_space_dict.items():
        action_space_str_parts.append(f"'{key}': {val_str},")
    
    if not action_space_str_parts:
        action_space_definition = "self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)"
    else:
        # Indent the dictionary items correctly
        indented_items = textwrap.indent("\n".join(action_space_str_parts), ' ' * 8)
        action_space_definition = f"self.action_space = spaces.Dict({{\n{indented_items}\n        }})" # Adjusted indentation for closing brace

    # Return the actual AST object for the control function
    return action_space_definition, control_func_config.action_space_config, action_space_keys

def _generate_ode_solver_logic(hps: List[HybridProgramConfig], state_variable_names: List[str],
                               all_constants: List[str], all_function_names: List[str], all_predicate_names: List[str]):
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
            # Format the ODE equation for direct Python evaluation within the ODE function's scope
            # The ODE function's local scope will have state variables as direct names (e.g., 'x', 'y', 'v')
            # and 't_val'.
            ode_params_in_scope = state_variable_names + ['t_val']
            formatted_equation = _format_kx_expr_for_python_lambda_body(
                equation_single_line, ode_params_in_scope, all_constants, state_variable_names, all_function_names, all_predicate_names
            )
            derivatives[idx] = formatted_equation
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

def _generate_reset_logic(variables: List[VariableDeclarationConfig], precondition: PreconditionConfig, definitions: DefinitionsConfig) -> List[str]:
    """
    Generates the initial state assignment logic for the reset method,
    leveraging the precondition and predicate definitions.
    """
    reset_code_lines = []
    
    # 1. Initialize all state variables with default values (can be randomized or fixed)
    defaults = {
        'x': '0.0', 'y': '0.0', 'v': '1.0', 'a': '0.0',
        'dx': '1.0', 'dy': '0.0', 'w': '0.0', 'r': '1.0',
        'xo': '5.0', 'yo': '0.0', 't': '0.0', 'vxo': '0.0', 'vyo': '0.0'
    }
    
    reset_code_lines.append("    # Default initial values for state variables (can be randomized or fixed based on problem)")
    for var in variables:
        default_val = defaults.get(var.name, '0.0')
        reset_code_lines.append(f"    self.state['{var.name}'] = {default_val}")
    reset_code_lines.append("")

    # 2. Expand the precondition formula (KeYmaera X syntax)
    expanded_precondition_kx = _expand_formula(precondition.formula, definitions)
    
    # 3. Parse the expanded precondition for explicit assignments
    # This regex looks for 'variable = value' or 'variable == value'
    # It assumes the value is a simple number or constant identifier.
    assignment_matches = re.findall(r'(\b\w+\b)\s*(?:=|==)\s*([a-zA-Z0-9_.]+)', expanded_precondition_kx)
    
    if assignment_matches:
        reset_code_lines.append("    # Apply explicit assignments from the precondition")
        for var_name, value_str in assignment_matches:
            # Only apply if it's a known state variable
            if var_name in [v.name for v in variables]:
                reset_code_lines.append(f"    self.state['{var_name}'] = {value_str}")
        reset_code_lines.append("")

    # 4. Special handling for dx, dy (isWellformedDir)
    # Check for the presence of the pattern related to dx^2+dy^2 in the expanded formula.
    # The actual predicate body for isWellformedDir is: `1-Ud <= dx^2+dy^2 & dx^2 + dy^2 <= 1+Ud`
    if re.search(r'dx\^2\s*\+\s*dy\^2', expanded_precondition_kx): # Check for KeYmaera X power operator
        reset_code_lines.append(f"    # Ensure dx, dy satisfy isWellformedDir (approximate unit vector)")
        reset_code_lines.append(f"    norm_sq = self.state['dx']**2 + self.state['dy']**2")
        reset_code_lines.append(f"    ud_val = self.constants.get('Ud', 0.0) # Get Ud from constants, default to 0.0 if not found")
        reset_code_lines.append(f"    min_norm_sq_target = (1.0 - ud_val)**2")
        reset_code_lines.append(f"    max_norm_sq_target = (1.0 + ud_val)**2")
        reset_code_lines.append(f"    if not (min_norm_sq_target <= norm_sq <= max_norm_sq_target):")
        reset_code_lines.append(f"        # If not within bounds, normalize and scale to satisfy the condition.")
        reset_code_lines.append(f"        # A common way to satisfy this is to make it a unit vector (dx^2+dy^2 = 1).")
        reset_code_lines.append(f"        # This satisfies 1-Ud <= 1 <= 1+Ud if Ud >= 0.")
        reset_code_lines.append(f"        norm = np.sqrt(norm_sq)")
        reset_code_lines.append(f"        if norm > 1e-6: # Avoid division by zero")
        reset_code_lines.append(f"            self.state['dx'] /= norm")
        reset_code_lines.append(f"            self.state['dy'] /= norm")
        reset_code_lines.append(f"        else: # Default to (1,0) if too small")
        reset_code_lines.append(f"            self.state['dx'] = 1.0")
        reset_code_lines.append(f"            self.state['dy'] = 0.0")
        reset_code_lines.append("")

    # 5. Special handling for (x-xo)^2 + (y-yo)^2 > 0
    # Check for the pattern in the expanded precondition (KeYmaera X syntax).
    if re.search(r'\(x\s*-\s*xo\)\^2\s*\+\s*\(y\s*-\s*yo\)\^2\s*>\s*0', expanded_precondition_kx):
        reset_code_lines.append(f"    # Ensure robot and obstacle are not at the same position")
        reset_code_lines.append(f"    if self.state['x'] == self.state['xo'] and self.state['y'] == self.state['yo']:")
        reset_code_lines.append(f"        self.state['x'] += 0.1 # Slightly perturb x to ensure separation")
        reset_code_lines.append("")

    # 6. Final verification
    # The _evaluate_expression method expects Python syntax, so format the expanded_precondition_kx now.
    final_precondition_for_eval = _format_kx_expr_for_python_eval_context(expanded_precondition_kx)
    
    # Escape strings for embedding into the generated Python code's f-string
    original_precondition_formula_escaped = precondition.formula.replace("'", "\\'").replace('\\', '\\\\').replace('\n', ' ')
    final_precondition_for_eval_escaped = final_precondition_for_eval.replace("'", "\\'").replace('\\', '\\\\').replace('\n', ' ')

    reset_code_lines.append(f"    # Verify the full precondition after setting initial states")
    reset_code_lines.append(f"    precondition_satisfied = self._evaluate_expression('{final_precondition_for_eval_escaped}', is_constraint=True)")
    reset_code_lines.append(f"    if not precondition_satisfied:")
    reset_code_lines.append(f"        print(f\"Warning: Initial state does not satisfy precondition: '{original_precondition_formula_escaped}'. Expanded: '{final_precondition_for_eval_escaped}'.\")")
    reset_code_lines.append(f"        # Optionally, raise an error or try to find a valid state")
    reset_code_lines.append(f"        # For now, we proceed but the environment might be in an invalid state.")
    reset_code_lines.append("")

    return reset_code_lines


# --- Main Execution (for testing the generator) ---
if __name__ == "__main__":
    # This part is for testing the generator directly with the provided PyAST structure
    
    # Create a dummy GymEnvironmentConfig based on the example PyAST structure
    # This mimics the output of convert_keymaerax_ast_to_intermediate_pyast
    
    # Constants
    ep_const = ConstantDefinitionConfig(name='ep', type='Real', value='1.0') # Set a value for testing
    b_const = ConstantDefinitionConfig(name='b', type='Real', value='0.5') # Set a value for testing
    A_const = ConstantDefinitionConfig(name='A', type='Real', value='1.0') # Set a value for testing
    W_const = ConstantDefinitionConfig(name='W', type='Real', value='1.0') # Set a value for testing
    V_const = ConstantDefinitionConfig(name='V', type='Real', value='1.0') # Added V constant for testing
    # FIX: Set Ud to a value < 1.0 for typical problem interpretation
    Ud_const = ConstantDefinitionConfig(name='Ud', type='Real', value='0.1') # Added Ud constant for testing

    # Functions
    # Bodies are now Python-compatible and use self.constants/self.functions
    stopDist_func = FunctionDefinitionConfig(name='stopDist', return_type='Real', parameters=[('Real', 'v')], body='((1+Ud) * v^2 / (2*b))') # Use ^ for power
    accelComp_func = FunctionDefinitionConfig(name='accelComp', return_type='Real', parameters=[('Real', 'v')], body='((1+Ud) * (A/b + 1) * (A/2 * ep^2 + ep*v))') # Use ^ for power
    admissibleSeparation_func = FunctionDefinitionConfig(name='admissibleSeparation', return_type='Real', parameters=[('Real', 'v')], body='stopDist(v) + accelComp(v)')

    # Predicates
    isWellformedDir_pred = PredicateDefinitionConfig(name='isWellformedDir', parameters=[('Real', 'dx'), ('Real', 'dy')], body='1-Ud <= dx^2+dy^2 & dx^2 + dy^2 <= 1+Ud')
    # FIX: Corrected bounds predicate to match 0 <= Ud < 1
    bounds_pred = PredicateDefinitionConfig(name='bounds', parameters=[], body='A >= 0 & b > 0 & ep > 0 & V >= 0 & 0<=Ud & Ud < 1') # Use & for AND, added V, Ud bounds
    initialState_pred = PredicateDefinitionConfig(name='initialState', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'dx'), ('Real', 'dy'), ('Real', 'xo'), ('Real', 'yo')], body='v = 0 & (x-xo)^2 + (y-yo)^2 > 0 & isWellformedDir(dx,dy)') # Use &
    loopinv_pred = PredicateDefinitionConfig(name='loopinv', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'dx'), ('Real', 'dy'), ('Real', 'xo'), ('Real', 'yo')], body='v >= 0 & isWellformedDir(dx,dy) & (abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v))')
    assumptions_pred = PredicateDefinitionConfig(name='assumptions', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'dx'), ('Real', 'dy'), ('Real', 'xo'), ('Real', 'yo')], body='bounds() & initialState(x,y,v,dx,dy,xo,yo)')


    # controlFunc Action Config
    # This structure directly reflects the problem description's controlFunc
    # Choice 0: a := -b
    # Choice 1: v = 0; a := 0; w := 0
    # Choice 2: a := A; w := *; -W<=w & w<=W; r := *; xo := *; yo := *; r!=0 & r*w = v; abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v)
    choice0_actions_branch0 = SequenceActionConfig(actions=[
        DeterministicAssignmentActionConfig(variable='a', value='-b')
    ])
    choice0_actions_branch1 = SequenceActionConfig(actions=[
        ActionConstraintConfig(formula='v = 0'),
        DeterministicAssignmentActionConfig(variable='a', value='0'),
        DeterministicAssignmentActionConfig(variable='w', value='0')
    ])
    choice0_actions_branch2 = SequenceActionConfig(actions=[
        DeterministicAssignmentActionConfig(variable='a', value='A'),
        NondeterministicAssignmentActionConfig(variable='w'),
        ActionConstraintConfig(formula='-W<=w & w<=W'), # Use &
        NondeterministicAssignmentActionConfig(variable='r'),
        NondeterministicAssignmentActionConfig(variable='xo'),
        NondeterministicAssignmentActionConfig(variable='yo'),
        ActionConstraintConfig(formula='r!=0 & r*w = v'),
        ActionConstraintConfig(formula='abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v)') # Use |
    ])

    control_func_action_config = ChoiceActionConfig(level_name='ChoiceLevel_0', choices=[
        choice0_actions_branch0, 
        choice0_actions_branch1,
        choice0_actions_branch2
    ])
    control_func = ControlFuncConfig(name='controlFunc', action_space_config=control_func_action_config)


    # stepFunc ODEs and Evolution Domain
    step_func_odes = [
        ODEConfig(variable='x', equation='v * dx'),
        ODEConfig(variable='y', equation='v * dy'),
        ODEConfig(variable='v', equation='a'),
        ODEConfig(variable='dx', equation='-w * dy'),
        ODEConfig(variable='dy', equation='w * dx'),
        ODEConfig(variable='w', equation='a/r'), # Corrected from '0'
        ODEConfig(variable='t', equation='1')
    ]
    step_func_evolution_domain = EvolutionDomainConfig(constraints=['t <= ep', 'v >= 0'])
    step_func = StepFuncConfig(name='stepFunc', odes=step_func_odes, evolution_domain=step_func_evolution_domain, duration='ep')

    # DefinitionsConfig
    definitions_config = DefinitionsConfig(
        constants=[ep_const, b_const, A_const, W_const, V_const, Ud_const], # Added Ud_const
        functions=[stopDist_func, accelComp_func, admissibleSeparation_func],
        predicates=[isWellformedDir_pred, bounds_pred, initialState_pred, loopinv_pred, assumptions_pred],
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
    postcondition = PostconditionConfig(formula='(x - xo)^2 + (y - yo)^2 > 0') # Simplified for this example
    
    # Loop body: { { controlFunc; t := 0; } { stepFunc;@invariant(...) } }
    loop_body_actions = SequenceActionConfig(actions=[
        SequenceActionConfig(actions=[
            CallHybridProgramConfig(hp_name='controlFunc'),
            DeterministicAssignmentActionConfig(variable='t', value='0')
        ]),
        HybridProgramStepConfig(hp_name='stepFunc', invariant_formula='t>=0 & isWellformedDir(dx, dy) & (v>0 -> (abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v)))') # Simplified invariant for example
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
    # This example env_name will now be sanitized:
    gym_config_example = GymEnvironmentConfig(
        env_name='IJRR17_Corollary_1_2__Static_safety_with_margin_for_imperfect_trajectories', # Original name
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
    
    # Update the example usage to reflect the sanitized class name
    sanitized_example_env_name = re.sub(r'[^a-zA-Z0-9_]', '_', gym_config_example.env_name)
    sanitized_example_env_name = re.sub(r'_+', '_', sanitized_env_name).strip('_')
    if sanitized_example_env_name and sanitized_example_env_name[0].isdigit():
        sanitized_example_env_name = '_' + sanitized_example_env_name
    if not sanitized_example_env_name:
        sanitized_example_env_name = "GeneratedEnv"

    print(f"""
To use the generated environment:

1.  Make sure `gymnasium`, `numpy`, `scipy`, and `re` are installed (`pip install gymnasium numpy scipy`).
2.  **CRITICAL:** Save the generated code above as `{output_filename}`. **Ensure you completely replace your existing generator file with this one.**
3.  Ensure `pyast_config.py` is in the same directory.
4.  You can then import and use the generated environment in another Python script or interactive session:

```python
import gymnasium as gym
from generated_kx_env import {sanitized_example_env_name} 
import numpy as np

# Instantiate the environment
env = {sanitized_example_env_name}()

observation, info = env.reset()
print("Initial observation:", observation)
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

# Example 1: Choose the first branch of ChoiceLevel_0 (a := -b)
print("\\nTaking 5 steps with action_branch_0 (a := -b):")
env.reset() # Reset for a fresh start
for i in range(5):
    print(f"Step {{i+1}} (before action): State={{env.state}}")
    action_to_take = {{
        'choice_ChoiceLevel_0': 0, # Select the first branch (a := -b)
    }}
    observation, reward, terminated, truncated, info = env.step(action_to_take)
    print(f"Step {{i+1}}: Obs={{observation}}, Reward={{reward:.2f}}, Terminated={{terminated}}, Truncated={{truncated}}")
    if terminated or truncated:
        print("Episode finished.")
        break
env.close()

# Example 2: Reset and choose the third branch of ChoiceLevel_0
# (a := A; w := *; etc.)
print("\\nResetting environment and taking 5 steps with action_branch_2 (a := A; w := *; etc.):")
env = {sanitized_example_env_name}() # Re-instantiate for a fresh start
observation, info = env.reset()

# Define action for ChoiceLevel_0 branch 2 (index 2)
# It's crucial that r != 0 and r*w = v for the constraint to pass.
# Let's assume v is initially 0.0 (from initialState).
# If w = 0.0, then r*w = 0.0, which matches v=0.0.
action_branch_2 = {{
    'choice_ChoiceLevel_0': 2, # Select the third branch (a := A; w := *; etc.)
    'assign_w': 0.0,           # Example value for w (within -W to W, i.e., -1.0 to 1.0)
    'assign_r': 1.0,           # Example value for r (r!=0)
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
""")