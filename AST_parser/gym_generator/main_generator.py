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
from gym_generator.expression_helpers import format_kx_expr_for_python_lambda_body_generator, expand_formula
from gym_generator.gym_logic_sub_generators import generate_action_space_and_logic, generate_ode_solver_logic, generate_reset_logic
from gym_generator.embedded_methods import (
    SPLIT_TOP_LEVEL_AND_LOCAL_METHOD_CODE,
    EVALUATE_EXPRESSION_METHOD_TEMPLATE,
    KX_EXPR_TO_Z3_FORMULA_METHOD_CODE,
    SOLVE_INITIAL_STATE_WITH_Z3_METHOD_CODE
)


def generate_gym_environment_code(config: GymEnvironmentConfig, envName: str = "") -> str:
    """
    Generates Python code for a Gymnasium environment based on the provided
    GymEnvironmentConfig AST.
    """
    if not envName:
        original_env_name = config.env_name
        
        # --- SANITIZE ENV_NAME FOR PYTHON CLASS IDENTIFIER ---
        # 1. Replace non-alphanumeric characters (excluding underscore) with underscore
        sanitized_env_name = re.sub(r'[^a-zA-Z0-9_]', '_', original_env_name)
        # 2. Replace multiple underscores with a single underscore and strip leading/trailing underscores
        sanitized_env_name = re.sub(r'_+', '_', sanitized_env_name).strip('_')
        
        # 3. Ensure it's not starting with a digit
        if sanitized_env_name and sanitized_env_name[0].isdigit():
            sanitized_env_name = '_' + sanitized_env_name
        
        # 4. Fallback if the name becomes empty after sanitization
        if not sanitized_env_name:
            sanitized_env_name = "GeneratedEnv" 
        
        env_name_for_class = sanitized_env_name # Use this for the class definition
    
    else: 
        env_name_for_class = envName # Use provided envName directly for class definition
    # ---------------------------------------------------

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
    indented_reset_block = textwrap.indent("\n".join(reset_initial_state_code_lines), '    ') # Adjusted indentation for reset body

    # --- Problem Logic (Postcondition, Loop Invariants) ---
    # Pass raw KeYmaera X formulas to be parsed at runtime by _evaluate_expression
    postcondition_formula_py = config.problem.postcondition.formula.replace('\n', ' ').strip()
    loop_invariant_formula_py = config.problem.loop_execution.loop_invariant_formula.replace('\n', ' ').strip()

    # Imports for AST classes needed for _control_func_action_config
    ast_imports = [
        "ActionType", 
        "ChoiceActionConfig", "SequenceActionConfig", "DeterministicAssignmentActionConfig",
        "NondeterministicAssignmentActionConfig", "ActionConstraintConfig", "CallHybridProgramConfig",
        "HybridProgramStepConfig", "GymEnvironmentConfig", "DefinitionsConfig", "VariablesConfig",
        "ProblemConfig", "ConstantDefinitionConfig", "FunctionDefinitionConfig", "PredicateDefinitionConfig",
        "HybridProgramConfig", "ControlFuncConfig", "StepFuncConfig", "ODEConfig", "EvolutionDomainConfig",
        "VariableDeclarationConfig", "PreconditionConfig", "PostconditionConfig", "LoopExecutionConfig"
    ]
    ast_import_statement = f"from pyast_config import ({', '.join(ast_imports)})"

    # Imports for typing hints
    typing_imports = ["Dict", "Any", "Tuple", "List", "Optional", "Union"]
    typing_import_statement = f"from typing import ({', '.join(typing_imports)})"


    # --- Main Environment Class Structure ---
    env_code = textwrap.dedent(f"""import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import math # For math functions like sqrt, sin, cos, etc.
import textwrap
import re # For regex replacements in _evaluate_expression
import z3 # For SMT solving in reset()
{typing_import_statement} # Import typing hints
{ast_import_statement} # Import AST classes

class {env_name_for_class}(gym.Env):
    metadata = {{"render_modes": ["human"], "render_fps": 30}}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # --- Store original AST config objects for Z3 and other runtime needs ---
        self._definitions_config = {serialize_ast_node(config.definitions, indent_level=2)}
        self._variables_config = {serialize_ast_node(config.variables, indent_level=2)}
        self._problem_config = {serialize_ast_node(config.problem, indent_level=2)}

        # --- Semantic Gap Bridging: Floating Point Equality Tolerance ---
        self.equality_tolerance = 1e-9 # Default absolute tolerance for np.isclose
        
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
        # Initialize to 0.0 as placeholders; these will be overwritten by reset()
        self.state = {{key: 0.0 for key in self._state_keys}} 
        # Initial state values are set in reset()
        self.last_failing_precondition_formula = None # Initialize for debugging

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

        # Initialize the environment to a valid state using the reset method
        self.reset() # <--- ADDED THIS LINE
        
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

    def _approx_eq(self, a, b):
        \"\"\"
        Performs an approximate equality check using numpy.isclose.
        This bridges the semantic gap between formal exact equality and float approximation.
        \"\"\"
        return np.isclose(a, b, atol=self.equality_tolerance)

    def _expand_formula(self, formula_str: str, definitions: DefinitionsConfig) -> str:
        \"\"\"
        Recursively expands predicate calls within a formula string.
        This returns the formula in KeYmaera X syntax, suitable for further parsing.
        It handles simple argument substitution.
        \"\"\"
        expanded_formula = formula_str
        
        predicate_map = {{p.name: (p.body, [param[1] for param in p.parameters]) for p in definitions.predicates}}
        
        while True:
            num_substitutions_in_pass = 0
            # Sort by length descending to handle cases where one predicate name is a substring of another
            sorted_pred_names = sorted(predicate_map.keys(), key=len, reverse=True)
            
            for pred_name in sorted_pred_names:
                pred_body, pred_params = predicate_map[pred_name]
                # Regex to find calls to this specific predicate, capturing arguments
                pattern = r'\\b' + re.escape(pred_name) + r'\\s*\\(([^)]*)\\)'
                
                def replace_match(match):
                    nonlocal num_substitutions_in_pass
                    num_substitutions_in_pass += 1
                    call_args_str = match.group(1)
                    call_args = [arg.strip() for arg in call_args_str.split(',') if arg.strip()]
                    
                    if len(call_args) != len(pred_params):
                        print(f"Warning: Mismatch in number of arguments for predicate '{{pred_name}}'. Expected {{len(pred_params)}}, got {{len(call_args)}} in call '{{match.group(0)}}'.")
                        return match.group(0)
                    
                    substituted_body = pred_body
                    for i, param_name in enumerate(pred_params):
                        # Replace `param_name` with `call_args[i]` in the body
                        # Use word boundaries to avoid partial replacements (e.g., replacing 'x' in 'dx')
                        substituted_body = re.sub(r'\\b' + re.escape(param_name) + r'\\b', call_args[i], substituted_body)
                    
                    return f"({{substituted_body}})" # Enclose in parens to maintain logical grouping

                expanded_formula = re.sub(pattern, replace_match, expanded_formula)
            
            if num_substitutions_in_pass == 0:
                break
                
        return expanded_formula

    def _get_predicate_metadata(self, predicate_name: str) -> Dict[str, Any]:
        \"\"\"Retrieves metadata for a given predicate name.\"\"\"
        for pred_config in self._definitions_config.predicates:
            if pred_config.name == predicate_name:
                return pred_config.metadata
        return {{}} # Return empty dict if not found

    def _apply_meta_effect(self, metadata: Dict[str, Any], current_reward: float, current_terminated: bool, current_truncated: bool, is_failure_case: bool) -> Tuple[float, bool, bool]:
        \"\"\"
        Applies reward/penalty/termination/truncation effects based on metadata.
        'on_success': 'action:status=value' (e.g., 'terminate:status=success', 'continue:status=none')
        'on_failure': 'action:status=value' (e.g., 'truncate:status=failure', 'terminate:status=failure')
        
        action can be 'continue', 'terminate', 'truncate'.
        status can be 'success', 'failure', 'none'.

        is_failure_case: True if this call is due to a condition failing, False if it's a success path.
        \"\"\"
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
        \"\"\"
        Evaluates a shape expression from metadata.
        Shape expressions might use ternary operators (e.g., 'v > 0 ? 1/v : v').
        \"\"\"
        if not shape_expr:
            return 0.0
        
        # Handle ternary operator 'condition ? value_if_true : value_if_false'
        # This is a simple regex-based parser, might need more robustness for complex cases.
        match = re.match(r'(.+)\\s*\\?\\s*(.+)\\s*:\\s*(.+)', shape_expr)
        if match:
            condition_str, true_val_str, false_val_str = match.groups()
            try:
                condition_result = self._evaluate_expression(condition_str)
                if condition_result:
                    return float(self._evaluate_expression(true_val_str))
                else:
                    return float(self._evaluate_expression(false_val_str))
            except Exception as e:
                print(f"Warning: Error evaluating shape ternary expression '{{shape_expr}}': {{e}}")
                return 0.0
        else:
            # If not a ternary, just evaluate as a regular expression
            try:
                return float(self._evaluate_expression(shape_expr))
            except Exception as e:
                print(f"Warning: Error evaluating shape expression '{{shape_expr}}': {{e}}")
                return 0.0

{textwrap.indent(KX_EXPR_TO_Z3_FORMULA_METHOD_CODE, '    ')}

{textwrap.indent(SOLVE_INITIAL_STATE_WITH_Z3_METHOD_CODE, '    ')}

{textwrap.indent(EVALUATE_EXPRESSION_METHOD_TEMPLATE, '    ')}

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
                # Value is a raw KeYmaera X string, parsed at runtime by _evaluate_expression
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
            
            # Try to find metadata for this constraint if it's a predicate call
            predicate_name_match = re.match(r'^\\s*(\\w+)\\s*\\(', action_config.formula)
            if predicate_name_match:
                pred_name = predicate_name_match.group(1)
                metadata = self._get_predicate_metadata(pred_name)
            else:
                # Default metadata for generic constraints
                metadata = {{'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'}}
            
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

{textwrap.indent(SPLIT_TOP_LEVEL_AND_LOCAL_METHOD_CODE, '    ')}

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
            # print("Control action not applied due to constraint violation or missing assignment.")
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
                    # print(f"ODE integration failed: {{solution.message}}")
                    reward, terminated, truncated = self._apply_meta_effect(
                        {{'type': 'safety', 'penalty': '-150.0', 'on_failure': 'terminate:status=failure'}},
                        reward, terminated, truncated, is_failure_case=True
                    )
                    ode_terminated_by_event = True

            except Exception as e:
                # print(f"Error during ODE integration: {{e}}")
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