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
from gym_generator.expression_helpers import format_kx_expr_for_python_lambda_body_generator, expand_formula
from gym_generator.ast_serialization import serialize_ast_node

# Note: gymnasium.spaces is needed for action space definition but is imported in the generated code itself.
# We don't import it here directly to avoid circular dependencies if pyast_config.py were to import from here.

def generate_action_space_and_logic(hps: List[HybridProgramConfig]) -> Tuple[str, ActionType, Dict[str, str]]:
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
    # The parameters in scope for ODE equations are the state variables themselves and 't_val'.
    # We pass all_constants, all_function_names, all_predicate_names to format_kx_expr_for_python_lambda_body_generator
    # so it can correctly prefix them with `self.constants['...']`, `self.functions['...']`, etc.
    # The state variables (x, y, v, etc.) are directly in the local scope of ode_func, so they are not prefixed.
    ode_params_in_scope = list(ode_state_map.keys()) + ['t_val'] # These are direct local variables

    for ode in step_func_config.odes:
        idx = ode_state_map.get(ode.variable)
        if idx is not None:
            # Ensure ODE equation is a single line
            equation_single_line = ode.equation.replace('\n', ' ').strip()
            # Format the ODE equation for direct Python evaluation within the ODE function's scope
            formatted_equation = format_kx_expr_for_python_lambda_body_generator(
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


def generate_reset_logic(variables: List[VariableDeclarationConfig], precondition: PreconditionConfig, definitions: DefinitionsConfig) -> List[str]:
    """
    Generates the initial state assignment logic for the reset method,
    leveraging the precondition and predicate definitions.
    Prioritizes Z3 solver for initial state, and then attempts to perturb it randomly.
    Falls back to heuristics if Z3 fails or perturbation is unsuccessful.
    """
    reset_code_lines = []
    
    # 1. Always initialize all state variables with default values first
    # These defaults can be overridden by Z3 or by specific heuristics later.
    defaults = {
        'x': '0.0', 'y': '0.0', 'v': '1.0', 'a': '0.0',
        'dx': '1.0', 'dy': '0.0', 'w': '0.0', 'r': '1.0',
        'xo': '5.0', 'yo': '0.0', 't': '0.0', 'vxo': '0.0', 'vyo': '0.0'
    }
    
    reset_code_lines.append("    # Default initial values for state variables")
    for var in variables:
        default_val = defaults.get(var.name, '0.0')
        reset_code_lines.append(f"    self.state['{var.name}'] = {default_val}")
    reset_code_lines.append("")

    # --- Z3 Solver Integration ---
    # Attempt to find a satisfying initial state using Z3 solver
    reset_code_lines.append(f"    z3_base_solution = self._solve_initial_state_with_z3(")
    reset_code_lines.append(f"        self._variables_config.state_variables, ")
    reset_code_lines.append(f"        self._problem_config.precondition.formula")
    reset_code_lines.append(f"    )")
    
    
    reset_code_lines.append(f"    final_initial_state = None")

    reset_code_lines.append(f"    if z3_base_solution:")
    reset_code_lines.append(f"        # Z3 found a base solution. Now, perturb it to get 'new' values.")
    reset_code_lines.append(f"        # We will try to perturb it and verify, falling back to the base if perturbation fails.")
    reset_code_lines.append(f"        perturbation_magnitude = 0.05 # Small perturbation range")
    reset_code_lines.append(f"        num_perturbation_attempts = 5")
    reset_code_lines.append(f"        perturbed_state_found = False")
    reset_code_lines.append(f"        ")
    reset_code_lines.append(f"        for _ in range(num_perturbation_attempts):")
    reset_code_lines.append(f"            trial_state = z3_base_solution.copy()")
    reset_code_lines.append(f"            for var_name, value in z3_base_solution.items():") # Iterate over Z3's solution
    reset_code_lines.append(f"                if var_name in self.state:") # Only perturb state variables
    reset_code_lines.append(f"                    perturb = self.np_random.uniform(-perturbation_magnitude, perturbation_magnitude)")
    reset_code_lines.append(f"                    trial_state[var_name] = value + perturb") # Apply perturbation to base value
    reset_code_lines.append(f"            ")
    
    # Escape the precondition formula for embedding into the f-string
    precondition_formula_escaped_for_eval = precondition.formula.replace("'", "\\'").replace('\\', '\\\\').replace('\n', ' ')
    
    reset_code_lines.append(f"            # Verify the perturbed state against the precondition")
    reset_code_lines.append(f"            # Temporarily apply trial state to self.state and self.constants for precondition check")
    reset_code_lines.append(f"            original_state_copy = self.state.copy()")
    reset_code_lines.append(f"            original_constants_copy = self.constants.copy()") # Store original constants too
    reset_code_lines.append(f"            for var_name, value in trial_state.items():")
    reset_code_lines.append(f"                if var_name in self.state: # Update state variables")
    reset_code_lines.append(f"                    self.state[var_name] = value")
    reset_code_lines.append(f"                elif var_name in self.constants: # Update constants that Z3 solved for")
    reset_code_lines.append(f"                    self.constants[var_name] = value")
    reset_code_lines.append(f"            ")
    
    reset_code_lines.append(f"            if self._evaluate_expression('{precondition_formula_escaped_for_eval}', is_constraint=True):")
    reset_code_lines.append(f"                final_initial_state = trial_state")
    reset_code_lines.append(f"                perturbed_state_found = True")
    reset_code_lines.append(f"                # Revert self.state and self.constants before breaking, as final_initial_state will be applied later")
    reset_code_lines.append(f"                self.state.update(original_state_copy)") 
    reset_code_lines.append(f"                self.constants.update(original_constants_copy)")
    reset_code_lines.append(f"                break")
    reset_code_lines.append(f"            self.state.update(original_state_copy) # Revert if perturbation failed")
    reset_code_lines.append(f"            self.constants.update(original_constants_copy)")
    reset_code_lines.append(f"        ")
    reset_code_lines.append(f"        if not perturbed_state_found:")
    reset_code_lines.append(f"            print(f\"Warning: Could not find a valid perturbed state after {{{{num_perturbation_attempts}}}} attempts. Using base Z3 solution.\")")
    reset_code_lines.append(f"            final_initial_state = z3_base_solution # Fallback to the unperturbed Z3 solution")
    reset_code_lines.append(f"        ")
    reset_code_lines.append(f"        # Apply the chosen initial state (perturbed or base Z3 solution)")
    reset_code_lines.append(f"        for var_name, value in final_initial_state.items():")
    reset_code_lines.append(f"            if var_name in self.state:") # Update state variables
    reset_code_lines.append(f"                self.state[var_name] = value")
    reset_code_lines.append(f"            elif var_name in self.constants:") # Update constants that Z3 solved for
    reset_code_lines.append(f"                self.constants[var_name] = value")
    reset_code_lines.append(f"        self.last_failing_precondition_formula = None")
    reset_code_lines.append(f"        print(f\"Initial state set by Z3 (possibly perturbed): {{self.state}}\")")
    reset_code_lines.append(f"        print(f\"Constants possibly set by Z3: {{self.constants}}\")") # Added for visibility
    reset_code_lines.append(f"    else:")
    reset_code_lines.append(f"        # Z3 failed, fall back to heuristics (existing code).")
    reset_code_lines.append(f"        print(f\"Warning: Z3 solver failed to find a satisfying initial state for the precondition: '{{self._problem_config.precondition.formula}}'. Falling back to heuristics. This may lead to the initial state not satisfying the precondition, which could cause immediate termination or warnings during reset. Consider revising the precondition or ensuring it is satisfiable with the given definitions.\")")

    # 2. Parse the expanded precondition for explicit assignments (fallback if Z3 fails)
    expanded_precondition_kx = expand_formula(precondition.formula, definitions)
    print(f"Expanded precondition for parsing explicit assignments: '{expanded_precondition_kx}'") # Debug print to verify the expanded precondition string
    # Expanded precondition for parsing explicit assignments: '((A >= 0
    # & b > 0
    # & ep > 0) & (v = 0
    # & (x-xo)^2 + (y-yo)^2 > 0
    # & (dx^2 + dy^2 = 1)
    # & r != 0))'
    # The regex below looks for 'var = value' patterns.
    assignment_matches = re.findall(r'(\b\w+\b)\s*=\s*([a-zA-Z0-9_.]+)', expanded_precondition_kx)
    
    reset_code_lines.append(f"        if not z3_base_solution and {len(assignment_matches)} > 0:") # Only run if Z3 failed AND there are assignments
    reset_code_lines.append(f"            # Apply explicit assignments from the precondition (fallback)")
    for var_name, value_str in assignment_matches:
        if var_name in [v.name for v in variables]:
            try:
                float_val = float(value_str)
                reset_code_lines.append(f"            self.state['{var_name}'] = {float_val}")
            except ValueError:
                reset_code_lines.append(f"            # Could not parse '{value_str}' as float for '{var_name}'. Will be checked by _evaluate_expression.")
    reset_code_lines.append("")

    # 3. Special handling for dx, dy (isWellformedDir) (fallback if Z3 fails)
    if re.search(r'dx\\^2\\s*\\+\\s*dy\\^2', expanded_precondition_kx):
        reset_code_lines.append(f"        if not z3_base_solution:") # Only run if Z3 failed
        reset_code_lines.append(f"            # Ensure dx, dy satisfy isWellformedDir (approximate unit vector) (fallback)")
        reset_code_lines.append(f"            norm_sq = self.state['dx']**2 + self.state['dy']**2")
        reset_code_lines.append(f"            ud_val = self.constants.get('Ud', 0.0) # Get Ud from constants, default to 0.0 if not found")
        reset_code_lines.append(f"            # The actual isWellformedDir is '1-Ud <= dx^2+dy^2 & dx^2 + dy^2 <= 1+Ud'")
        reset_code_lines.append(f"            # For fallback, we normalize to 1 if it's too far off, or if Ud is 0.")
        reset_code_lines.append(f"            if not self._approx_eq(norm_sq, 1.0):") # Check if it's approximately 1
        reset_code_lines.append(f"                norm = np.sqrt(norm_sq)")
        reset_code_lines.append(f"                if norm > self.equality_tolerance: # Avoid division by zero with tolerance")
        reset_code_lines.append(f"                    self.state['dx'] /= norm")
        reset_code_lines.append(f"                    self.state['dy'] /= norm")
        reset_code_lines.append(f"                else: # Default to (1,0) if too small")
        reset_code_lines.append(f"                    self.state['dx'] = 1.0")
        reset_code_lines.append(f"                    self.state['dy'] = 0.0")
        reset_code_lines.append("")

    # 4. Special handling for (x-xo)^2 + (y-yo)^2 > 0 (fallback if Z3 fails)
    if re.search(r'\\(x\\s*-\\s*xo\\)\\^2\\s*\\+\\s*\\(y\\s*-\\s*yo\\)\\^2\\s*>\\s*0', expanded_precondition_kx):
        reset_code_lines.append(f"        if not z3_base_solution:") # Only run if Z3 failed
        reset_code_lines.append(f"            # Ensure robot and obstacle are not at the same position (fallback)")
        reset_code_lines.append(f"            if self._approx_eq(self.state['x'], self.state['xo']) and self._approx_eq(self.state['y'], self.state['yo']):")
        reset_code_lines.append(f"                self.state['x'] += 0.1 # Slightly perturb x to ensure separation")
        reset_code_lines.append("")

    # 5. Final verification (always runs)
    # The _evaluate_expression method expects KeYmaera X syntax (with '=' for equality).
    # It will internally convert '=' to self._approx_eq.
    final_precondition_for_eval = expand_formula(precondition.formula, definitions)
    print(f"Final expanded precondition for evaluation: '{final_precondition_for_eval}'") # Debug print to verify the final precondition string used for evaluation
    # Remove extra spaces for cleaner embedding
    normalized_string = " ".join(final_precondition_for_eval.split())
    
    # Escape strings for embedding into the generated Python code's f-string
    original_precondition_formula_escaped = precondition.formula.replace("'", "\\'").replace('\\', '\\\\').replace('\n', ' ')
    final_precondition_for_eval_escaped = normalized_string.replace("'", "\\'").replace('\\', '\\\\')
    
    print(f"Original precondition formula (escaped for embedding): '{original_precondition_formula_escaped}'") # Debug print
    print(f"Final precondition for evaluation (escaped for embedding): '{final_precondition_for_eval_escaped}'") # Debug print
    
    reset_code_lines.append(f"    # Verify the full precondition after setting initial states")
    reset_code_lines.append(f"    # Evaluate each conjunct of the precondition to identify specific failures")
    
    # This variable will hold the failing formula if any.
    reset_code_lines.append(f"    failing_formula = None")
    
    reset_code_lines.append(f"    precondition_conjuncts = self._split_top_level_and_local('{final_precondition_for_eval_escaped}')")
    reset_code_lines.append(f"    precondition_satisfied = True")
    reset_code_lines.append(f"    for conj_str in precondition_conjuncts:")
    reset_code_lines.append(f"        # Evaluate each conjunct. _evaluate_expression will handle the parsing.")
    reset_code_lines.append(f"        if not self._evaluate_expression(conj_str, is_constraint=True):")
    reset_code_lines.append(f"            precondition_satisfied = False")
    reset_code_lines.append(f"            failing_formula = conj_str")
    reset_code_lines.append(f"            break # Report the first failing conjunct")
    
    reset_code_lines.append(f"    if not precondition_satisfied:")
    reset_code_lines.append(f"        self.last_failing_precondition_formula = failing_formula")
    reset_code_lines.append(f"        print(f\"Warning: Initial state does not satisfy precondition: '{original_precondition_formula_escaped}'.\")")
    reset_code_lines.append(f"        print(f\"         Failing conjunct: '{{failing_formula}}'.\")")
    reset_code_lines.append(f"    else:")
    reset_code_lines.append(f"        self.last_failing_precondition_formula = None")
    reset_code_lines.append("")

    return reset_code_lines