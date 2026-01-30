from __future__ import annotations # MOVED TO THE VERY TOP OF THE FILE

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import sympy
import textwrap
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

# Import all dataclass definitions from ast_config.py, including ActionType
# Assuming ast_config.py exists and provides these definitions
from ast_config import (
    ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig, DefinitionsConfig,
    VariablesConfig, VariableDeclarationConfig,
    NondeterministicAssignmentActionConfig, DeterministicAssignmentActionConfig, ActionConstraintConfig,
    SequenceActionConfig, ChoiceActionConfig, ControlFuncConfig, CallHybridProgramConfig,
    ODEConfig, EvolutionDomainConfig, StepFuncConfig, HybridProgramConfig, HybridProgramStepConfig,
    PreconditionConfig, PostconditionConfig, LoopExecutionConfig, ProblemConfig,
    GymEnvironmentConfig,
    ActionType
)

# Configure logging for gym_generator.py
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logs, INFO for less verbose


# --- Helper for symbolic evaluation ---
class SympyEvaluator:
    """
    Helper class to manage symbolic variables and evaluate expressions during code generation.
    This class primarily prepares the SymPy symbols and function objects
    that will be used to parse expressions.
    """
    def __init__(self, state_variables: List[VariableDeclarationConfig], auxiliary_variables: List[VariableDeclarationConfig],
                 constants: List[ConstantDefinitionConfig], functions: List[FunctionDefinitionConfig],
                 predicates: List[PredicateDefinitionConfig]):
        self.sympy_symbols = {}       # Stores sympy.Symbol objects for variables and constants
        self.sympy_functions = {}     # Stores sympy.Function objects for user-defined functions
        self.sympy_predicates = {}    # Stores sympy.Function objects for user-defined predicates
        self.known_constants_raw = {} # Store raw strings for constants

        # Create sympy symbols for all variables
        for var in state_variables + auxiliary_variables:
            self.sympy_symbols[var.name] = sympy.symbols(var.name)

        # Process constants (store raw string, evaluation happens in generated env)
        for const in constants:
            self.known_constants_raw[const.name] = const.value
            self.sympy_symbols[const.name] = sympy.symbols(const.name) # Also add as symbol for expressions

        # Process functions and predicates for sympy parsing
        for func in functions:
            self.sympy_functions[func.name] = sympy.Function(func.name)

        for pred in predicates:
            self.sympy_predicates[pred.name] = sympy.Function(pred.name)

    def _parse_expression(self, expr_str: str):
        """
        Parses a mathematical expression string into a sympy expression.
        It uses all known symbols and function objects for correct parsing.
        """
        expr_str = expr_str.replace('^', '**')

        # Combine all known symbols and sympy Function objects for parsing context
        parsing_locals = {
            **self.sympy_symbols,
            **self.sympy_functions,
            **self.sympy_predicates
        }

        try:
            return sympy.sympify(expr_str, locals=parsing_locals)
        except (sympy.SympifyError, SyntaxError) as e:
            raise ValueError(f"Could not parse expression '{expr_str}': {e}")

    def get_ode_function_body_str(self, odes: List[ODEConfig], state_variables: List[VariableDeclarationConfig]):
        """
        Generates the string representation of the ODE system's return values
        for the generated _ode_system method.
        """
        ode_eq_list_str = []
        # Ensure order matches observation_vars
        for var in state_variables:
            ode_config = next((o for o in odes if o.variable == var.name), None)
            if ode_config:
                # Prepare the ODE equation string for embedding into a self._eval_expr call
                ode_eq_list_str.append(f"self._eval_expr('{_prepare_expr_for_embedding(ode_config.equation)}', state_dict)")
            else:
                ode_eq_list_str.append("0.0") # Variable not changing
        return f"np.array([{', '.join(ode_eq_list_str)}], dtype=np.float32)"

    def get_event_function_body_str(self, evolution_domain: EvolutionDomainConfig):
        """
        Generates the string representation of the event function's return value
        for the generated _ode_event method.
        """
        if not evolution_domain or not evolution_domain.constraints:
            return "True" # No event if no constraints (always true, so event never triggers)

        formula = " and ".join(f"({c})" for c in evolution_domain.constraints)
        # Clean newlines and escape quotes from the formula before returning
        return _prepare_expr_for_embedding(formula)


# Helper function to prepare expression strings for embedding into f-strings
def _prepare_expr_for_embedding(expr: Optional[str]) -> str:
    if expr is None:
        return "None" # Return string "None" for actual None values, to be handled by _eval_expr
    # 1. Remove newlines
    cleaned_expr = expr.replace('\n', ' ').replace('\r', '')
    # 2. Escape single quotes for embedding in a single-quoted string literal
    escaped_expr = cleaned_expr.replace("'", "\\'")
    return escaped_expr

# UPDATED HELPER FUNCTION: Flatten nested SequenceActionConfig and HybridProgramStepConfig objects
def _flatten_actions(action_list: List[ActionType]) -> List[ActionType]:
    """
    Recursively flattens a list of actions, unpacking SequenceActionConfig
    and HybridProgramStepConfig objects.
    """
    flattened = []
    for action in action_list:
        if isinstance(action, SequenceActionConfig):
            flattened.extend(_flatten_actions(action.actions))
        elif isinstance(action, HybridProgramStepConfig):
            # A HybridProgramStepConfig represents a call to a hybrid program (like stepFunction)
            # with an optional invariant. We represent this as a single item in the flattened list.
            flattened.append(action)
        else:
            flattened.append(action)
    return flattened


def generate_gym_environment_code(env_config: GymEnvironmentConfig) -> str:
    """
    Generates the Python code for a Gymnasium environment from the GymEnvironmentConfig.
    """
    env_name = env_config.env_name
    definitions = env_config.definitions
    variables = env_config.variables
    problem = env_config.problem

    # Retrieve control_func and step_func from hybrid_programs list
    control_func: Optional[ControlFuncConfig] = next(
        (hp for hp in definitions.hybrid_programs if isinstance(hp, ControlFuncConfig)), None
    )
    step_func: Optional[StepFuncConfig] = next(
        (hp for hp in definitions.hybrid_programs if isinstance(hp, StepFuncConfig)), None
    )

    if not control_func:
        raise ValueError("ControlFuncConfig not found in definitions.hybrid_programs")
    if not step_func:
        raise ValueError("StepFuncConfig not found in definitions.hybrid_programs")


    # Initialize SympyEvaluator to prepare symbolic context for parsing expressions
    sympy_evaluator = SympyEvaluator(
        state_variables=variables.state_variables,
        auxiliary_variables=variables.auxiliary_variables,
        constants=definitions.constants,
        functions=definitions.functions,
        predicates=definitions.predicates
    )

    # --- 1. Generate Definitions (Constants, Functions, Predicates) ---
    init_definitions_code = [] # For constants, in __init__
    class_method_definitions_code = [] # For functions/predicates, as class methods

    for const in definitions.constants:
        # Prepare constant value for embedding. This will be assigned as `self.CONSTANT_NAME = self._eval_expr(...)`
        const_value_str = _prepare_expr_for_embedding(const.value)
        init_definitions_code.append(f"self.{const.name} = self._eval_expr('{const_value_str}', {{}})")

    for func in definitions.functions:
        param_names = [p_name for _, p_name in func.parameters]
        params_str = ", ".join(param_names)
        param_context_str = ", ".join([f"'{p}': {p}" for p in param_names])

        # Prepare func.body for embedding
        cleaned_func_body = _prepare_expr_for_embedding(func.body)
        class_method_definitions_code.append(f"    def {func.name}(self, {params_str}):")
        class_method_definitions_code.append(f"        return self._eval_expr('{cleaned_func_body}', {{{param_context_str}}})")
        class_method_definitions_code.append("") # Added blank line for readability

    for pred in definitions.predicates:
        param_names = [p_name for _, p_name in pred.parameters]
        params_str = ", ".join(param_names)
        param_context_str = ", ".join([f"'{p}': {p}" for p in param_names])

        # Prepare pred.body for embedding
        cleaned_pred_body = _prepare_expr_for_embedding(pred.body)
        class_method_definitions_code.append(f"    def {pred.name}(self, {params_str}):")
        class_method_definitions_code.append(f"        return bool(self._eval_expr('{cleaned_pred_body}', {{{param_context_str}}}))")
        class_method_definitions_code.append("") # Added blank line for readability


    # --- 2. Observation Space ---
    observation_vars = [v.name for v in variables.state_variables]
    observation_space_code = []
    low_bounds = []
    high_bounds = []
    for var in variables.state_variables:
        # Prepare bounds for embedding
        low_bounds.append(f"self._eval_expr('{_prepare_expr_for_embedding(var.lower_bound or '-np.inf')}', {{}})")
        high_bounds.append(f"self._eval_expr('{_prepare_expr_for_embedding(var.upper_bound or 'np.inf')}', {{}})")

    # Format multi-line Box definition for cleaner indentation by textwrap.indent
    observation_space_code.append(f"self.observation_space = spaces.Box(")
    observation_space_code.append(f"    low=np.array([{', '.join(low_bounds)}], dtype=np.float32),")
    observation_space_code.append(f"    high=np.array([{', '.join(high_bounds)}], dtype=np.float32),")
    observation_space_code.append(f"    dtype=np.float32")
    observation_space_code.append(f")")
    observation_space_code.append(f"self.observation_vars = {observation_vars}")

    # --- 3. Action Space & Action Application Logic ---
    action_space_dict_str = {}
    unindented_action_mapping_logic = [] # Lines for _apply_agent_action body, with 0 leading spaces initially

    def _process_action_node(node: ActionType, current_choice_level: int, all_nondet_vars: set) -> Tuple[Dict, List[str]]:
        """
        Recursively processes action structure to build action space and mapping logic.
        Generates lines with 0 leading spaces, which will be indented later.
        """
        space_part = {}
        logic_part = []

        if isinstance(node, ChoiceActionConfig):
            level_key = f"choice_{node.level_name}"
            space_part[level_key] = f"spaces.Discrete({len(node.choices)})"
            logic_part.append(f"# Process choice for level {current_choice_level}")
            logic_part.append(f"chosen_idx = action['{level_key}']")
            logic_part.append(f"if not (0 <= chosen_idx < {len(node.choices)}):")
            logic_part.append(f"    raise ValueError(f'Invalid choice index {{chosen_idx}} for level {current_choice_level}')")

            for i, choice_seq in enumerate(node.choices):
                logic_part.append(f"if chosen_idx == {i}:")
                # Recursive call with increased choice level
                seq_space, seq_logic = _process_action_node(choice_seq, current_choice_level + 1, all_nondet_vars)
                if seq_space:
                    space_part.update(seq_space)
                # Indent sub-logic by 4 spaces relative to the 'if chosen_idx == i:' block
                logic_part.extend([textwrap.indent(line, ' ' * 4) for line in seq_logic])
            return space_part, logic_part

        elif isinstance(node, SequenceActionConfig):
            for action_item in node.actions:
                # Recursive call for nested choices or hybrid program steps within a sequence
                # For controlFunc, HybridProgramStepConfig should not appear directly.
                if isinstance(action_item, ChoiceActionConfig):
                    nested_space, nested_logic = _process_action_node(action_item, current_choice_level + 1, all_nondet_vars)
                    space_part.update(nested_space)
                    logic_part.extend(nested_logic)
                elif isinstance(action_item, NondeterministicAssignmentActionConfig):
                    var_name = action_item.variable
                    var_config = next((v for v in variables.state_variables + variables.auxiliary_variables if v.name == var_name), None)
                    if not var_config:
                        raise ValueError(f"Nondeterministic assignment to undeclared variable: {var_name}")

                    # Prepare bounds for embedding
                    low_bound_str = _prepare_expr_for_embedding(var_config.lower_bound or "-np.inf")
                    high_bound_str = _prepare_expr_for_embedding(var_config.upper_bound or "np.inf")

                    action_key = f"assign_{var_name}"
                    # Using _eval_expr for consistency
                    space_part[action_key] = f"spaces.Box(low=self._eval_expr('{low_bound_str}', {{}}), high=self._eval_expr('{high_bound_str}', {{}}), shape=(), dtype=np.float32)"
                    logic_part.append(f"self.state['{var_name}'] = action['{action_key}']")
                    all_nondet_vars.add(var_name) # Add to the set of all nondet vars

                elif isinstance(action_item, DeterministicAssignmentActionConfig):
                    var_name = action_item.variable
                    value_expr = action_item.value
                    # Prepare value_expr for embedding
                    cleaned_value_expr = _prepare_expr_for_embedding(value_expr)
                    logic_part.append(f"self.state['{var_name}'] = float(self._eval_expr('{cleaned_value_expr}', self.state))")

                elif isinstance(action_item, ActionConstraintConfig):
                    formula = action_item.formula
                    # Prepare formula for embedding
                    cleaned_formula = _prepare_expr_for_embedding(formula)
                    logic_part.append(f"if not self._eval_expr('{cleaned_formula}', self.state):")
                    logic_part.append(f"    self.terminated = True # Invalid action leads to termination")
                    logic_part.append(f"    self.reward -= 100.0 # Penalty for invalid action")
                    logic_part.append(f"    return False # Indicate invalid action")
                elif isinstance(action_item, CallHybridProgramConfig):
                    # This should not happen for controlFunc as it's the top-level HP call
                    raise ValueError(f"Call to hybrid program '{action_item.hp_name}' found directly in agent's action structure. "
                                     "This is not supported here; it should be wrapped in a Sequence or Choice if part of agent's action.")
                else:
                    raise ValueError(f"Unsupported action item type in sequence: {type(action_item)}")
            return space_part, logic_part
        else:
            raise ValueError(f"Unsupported action node type at root of _process_action_node: {type(node)}")

    all_nondet_vars = set()
    if control_func.action_space_config:
        # Start processing from choice level 0
        action_space_dict_str, unindented_action_mapping_logic = _process_action_node(control_func.action_space_config, 0, all_nondet_vars)

        # After processing all branches, ensure all nondeterministic variables from the overall action space
        # are assigned, even if a chosen branch doesn't explicitly use a particular nondeterministic assignment.
        # This is crucial for Gymnasium's Dict space where all keys must be present in the action.
        # This logic needs to be applied *after* the main choice processing.
        # It's best to handle this by iterating through the main choice branches again.
        if isinstance(control_func.action_space_config, ChoiceActionConfig):
            # This is a bit complex to insert correctly after the fact.
            # A more robust way might be to generate the full action space first,
            # then iterate through all possible choice paths to determine which
            # nondeterministic assignments are missing, and insert default assignments.
            # For now, we'll assume that the KeYmaera X model ensures that all
            # nondeterministic variables are assigned *somewhere* in each choice path
            # if they are part of the overall action space.
            # If this assumption proves false, we'd need to re-evaluate how to handle
            # unassigned nondeterministic variables within a choice branch.
            pass # Skipping complex insertion for now, relying on model structure.
        
    # Convert action_space_dict_str to a string representation for code generation
    action_space_str_parts = []
    for k, v in action_space_dict_str.items():
        action_space_str_parts.append(f"'{k}': {v}")
    action_space_code_gen_str = "{" + ", ".join(action_space_str_parts) + "}"

    init_action_space_code = []
    init_action_space_code.append(f"self.action_space = spaces.Dict({action_space_code_gen_str})")

    apply_action_method_code = []
    apply_action_method_code.append("    def _apply_agent_action(self, action):")
    apply_action_method_code.append("        # Apply agent's chosen action to the environment state")
    apply_action_method_code.append("        # Returns True if action was valid, False otherwise (e.g., violated constraint)")
    apply_action_method_code.append("        self.terminated = False # Reset termination flag for action application")
    apply_action_method_code.append("        self.reward = 0.0 # Reset reward for action application")
    # Indent the logic part by 8 spaces (for method body)
    apply_action_method_code.extend([textwrap.indent(line, ' ' * 8) for line in unindented_action_mapping_logic])
    apply_action_method_code.append("        return not self.terminated # Return True if no termination occurred due to action")


    # --- 4. ODE Integration Logic ---
    ode_integration_code = []
    ode_integration_code.append("    def _ode_system(self, t, y):")
    ode_integration_code.append(f"        state_dict = {{k: v for k, v in zip(self.observation_vars, y)}}")
    ode_integration_code.append(f"        return {sympy_evaluator.get_ode_function_body_str(step_func.odes, variables.state_variables)}")
    ode_integration_code.append("")

    ode_integration_code.append("    def _ode_event(self, t, y):")
    ode_integration_code.append(f"        state_dict = {{k: v for k, v in zip(self.observation_vars, y)}}")
    ode_integration_code.append(f"        # Event occurs when evolution domain becomes false (crosses 0 from positive)")
    # get_event_function_body_str now returns the prepared formula string
    event_formula = sympy_evaluator.get_event_function_body_str(step_func.evolution_domain)
    ode_integration_code.append(f"        event_condition = self._eval_expr('{event_formula}', state_dict)")
    ode_integration_code.append(f"        return event_condition")
    # Removed _ode_event.terminal and _ode_event.direction from here
    ode_integration_code.append("")

    ode_integration_code.append("    def _create_ode_event_function(self):")
    ode_integration_code.append("        def event_wrapper(t, y):")
    ode_integration_code.append("            return self._ode_event(t, y)")
    ode_integration_code.append("        event_wrapper.terminal = True")
    ode_integration_code.append("        event_wrapper.direction = -1")
    ode_integration_code.append("        return event_wrapper")
    ode_integration_code.append("")

    ode_integration_code.append("    def _integrate_dynamics(self, dt: float):")
    ode_integration_code.append(f"        y0 = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)")
    # t_span_str is not directly from step_func.duration, but from the dt passed to integrate_dynamics
    ode_integration_code.append(f"        t_span = (0, dt)")
    # Use self._ode_event_func which is set in __init__
    ode_integration_code.append("        sol = solve_ivp(self._ode_system, t_span, y0, events=self._ode_event_func, dense_output=True, rtol=1e-6, atol=1e-9)")
    ode_integration_code.append("        if sol.status == 0: # Integration successful, no event")
    ode_integration_code.append("            final_y = sol.y[:, -1]")
    ode_integration_code.append("            self.current_time += dt")
    ode_integration_code.append("        elif sol.status == 1: # Event occurred")
    ode_integration_code.append("            if sol.y_events[0].size > 0 and sol.t_events[0].size > 0:")
    ode_integration_code.append("                final_y = sol.y_events[0][-1]")
    ode_integration_code.append("                self.current_time += sol.t_events[0][-1]")
    ode_integration_code.append("            else:")
    ode_integration_code.append("                final_y = y0")
    ode_integration_code.append("                self.current_time += dt")
    ode_integration_code.append("            self.terminated = True")
    ode_integration_code.append("        else:")
    ode_integration_code.append("            final_y = y0")
    ode_integration_code.append("            self.terminated = True")
    ode_integration_code.append(f"            self.logger.error(f\"ODE integration failed: {{sol.message}}\")") # Changed print to self.logger.error
    ode_integration_code.append("")
    ode_integration_code.append(f"        for i, var_name in enumerate(self.observation_vars):")
    ode_integration_code.append(f"            self.state[var_name] = final_y[i]")
    ode_integration_code.append("")

    # --- 5. Problem Configuration (reset, step logic) ---
    reset_method_body_code = []
    reset_method_body_code.append("# Initialize state variables")
    for var in variables.state_variables + variables.auxiliary_variables:
        # Prepare initial value for embedding
        initial_value_str = _prepare_expr_for_embedding(var.lower_bound or "0.0") # Use lower bound as initial value
        reset_method_body_code.append(f"self.state['{var.name}'] = float(self._eval_expr('{initial_value_str}', {{}}))")
    reset_method_body_code.append("")
    reset_method_body_code.append("# Apply precondition (e.g., sample initial state satisfying it)")
    precondition_formula = problem.precondition.formula
    # Prepare precondition_formula for embedding
    cleaned_precondition_formula = _prepare_expr_for_embedding(precondition_formula)
    reset_method_body_code.append(f"# For now, we assume precondition defines a fixed initial state or is satisfied by default init.")
    reset_method_body_code.append(f"# A more advanced generator would sample until self._eval_expr('{cleaned_precondition_formula}', self.state) is True.")
    reset_method_body_code.append(f"if not self._eval_expr('{cleaned_precondition_formula}', self.state):")
    reset_method_body_code.append(f"    self.logger.warning(f\"Initial state does not satisfy precondition: {cleaned_precondition_formula}\")") # Changed print to self.logger.warning
    reset_method_body_code.append(f"    # Potentially re-sample or raise error, or terminate immediately")
    reset_method_body_code.append(f"    # self.terminated = True # Example: Terminate if precondition not met")
    reset_method_body_code.append("")
    reset_method_body_code.append("self.current_time = 0.0")
    reset_method_body_code.append("self.terminated = False")
    reset_method_body_code.append("self.truncated = False")
    reset_method_body_code.append("self.reward = 0.0")
    reset_method_body_code.append(f"observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)")
    reset_method_body_code.append("info = {}")
    reset_method_body_code.append("return observation, info")

    # Code for processing loop_body actions within the step method
    step_loop_body_processing_code = []
    step_loop_body_processing_code.append(f"# Execute the problem's loop_body (environment's internal evolution)")
    # Prepare problem.loop_execution.iterations for embedding
    step_loop_body_processing_code.append(f"num_loop_iterations_str = '{_prepare_expr_for_embedding(problem.loop_execution.iterations)}'")
    # Handle 'inf' directly without _eval_expr
    step_loop_body_processing_code.append(f"if num_loop_iterations_str == 'inf':")
    step_loop_body_processing_code.append(f"    num_loop_iterations = 1 # Assuming one integration step if 'inf' is specified")
    step_loop_body_processing_code.append(f"else:")
    step_loop_body_processing_code.append(f"    num_loop_iterations = int(self._eval_expr(num_loop_iterations_str, self.state))")
    step_loop_body_processing_code.append(f"if num_loop_iterations <= 0: num_loop_iterations = 1 # Default to at least one iteration")
    step_loop_body_processing_code.append("")
    step_loop_body_processing_code.append(f"for _ in range(num_loop_iterations):")
    step_loop_body_processing_code.append(f"    if self.terminated or self.truncated: break") # Stop if already terminated/truncated

    step_loop_body_processing_code.append(f"    current_dt = self.step_func_duration / num_loop_iterations")
    step_loop_body_processing_code.append(f"    # Process each action defined in the problem's loop_body")
    
    # Generate code for each flattened loop action
    inner_loop_actions_code_raw = []
    for loop_action_item in _flatten_actions(problem.loop_execution.loop_body.actions):
        # Handle CallHybridProgramConfig (e.g., controlFunc)
        if isinstance(loop_action_item, CallHybridProgramConfig):
            if loop_action_item.hp_name == control_func.name:
                raise ValueError(
                    f"Semantic Error: Call to hybrid program '{loop_action_item.hp_name}' "
                    f"found in the problem's loop_body. The '{control_func.name}' is "
                    f"reserved for defining the agent's action and is applied externally "
                    f"to the environment's step. It should not be part of the environment's "
                    f"internal evolution defined by the loop_body. "
                    f"Please check your KeYmaera X problem definition to ensure the loop_body "
                    f"only contains environment dynamics (e.g., calls to '{step_func.name}', "
                    f"deterministic assignments, or constraints)."
                )
            else:
                raise NotImplementedError(
                    f"Call to hybrid program '{loop_action_item.hp_name}' not supported "
                    f"in the problem's loop_body. Only '{step_func.name}' (represented "
                    f"as HybridProgramStepConfig) is currently supported for hybrid program "
                    f"calls within the loop_body."
                )
        # Handle HybridProgramStepConfig (expected for step_func)
        elif isinstance(loop_action_item, HybridProgramStepConfig):
            if loop_action_item.hp_name == step_func.name:
                inner_loop_actions_code_raw.append(f"self._integrate_dynamics(current_dt)")
                if loop_action_item.invariant_formula:
                    cleaned_invariant_formula = _prepare_expr_for_embedding(loop_action_item.invariant_formula)
                    inner_loop_actions_code_raw.append(f"if not self._eval_expr('{cleaned_invariant_formula}', self.state):")
                    inner_loop_actions_code_raw.append(f"    self.terminated = True # Invariant violation leads to termination")
                    inner_loop_actions_code_raw.append(f"    self.reward -= 100.0 # Penalty for invariant violation")
                    inner_loop_actions_code_raw.append(f"    break # Exit loop if invariant violated")
            else:
                raise NotImplementedError(
                    f"Call to hybrid program '{loop_action_item.hp_name}' not supported "
                    f"in loop body. Expected '{step_func.name}' for continuous evolution."
                )
        # Handle DeterministicAssignmentActionConfig (environment updates)
        elif isinstance(loop_action_item, DeterministicAssignmentActionConfig):
            var_name = loop_action_item.variable
            value_expr = loop_action_item.value
            cleaned_value_expr = _prepare_expr_for_embedding(value_expr)
            inner_loop_actions_code_raw.append(f"self.state['{var_name}'] = float(self._eval_expr('{cleaned_value_expr}', self.state))")
        # Handle ActionConstraintConfig (environment constraints)
        elif isinstance(loop_action_item, ActionConstraintConfig):
            formula = loop_action_item.formula
            cleaned_formula = _prepare_expr_for_embedding(formula)
            inner_loop_actions_code_raw.append(f"if not self._eval_expr('{cleaned_formula}', self.state):")
            inner_loop_actions_code_raw.append(f"    self.terminated = True")
            inner_loop_actions_code_raw.append(f"    self.reward -= 100.0 # Penalty for violating loop constraint")
            inner_loop_actions_code_raw.append(f"    break # Exit loop if constraint violated")
        # Handle NondeterministicAssignmentActionConfig (agent's choice, not env evolution)
        elif isinstance(loop_action_item, NondeterministicAssignmentActionConfig):
            raise ValueError(
                f"Semantic Error: Nondeterministic assignment to '{loop_action_item.variable}' "
                f"found in the problem's loop_body. Nondeterministic assignments are typically "
                f"part of the agent's action space (controlFunc) and should not be part of "
                f"the environment's internal evolution loop_body."
            )
        # Handle ChoiceActionConfig (agent's choice, not env evolution)
        elif isinstance(loop_action_item, ChoiceActionConfig):
            raise ValueError(
                f"Semantic Error: Choice action (level '{loop_action_item.level_name}') "
                f"found in the problem's loop_body. Choice actions are typically part of "
                f"the agent's action space (controlFunc) and should not be part of "
                f"the environment's internal evolution loop_body."
            )
        # Catch any other unexpected ActionType
        else:
            raise ValueError(
                f"Unsupported or unexpected ActionType '{type(loop_action_item).__name__}' "
                f"found in the problem's loop_body. The loop_body should primarily contain "
                f"environment dynamics (e.g., calls to '{step_func.name}' as HybridProgramStepConfig, "
                f"deterministic assignments, or constraints)."
            )

        inner_loop_actions_code_raw.append(f"    if self.terminated or self.truncated: break") # Break loop if terminated by inner actions/integration
    inner_loop_actions_code_raw.append("") # Add a blank line for readability


    step_postcondition_code = []
    postcondition_formula = problem.postcondition.formula
    # Prepare postcondition_formula for embedding
    cleaned_postcondition_formula = _prepare_expr_for_embedding(postcondition_formula)
    step_postcondition_code.append(f"# Evaluate postcondition for reward and termination")
    step_postcondition_code.append(f"if self._eval_expr('{cleaned_postcondition_formula}', self.state):")
    step_postcondition_code.append(f"    self.reward += 1.0 # Example: reward for satisfying postcondition")
    step_postcondition_code.append(f"    self.terminated = True # Example: termination upon satisfying postcondition")
    step_postcondition_code.append(f"else:")
    step_postcondition_code.append(f"    self.reward -= 0.01 # Example: small penalty per step")
    step_postcondition_code.append("")
    step_postcondition_code.append(f"# Check for truncation (e.g., max steps, max time)")
    # Prepare step_func.duration for embedding
    max_duration_str = _prepare_expr_for_embedding(step_func.duration or "1.0") # Default to 1.0 if not specified
    step_postcondition_code.append(f"if self.current_time >= self._eval_expr('{max_duration_str}', {{}}):")
    step_postcondition_code.append(f"    self.truncated = True")
    step_postcondition_code.append(f"    self.reward -= 10.0 # Penalty for truncation")
    step_postcondition_code.append("")
    step_postcondition_code.append(f"observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)")
    step_postcondition_code.append("info = {}")
    step_postcondition_code.append("return observation, self.reward, self.terminated, self.truncated, info")


    # --- Pre-compute list literals as strings using repr() for embedding ---
    constants_names_list_repr = repr([c.name for c in definitions.constants])
    functions_names_list_repr = repr([f.name for f in definitions.functions])
    predicates_names_list_repr = repr([p.name for p in definitions.predicates])

    # --- Generate direct instantiation code for problem.loop_execution.loop_body.actions ---
    # FIRST: Flatten the loop body actions
    flattened_loop_body_actions = _flatten_actions(problem.loop_execution.loop_body.actions)

    init_loop_body_actions_instantiation_code = []
    for action_obj in flattened_loop_body_actions: # Iterate over flattened actions
        # Note: The logic above handles validation for these types.
        # This part is for generating the instantiation code for the _problem_loop_body_actions list.
        if isinstance(action_obj, CallHybridProgramConfig):
            init_loop_body_actions_instantiation_code.append(f"ast_config.CallHybridProgramConfig(hp_name='{action_obj.hp_name}')")
        elif isinstance(action_obj, HybridProgramStepConfig):
            invariant_str = f"'{_prepare_expr_for_embedding(action_obj.invariant_formula)}'" if action_obj.invariant_formula else "None"
            init_loop_body_actions_instantiation_code.append(f"ast_config.HybridProgramStepConfig(hp_name='{action_obj.hp_name}', invariant_formula={invariant_str})")
        elif isinstance(action_obj, DeterministicAssignmentActionConfig):
            value_str = _prepare_expr_for_embedding(action_obj.value)
            init_loop_body_actions_instantiation_code.append(f"ast_config.DeterministicAssignmentActionConfig(variable='{action_obj.variable}', value='{value_str}')")
        elif isinstance(action_obj, ActionConstraintConfig):
            formula_str = _prepare_expr_for_embedding(action_obj.formula)
            init_loop_body_actions_instantiation_code.append(f"ast_config.ActionConstraintConfig(formula='{formula_str}')")
        elif isinstance(action_obj, NondeterministicAssignmentActionConfig):
             # This should have been caught by the validation above, but for completeness in instantiation
             init_loop_body_actions_instantiation_code.append(f"ast_config.NondeterministicAssignmentActionConfig(variable='{action_obj.variable}')")
        elif isinstance(action_obj, ChoiceActionConfig):
             # This should have been caught by the validation above, but for completeness in instantiation
             # This would require recursively instantiating choices, which is not needed for _problem_loop_body_actions
             # as it's processed dynamically. For now, we'll raise an error here too if it somehow gets past.
             raise ValueError(f"Unexpected ChoiceActionConfig in flattened loop_body for instantiation: {action_obj.level_name}")
        else:
            # This should also have been caught by the validation above
            raise ValueError(f"Unsupported ActionType in flattened loop_body actions for direct code generation: {type(action_obj)}")

    # --- Assemble the full class code ---
    # Prepare blocks as single strings before embedding in f-string
    init_definitions_block = "\n".join(init_definitions_code)
    class_method_definitions_block = "\n".join(class_method_definitions_code)
    observation_space_block = "\n".join(observation_space_code)
    init_action_space_block = "\n".join(init_action_space_code)
    apply_action_method_block = "\n".join(apply_action_method_code)
    ode_integration_block = "\n".join(ode_integration_code)
    reset_method_body_block = "\n".join(reset_method_body_code)
    step_loop_body_processing_block = "\n".join(step_loop_body_processing_code)
    step_postcondition_block = "\n".join(step_postcondition_code)

    # Prepare step_func.duration for embedding in __init__
    init_step_func_duration_str = _prepare_expr_for_embedding(step_func.duration or "1.0")

    # Extract _eval_expr method into its own block
    # CRITICAL FIX: Simplified _eval_expr to rely fully on sympy.lambdify
    _eval_expr_code_lines = [
        "    def _eval_expr(self, expr_str: str, context: dict) -> Any:",
        "        # Handle direct numeric string values for efficiency, avoiding sympy overhead",
        "        try:",
        "            return float(expr_str)",
        "        except ValueError:",
        "            pass # Not a simple float, proceed with symbolic evaluation",
        "        ",
        "        if expr_str == 'None':",
        "            return None",
        "        if expr_str == 'np.inf' or expr_str == 'inf':",
        "            return float('inf')",
        "        if expr_str == '-np.inf' or expr_str == '-inf':",
        "            return float('-inf')",
        "        ",
        "        full_context = {**context}",
        "        # Add constant values (e.g., self.V, self.W) to the context for symbol lookup",
        "        for k_name in self.constants_names:",
        "            if hasattr(self, k_name):",
        "                full_context[k_name] = getattr(self, k_name)",
        "        ",
        "        # Prepare the expression string for SymPy parsing",
        "        processed_expr_str = expr_str.replace('^', '**')",
        "        processed_expr_str = processed_expr_str.replace('=', '==') # KeYmaera X uses '=' for equality in formulas",
        "        processed_expr_str = processed_expr_str.replace('&', 'and')",
        "        processed_expr_str = processed_expr_str.replace('|', 'or')",
        "        processed_expr_str = processed_expr_str.lstrip('?') # Remove KeYmaera X test operator",
        "        ",
        "        # Create a dictionary of SymPy symbols and functions for sympify's locals",
        "        # This helps sympify correctly parse the expression string, distinguishing variables from function calls.",
        "        sympy_locals_for_sympify = {}",
        "        for k, v in full_context.items():",
        "            # For user-defined functions/predicates, tell sympify they are functions",
        "            if k in self.kx_functions or k in self.kx_predicates:",
        "                sympy_locals_for_sympify[k] = sympy.Function(k)",
        "            else:",
        "                # Otherwise, treat them as symbols (variables or constants)",
        "                sympy_locals_for_sympify[k] = sympy.Symbol(k)",
        "        sympy_locals_for_sympify['inf'] = sympy.oo # Handle 'inf' as sympy infinity",
        "        sympy_locals_for_sympify['oo'] = sympy.oo",
        "        sympy_locals_for_sympify['np'] = sympy.Symbol('np') # Treat 'np' as a symbol for parsing, lambdify handles actual numpy",
        "        sympy_locals_for_sympify['abs'] = sympy.Function('abs') # Treat 'abs' as a function for parsing",
        "        ",
        "        try:",
        "            sympy_expr = sympy.sympify(processed_expr_str, locals=sympy_locals_for_sympify)",
        "            ",
        "            # Prepare modules for lambdify. This maps function names in the sympy_expr",
        "            # to their corresponding Python callables (e.g., self.stopDist, np.sqrt).",
        "            lambdify_modules_dict = {}",
        "            for func_name, py_callable in self.kx_functions.items():",
        "                lambdify_modules_dict[func_name] = py_callable",
        "            for pred_name, py_callable in self.kx_predicates.items():",
        "                lambdify_modules_dict[pred_name] = py_callable",
        "            # Add common functions/modules that might be used in expressions",
        "            lambdify_modules_dict['abs'] = abs # Python's built-in abs",
        "            lambdify_modules_dict['np'] = np   # The numpy module itself",
        "            ",
            "            # Identify free symbols (variables and constants) in the SymPy expression.",
            "            # These are the arguments that lambdify will expect numerical values for.",
            "            # Exclude SymPy Function objects from free_symbols, as their implementation is in lambdify_modules_dict.",
            "            variable_symbols_in_expr = [s for s in sympy_expr.free_symbols if not isinstance(s, sympy.Function)]",
            "            ",
            "            # Prepare the list of numerical values corresponding to the identified symbols.",
            "            lambdify_values = []",
            "            for s in variable_symbols_in_expr:",
            "                s_name = str(s)",
            "                if s_name in full_context:",
            "                    lambdify_values.append(full_context[s_name])",
            "                else:",
            "                    # If a symbol is in the expression but not in the context, it's an error.",
            "                    raise NameError(f\"Symbol '{s_name}' found in expression '{expr_str}' but not in evaluation context.\")",
            "            ",
            "            # Create a numerical function from the SymPy expression using lambdify.",
            "            # The 'modules' argument tells lambdify where to find implementations for functions (kx_functions/predicates, numpy).",
            "            numerical_func = sympy.lambdify(variable_symbols_in_expr, sympy_expr, modules=[lambdify_modules_dict, 'numpy'])",
            "            ",
            "            # Evaluate the numerical function with the prepared values.",
            "            return numerical_func(*lambdify_values)",
            "        ",
            "        except (sympy.SympifyError, SyntaxError, TypeError, NameError, KeyError) as e:",
            "            # Log the error with full context for debugging",
            "            self.logger.error(f\"Error evaluating expression '{expr_str}' (processed: '{processed_expr_str}') with context '{full_context}': {e}\")",
            "            raise ValueError(f\"Error evaluating expression '{expr_str}': {e}\") from e"
    ]


    class_template = f"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import sympy
from typing import List, Dict, Any, Optional, Union, Tuple
import logging # Re-import logging here for the generated code as well

# Import ast_config to make dataclass definitions available for _problem_loop_body_actions
import ast_config

class {env_name}(gym.Env):
    metadata = {{"render_modes": ["human"], "render_fps": 30}}

{textwrap.indent("".join(class_method_definitions_code), ' ' * 4)}

{textwrap.indent("".join(_eval_expr_code_lines), ' ' * 4)}

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__) # Added: Initialize logger for the generated class
        self.state = {{}} # Internal state dictionary
        self.current_time = 0.0
        self.terminated = False
        self.truncated = False
        self.reward = 0.0

        # --- A. Initialize lists/dicts of *names* and *callable references* ---
        # These are needed by _eval_expr to know *what* to look for.
        # They do NOT call _eval_expr themselves, so they can be set first.
        self.constants_names = {constants_names_list_repr}

        self.kx_functions = {{}}
        for name in {functions_names_list_repr}:
            if hasattr(self, name): # Ensure the method exists
                self.kx_functions[name] = getattr(self, name)
            else:
                raise AttributeError(f"Generated function '{{name}}' expected but not found as a method in class {{self.__class__.__name__}}. Check definition generation.")

        self.kx_predicates = {{}}
        for name in {predicates_names_list_repr}:
            if hasattr(self, name): # Ensure the method exists
                self.kx_predicates[name] = getattr(self, name)
            else:
                raise AttributeError(f"Generated predicate '{{name}}' expected but not found as a method in class {{self.__class__.__name__}}. Check definition generation.")

        # Initialize ODE event function wrapper (does not call _eval_expr directly yet)
        self._ode_event_func = self._create_ode_event_function()

        # Store the problem's loop_body actions as a list of instantiated dataclass objects
        self._problem_loop_body_actions = [
{textwrap.indent(",\n".join(init_loop_body_actions_instantiation_code), ' ' * 12)}
        ]

        # --- B. Assign constant *values* ---
        # This block sets `self.W`, `self.R`, etc. These calls to _eval_expr
        # will now find `self.constants_names`, `self.kx_functions`, `self.kx_predicates` initialized.
{textwrap.indent(init_definitions_block, ' ' * 8)}

        # --- C. Now, other attributes can be safely initialized ---
        # These may call _eval_expr and use constants whose values are now set.
        self.step_func_name = "{step_func.name}"
        self.step_func_duration = float(self._eval_expr('{init_step_func_duration_str}', {{}}))

        # Generated Observation Space (calls _eval_expr for bounds)
{textwrap.indent(observation_space_block, ' ' * 8)}

        # Generated Action Space (part 1: definition in __init__) (calls _eval_expr for bounds)
{textwrap.indent(init_action_space_block, ' ' * 8)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
{textwrap.indent(reset_method_body_block, ' ' * 8)}

    def step(self, action):
        # 1. Apply agent's discrete action (from control_func)
        action_valid = self._apply_agent_action(action)
        if not action_valid:
            # Action was invalid, environment terminated or heavily penalized
            observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)
            info = {{"action_invalid": True}}
            return observation, self.reward, self.terminated, self.truncated, info

{textwrap.indent(step_loop_body_processing_block, ' ' * 8)}

        # 3. Evaluate postcondition, truncation, and return
{textwrap.indent(step_postcondition_block, ' ' * 8)}

    def render(self):
        # Implement rendering logic here (e.g., matplotlib, pygame)
        pass

    def close(self):
        # Clean up resources
        pass

{textwrap.indent(apply_action_method_block, ' ' * 4)}

    # --- Generated ODE System and Event Functions ---
{textwrap.indent(ode_integration_block, ' ' * 4)}
"""
    return class_template