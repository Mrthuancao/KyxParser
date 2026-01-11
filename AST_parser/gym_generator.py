# gym_generator.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import sympy
import textwrap
from typing import List, Dict, Any, Optional, Union, Tuple

# Import all dataclass definitions from ast_config.py, including ActionType
from ast_config import (
    PyASTNode,
    ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig, DefinitionsConfig,
    VariableConfig, VariablesConfig,
    NondeterministicAssignmentActionConfig, DeterministicAssignmentActionConfig, ActionConstraintConfig,
    SequenceActionConfig, ChoiceActionConfig, ControlFuncConfig, CallHybridProgramConfig,
    ODEConfig, EvolutionDomainConfig, StepFuncConfig, HybridProgramConfig,
    PreconditionConfig, PostconditionConfig, LoopExecutionConfig, ProblemConfig,
    GymEnvironmentConfig,
    ActionType
)


# --- Helper for symbolic evaluation ---
class SympyEvaluator:
    """
    Helper class to manage symbolic variables and evaluate expressions during code generation.
    This class primarily prepares the SymPy symbols and function objects
    that will be used to parse expressions.
    """
    def __init__(self, state_variables: List[VariableConfig], auxiliary_variables: List[VariableConfig],
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

    def get_ode_function_body_str(self, odes: List[ODEConfig], state_variables: List[VariableConfig]):
        """
        Generates the string representation of the ODE system's return values
        for the generated _ode_system method.
        """
        ode_eq_list_str = []
        for var in state_variables:
            ode_config = next((o for o in odes if o.variable == var.name), None)
            if ode_config:
                # Use _eval_expr from the generated environment class
                ode_eq_list_str.append(f"self._eval_expr('{ode_config.equation}', state_dict)")
            else:
                ode_eq_list_str.append("0.0") # Variable not changing
        return f"np.array([{', '.join(ode_eq_list_str)}], dtype=np.float32)"

    def get_event_function_body_str(self, evolution_domain: EvolutionDomainConfig):
        """
        Generates the string representation of the event function's return value
        for the generated _ode_event method.
        """
        # Event occurs when evolution domain becomes false (crosses 0 from positive)
        # Join multiple constraints with 'and'
        if not evolution_domain or not evolution_domain.constraints:
            return "1.0" # No event if no constraints (always true)
        
        formula = " and ".join(f"({c})" for c in evolution_domain.constraints)
        return f"self._eval_expr('{formula}', state_dict)"

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
        const_value = f"'{const.value}'" if const.value is not None else "None"
        init_definitions_code.append(f"self.{const.name} = self._eval_expr({const_value}, {{}})")

    for func in definitions.functions:
        param_names = [p_name for _, p_name in func.parameters]
        params_str = ", ".join(param_names)
        param_context_str = ", ".join([f"'{p}': {p}" for p in param_names])
        
        class_method_definitions_code.append(f"def {func.name}(self, {params_str}):")
        class_method_definitions_code.append(f"    return self._eval_expr('{func.body}', {{{param_context_str}}})")

    for pred in definitions.predicates:
        param_names = [p_name for _, p_name in pred.parameters]
        params_str = ", ".join(param_names)
        param_context_str = ", ".join([f"'{p}': {p}" for p in param_names])
        
        class_method_definitions_code.append(f"def {pred.name}(self, {params_str}):")
        class_method_definitions_code.append(f"    return bool(self._eval_expr('{pred.body}', {{{param_context_str}}}))")


    # --- 2. Observation Space ---
    observation_vars = [v.name for v in variables.state_variables]
    observation_space_code = []
    low_bounds = []
    high_bounds = []
    for var in variables.state_variables:
        low = var.lower_bound if var.lower_bound else "-np.inf"
        high = var.upper_bound if var.upper_bound else "np.inf"
        low_bounds.append(f"self._eval_expr('{low}', {{}})")
        high_bounds.append(f"self._eval_expr('{high}', {{}})")

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

    def _process_action_node(node: ActionType) -> Tuple[Dict, List[str]]:
        """
        Recursively processes action structure to build action space and mapping logic.
        Generates lines with 0 leading spaces, which will be indented later.
        """
        space_part = {}
        logic_part = []
        
        if isinstance(node, ChoiceActionConfig):
            level_key = f"choice_{node.level_name}"
            space_part[level_key] = f"spaces.Discrete({len(node.choices)})"
            logic_part.append(f"# Process choice for level {node.level_name}")
            logic_part.append(f"chosen_idx = action['{level_key}']")
            logic_part.append(f"if not (0 <= chosen_idx < {len(node.choices)}):")
            logic_part.append(f"    raise ValueError(f'Invalid choice index {{chosen_idx}} for level {node.level_name}')")

            for i, choice_seq in enumerate(node.choices):
                logic_part.append(f"if chosen_idx == {i}:")
                seq_space, seq_logic = _process_action_node(choice_seq)
                if seq_space:
                    space_part.update(seq_space)
                # Indent sub-logic by 4 spaces relative to the 'if chosen_idx == i:' block
                logic_part.extend([textwrap.indent(line, ' ' * 4) for line in seq_logic])
            return space_part, logic_part

        elif isinstance(node, SequenceActionConfig):
            for action_item in node.actions:
                if isinstance(action_item, NondeterministicAssignmentActionConfig):
                    var_name = action_item.variable
                    var_config = next((v for v in variables.state_variables + variables.auxiliary_variables if v.name == var_name), None)
                    if not var_config:
                        raise ValueError(f"Nondeterministic assignment to undeclared variable: {var_name}")

                    low_bound_str = var_config.lower_bound if var_config.lower_bound else "-np.inf"
                    high_bound_str = var_config.upper_bound if var_config.upper_bound else "np.inf"

                    action_key = f"assign_{var_name}"
                    space_part[action_key] = f"spaces.Box(low=self._eval_expr('{low_bound_str}', {{}}), high=self._eval_expr('{high_bound_str}', {{}}), shape=(), dtype=np.float32)"
                    logic_part.append(f"self.state['{var_name}'] = action['{action_key}']")

                elif isinstance(action_item, DeterministicAssignmentActionConfig):
                    var_name = action_item.variable
                    value_expr = action_item.value
                    logic_part.append(f"self.state['{var_name}'] = float(self._eval_expr('{value_expr}', self.state))")

                elif isinstance(action_item, ActionConstraintConfig):
                    formula = action_item.formula
                    logic_part.append(f"if not self._eval_expr('{formula}', self.state):")
                    logic_part.append(f"    self.terminated = True # Invalid action leads to termination")
                    logic_part.append(f"    self.reward -= 100.0 # Penalty for invalid action")
                    logic_part.append(f"    return False # Indicate invalid action")
                elif isinstance(action_item, CallHybridProgramConfig):
                    raise ValueError(f"Call to hybrid program '{action_item.hp_name}' found in agent's action structure. "
                                     "This is not supported here; it should be in the problem's loop_body.")
            return space_part, logic_part
        else:
            raise ValueError(f"Unsupported action node type: {type(node)}")

    if control_func.action_structure:
        action_space_dict_str, unindented_action_mapping_logic = _process_action_node(control_func.action_structure)

    # Convert action_space_dict_str to a string representation for code generation
    action_space_str_parts = []
    for k, v in action_space_dict_str.items():
        action_space_str_parts.append(f"'{k}': {v}")
    action_space_code_gen_str = "{" + ", ".join(action_space_str_parts) + "}"

    init_action_space_code = []
    init_action_space_code.append(f"self.action_space = spaces.Dict({action_space_code_gen_str})")

    apply_action_method_code = []
    apply_action_method_code.append("def _apply_agent_action(self, action):")
    apply_action_method_code.append("    # Apply agent's chosen action to the environment state")
    apply_action_method_code.append("    # Returns True if action was valid, False otherwise (e.g., violated constraint)")
    apply_action_method_code.append("    self.terminated = False # Reset termination flag for action application")
    apply_action_method_code.append("    self.reward = 0.0 # Reset reward for action application")
    # Indent the logic part by 4 spaces (for method body)
    apply_action_method_code.extend([textwrap.indent(line, ' ' * 4) for line in unindented_action_mapping_logic])
    apply_action_method_code.append("    return not self.terminated # Return True if no termination occurred due to action")


    # --- 4. ODE Integration Logic ---
    ode_integration_code = []
    ode_integration_code.append("def _ode_system(self, t, y):")
    ode_integration_code.append(f"    state_dict = {{k: v for k, v in zip(self.observation_vars, y)}}")
    ode_integration_code.append(f"    return {sympy_evaluator.get_ode_function_body_str(step_func.odes, variables.state_variables)}")
    ode_integration_code.append("")

    ode_integration_code.append("def _ode_event(self, t, y):")
    ode_integration_code.append(f"    state_dict = {{k: v for k, v in zip(self.observation_vars, y)}}")
    ode_integration_code.append(f"    # Event occurs when evolution domain becomes false (crosses 0 from positive)")
    if step_func.evolution_domain and step_func.evolution_domain.constraints:
        ode_integration_code.append(f"    return {sympy_evaluator.get_event_function_body_str(step_func.evolution_domain)}")
    else:
        ode_integration_code.append("    return 1.0 # No evolution domain specified, so no event")

    ode_integration_code.append("    _ode_event.terminal = True")
    ode_integration_code.append("    _ode_event.direction = -1")
    ode_integration_code.append("")

    ode_integration_code.append("def _integrate_dynamics(self, dt: float):")
    ode_integration_code.append(f"    y0 = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)")
    ode_integration_code.append("    t_span = (0, dt)")
    ode_integration_code.append("    sol = solve_ivp(self._ode_system, t_span, y0, events=self._ode_event, dense_output=True, rtol=1e-6, atol=1e-9)")
    ode_integration_code.append("    if sol.status == 0: # Integration successful, no event")
    ode_integration_code.append("        final_y = sol.y[:, -1]")
    ode_integration_code.append("        self.current_time += dt")
    ode_integration_code.append("    elif sol.status == 1: # Event occurred")
    ode_integration_code.append("        if sol.y_events[0].size > 0 and sol.t_events[0].size > 0:")
    ode_integration_code.append("            final_y = sol.y_events[0][-1]")
    ode_integration_code.append("            self.current_time += sol.t_events[0][-1]")
    ode_integration_code.append("        else:")
    ode_integration_code.append("            final_y = y0")
    ode_integration_code.append("            self.current_time += dt")
    ode_integration_code.append("        self.terminated = True")
    ode_integration_code.append("    else:")
    ode_integration_code.append("        final_y = y0")
    ode_integration_code.append("        self.terminated = True")
    ode_integration_code.append("        print(f\"ODE integration failed: {{sol.message}}\")")
    ode_integration_code.append("")
    ode_integration_code.append(f"    for i, var_name in enumerate(self.observation_vars):")
    ode_integration_code.append(f"        self.state[var_name] = final_y[i]")
    ode_integration_code.append("")

    # --- 5. Problem Configuration (reset, step logic) ---
    reset_method_body_code = []
    reset_method_body_code.append("# Initialize state variables")
    for var in variables.state_variables + variables.auxiliary_variables:
        initial_value_str = var.lower_bound if var.lower_bound else "0.0"
        reset_method_body_code.append(f"self.state['{var.name}'] = float(self._eval_expr('{initial_value_str}', {{}}))")
    reset_method_body_code.append("")
    reset_method_body_code.append("# Apply precondition (e.g., sample initial state satisfying it)")
    precondition_formula = problem.precondition.formula
    reset_method_body_code.append(f"# For now, we assume precondition defines a fixed initial state or is satisfied by default init.")
    reset_method_body_code.append(f"# A more advanced generator would sample until self._eval_expr('{precondition_formula}', self.state) is True.")
    reset_method_body_code.append(f"if not self._eval_expr('{precondition_formula}', self.state):")
    reset_method_body_code.append(f"    print(f\"Warning: Initial state does not satisfy precondition: {precondition_formula}\")")
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
    step_loop_body_processing_code.append(f"num_loop_iterations_str = '{problem.loop_execution.iterations}'")
    step_loop_body_processing_code.append(f"num_loop_iterations = int(self._eval_expr(num_loop_iterations_str, self.state))")
    step_loop_body_processing_code.append(f"if num_loop_iterations <= 0: num_loop_iterations = 1 # Default to at least one iteration")
    step_loop_body_processing_code.append("")
    step_loop_body_processing_code.append(f"for _ in range(num_loop_iterations):")
    
    # Inner loop body actions need to be indented relative to the 'for' loop
    inner_loop_actions_code = []
    inner_loop_actions_code.append(f"# Process each action in the loop_body")
    for loop_action_item in problem.loop_execution.loop_body.actions:
        if isinstance(loop_action_item, CallHybridProgramConfig):
            if loop_action_item.hp_name == step_func.name:
                inner_loop_actions_code.append(f"if self.terminated: break # Break if previous action/integration terminated")
                inner_loop_actions_code.append(f"self._integrate_dynamics(self.step_func_duration)")
            else:
                inner_loop_actions_code.append(f"raise NotImplementedError(f\"Call to hybrid program '{loop_action_item.hp_name}' not supported in loop body.\")")
        elif isinstance(loop_action_item, DeterministicAssignmentActionConfig):
            var_name = loop_action_item.variable
            value_expr = loop_action_item.value
            inner_loop_actions_code.append(f"self.state['{var_name}'] = float(self._eval_expr('{value_expr}', self.state))")
        elif isinstance(loop_action_item, ActionConstraintConfig):
            formula = loop_action_item.formula
            inner_loop_actions_code.append(f"if not self._eval_expr('{formula}', self.state):")
            inner_loop_actions_code.append(f"    self.terminated = True")
            inner_loop_actions_code.append(f"    self.reward -= 100.0 # Penalty for violating loop constraint")
            inner_loop_actions_code.append(f"    break # Exit loop if constraint violated")
        else:
            inner_loop_actions_code.append(f"raise ValueError(f\"Unsupported action type in loop_body: {type(loop_action_item)}\")")
    inner_loop_actions_code.append(f"if self.terminated: break # Break outer loop if terminated by inner actions/integration")
    inner_loop_actions_code.append("")

    step_loop_body_processing_code.extend([textwrap.indent(line, ' ' * 4) for line in inner_loop_actions_code])


    step_postcondition_code = []
    postcondition_formula = problem.postcondition.formula
    step_postcondition_code.append(f"# Evaluate postcondition for reward and termination")
    step_postcondition_code.append(f"post_eval_result = self._eval_expr('{postcondition_formula}', self.state)")
    step_postcondition_code.append(f"if post_eval_result:")
    step_postcondition_code.append(f"    self.reward += 1.0 # Example: reward for satisfying postcondition")
    step_postcondition_code.append(f"    self.terminated = True # Example: termination upon satisfying postcondition")
    step_postcondition_code.append(f"else:")
    step_postcondition_code.append(f"    self.reward -= 0.01 # Example: small penalty per step")
    step_postcondition_code.append("")
    step_postcondition_code.append(f"# Check for truncation (e.g., max steps, max time)")
    max_duration_str = step_func.duration if step_func.duration else "10.0"
    step_postcondition_code.append(f"if self.current_time >= self._eval_expr('{max_duration_str}', {{}}):")
    step_postcondition_code.append(f"    self.truncated = True")
    step_postcondition_code.append(f"    self.reward -= 10.0 # Penalty for truncation")
    step_postcondition_code.append("")
    step_postcondition_code.append(f"observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)")
    step_postcondition_code.append("info = {}")
    step_postcondition_code.append("return observation, self.reward, self.terminated, self.truncated, info")


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


    class_template = f"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import sympy
from typing import List, Dict, Any, Optional, Union, Tuple

class {env_name}(gym.Env):
    metadata = {{"render_modes": ["human"], "render_fps": 30}}

    def __init__(self):
        super().__init__()
        self.state = {{}} # Internal state dictionary
        self.current_time = 0.0
        self.terminated = False
        self.truncated = False
        self.reward = 0.0

        # --- Store configuration for step method ---
        self.step_func_name = "{step_func.name}"
        self.step_func_duration = float(self._eval_expr('{step_func.duration if step_func.duration else "0.1"}', {{}})) # Default to 0.1 if not specified

        # --- Generated Definitions (Constants in __init__) ---
{textwrap.indent(init_definitions_block, ' ' * 8)}

        self.constants_names = [{', '.join([f"'{c.name}'" for c in definitions.constants])}]

        # Store generated functions/predicates as callables for _eval_expr context
        self.kx_functions = {{}}
        for name in [{', '.join([f"'{f.name}'" for f in definitions.functions])}]:
            self.kx_functions[name] = getattr(self, name)

        self.kx_predicates = {{}}
        for name in [{', '.join([f"'{p.name}'" for p in definitions.predicates])}]:
            self.kx_predicates[name] = getattr(self, name)

        # --- Generated Observation Space ---
{textwrap.indent(observation_space_block, ' ' * 8)}

        # --- Generated Action Space (part 1: definition in __init__) ---
{textwrap.indent(init_action_space_block, ' ' * 8)}

{textwrap.indent(class_method_definitions_block, ' ' * 4)}

{textwrap.indent(apply_action_method_block, ' ' * 4)}

    def _eval_expr(self, expr_str: str, context: dict) -> Any:
        
        if expr_str == "None": # Handle cases where Optional[str] is None
            return None

        # Prepare the full context for evaluation
        # This includes numerical values for variables/constants
        # and callable Python functions for kx_functions/kx_predicates
        full_context = {{**context}}
        for k_name in self.constants_names:
            full_context[k_name] = getattr(self, k_name)
        full_context.update(self.kx_functions)
        full_context.update(self.kx_predicates)

        # Create a dictionary of sympy symbols for variables and constants
        # and sympy Function objects for the defined functions/predicates.
        # This allows sympy.sympify to correctly parse expressions like 'func(x,y)'.
        sympy_namespace = {{}}
        for k, v in full_context.items():
            if isinstance(v, (int, float, np.number, bool)): # Include bool for direct evaluation
                sympy_namespace[k] = sympy.symbols(k)
            elif callable(v):
                sympy_namespace[k] = sympy.Function(k)
            else:
                # Fallback for other types, treat as symbol
                sympy_namespace[k] = sympy.symbols(k)

        # Replace common operators if necessary
        expr_str = expr_str.replace('^', '**')

        try:
            # Parse the expression using the custom sympy_namespace
            sympy_expr = sympy.sympify(expr_str, locals=sympy_namespace)

            # Separate free symbols into variables/constants and functions
            free_symbols_in_expr = list(sympy_expr.free_symbols)
            function_symbols_in_expr = [s for s in free_symbols_in_expr if isinstance(s, sympy.Function)]
            variable_symbols_in_expr = [s for s in free_symbols_in_expr if not isinstance(s, sympy.Function)]

            # Prepare arguments and values for lambdify
            lambdify_args = variable_symbols_in_expr + function_symbols_in_expr
            
            # Values for variables/constants
            lambdify_values = [full_context[str(s)] for s in variable_symbols_in_expr]
            
            # Modules for functions/predicates (Python callables)
            lambdify_modules = {{str(s): full_context[str(s)] for s in function_symbols_in_expr}}
            
            if not lambdify_args:
                # If no free symbols or function calls, just evaluate numerically
                # Use subs for constants if any, then evalf
                subs_dict_for_evalf = {{sympy_namespace[k]: v for k, v in full_context.items() if isinstance(v, (int, float, np.number, bool))}}
                result = sympy_expr.evalf(subs=subs_dict_for_evalf)
            else:
                # Create a numerical function from the sympy expression
                # The 'modules' argument allows passing Python callables for functions
                numerical_func = sympy.lambdify(lambdify_args, sympy_expr, modules=[lambdify_modules, 'numpy'])
                result = numerical_func(*lambdify_values)

            # Convert boolean sympy results to Python bool
            if result is sympy.true: return True
            if result is sympy.false: return False
            return result
        except (sympy.SympifyError, SyntaxError, TypeError, NameError) as e:
            print(f"Error evaluating expression '{{expr_str}}' with context '{{full_context}}': {{e}}")
            # For robustness, return False for boolean expressions on error, or raise an exception
            return False # Default to False on error for boolean contexts

    # --- Generated ODE System and Event Functions ---
{textwrap.indent(ode_integration_block, ' ' * 4)}

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
"""
    return class_template