# gym_generator.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import sympy
from sympy.parsing.mathematica import parse_mathematica # Not used in current _eval_expr, but kept if needed
import textwrap
from typing import List, Dict, Any, Optional, Union, Tuple

# Import all dataclass definitions from ast_config.py
from ast_config import (
    PyASTNode,
    ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig, DefinitionsConfig,
    VariableConfig, VariablesConfig,
    NondeterministicAssignmentActionConfig, DeterministicAssignmentActionConfig, ActionConstraintConfig,
    SequenceActionConfig, ChoiceActionConfig, ControlFuncConfig,
    ODEConfig, EvolutionDomainConfig, StepFuncConfig,
    PreconditionConfig, PostconditionConfig, LoopExecutionConfig, ProblemConfig,
    GymEnvironmentConfig
)


# --- Helper for symbolic evaluation ---
class SympyEvaluator:
    """
    Helper class to manage symbolic variables and evaluate expressions.
    This class primarily prepares the SymPy symbols and lambdified functions
    that will be used in the generated Gymnasium environment.
    """
    def __init__(self, state_variables: List[VariableConfig], auxiliary_variables: List[VariableConfig],
                 constants: List[ConstantDefinitionConfig], functions: List[FunctionDefinitionConfig],
                 predicates: List[PredicateDefinitionConfig]):
        self.sympy_symbols = {}
        self.known_constants_raw = {} # Store raw strings for constants
        self.known_functions = {}
        self.known_predicates = {}

        # Create sympy symbols for all variables
        for var in state_variables + auxiliary_variables:
            self.sympy_symbols[var.name] = sympy.symbols(var.name)

        # Process constants (store raw string, evaluation happens in generated env)
        for const in constants:
            self.known_constants_raw[const.name] = const.value
            self.sympy_symbols[const.name] = sympy.symbols(const.name) # Also add as symbol for expressions

        # Process functions and predicates (store as dicts for now, will be lambdified in generated env)
        for func in functions:
            self.known_functions[func.name] = {
                'parameters': func.parameters,
                'body': func.body
            }
        for pred in predicates:
            self.known_predicates[pred.name] = {
                'parameters': pred.parameters,
                'body': pred.body
            }

    def _parse_expression(self, expr_str: str):
        """Parses a mathematical expression string into a sympy expression."""
        # Replace common operators if necessary, or use a more robust parser
        expr_str = expr_str.replace('^', '**')
        try:
            return sympy.sympify(expr_str, locals=self.sympy_symbols)
        except (sympy.SympifyError, SyntaxError):
            raise ValueError(f"Could not parse expression: {expr_str}")

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
        return f"self._eval_expr('{evolution_domain.formula}', state_dict)"

def generate_gym_environment_code(env_config: GymEnvironmentConfig) -> str:
    """
    Generates the Python code for a Gymnasium environment from the GymEnvironmentConfig.
    """
    env_name = env_config.env_name
    definitions = env_config.definitions
    variables = env_config.variables
    control_func = env_config.control_func
    step_func = env_config.step_func
    problem = env_config.problem

    # Initialize SympyEvaluator to prepare symbolic context
    sympy_evaluator = SympyEvaluator(
        state_variables=variables.state_variables,
        auxiliary_variables=variables.auxiliary_variables,
        constants=definitions.constants,
        functions=definitions.functions,
        predicates=definitions.predicates
    )

    # --- 1. Generate Definitions (Constants, Functions, Predicates) ---
    definitions_code = []
    for const in definitions.constants:
        # Constants are evaluated using _eval_expr in the generated class
        definitions_code.append(f"        self.{const.name} = self._eval_expr('{const.value}', {{}})")

    for func in definitions.functions:
        params_str = ", ".join(func.parameters)
        # Create sympy expression for the function body
        sympy_expr = sympy_evaluator._parse_expression(func.body)
        sympy_params = [sympy_evaluator.sympy_symbols[p] for p in func.parameters]

        # Handle functions with no parameters for lambdify
        if not sympy_params:
            # lambdify needs at least one argument, so use a dummy and call with 0
            sympy_params = [sympy.symbols('_dummy_')]
            lambdified_func_str = f"sympy.lambdify({tuple(sympy_params)}, {sympy_expr}, 'numpy')(0)"
            definitions_code.append(f"        def {func.name}(self, {params_str}): return {lambdified_func_str}")
        else:
            lambdified_func_str = f"sympy.lambdify({tuple(sympy_params)}, {sympy_expr}, 'numpy')({params_str})"
            definitions_code.append(f"        def {func.name}(self, {params_str}): return {lambdified_func_str}")
        definitions_code[-1] = textwrap.indent(textwrap.dedent(definitions_code[-1]), ' ' * 8)


    for pred in definitions.predicates:
        params_str = ", ".join(pred.parameters)
        sympy_expr = sympy_evaluator._parse_expression(pred.body)
        sympy_params = [sympy_evaluator.sympy_symbols[p] for p in pred.parameters]

        if not sympy_params:
            sympy_params = [sympy.symbols('_dummy_')]
            lambdified_pred_str = f"sympy.lambdify({tuple(sympy_params)}, {sympy_expr}, 'numpy')(0)"
            definitions_code.append(f"        def {pred.name}(self, {params_str}): return bool({lambdified_pred_str})")
        else:
            lambdified_pred_str = f"sympy.lambdify({tuple(sympy_params)}, {sympy_expr}, 'numpy')({params_str})"
            definitions_code.append(f"        def {pred.name}(self, {params_str}): return bool({lambdified_pred_str})")
        definitions_code[-1] = textwrap.indent(textwrap.dedent(definitions_code[-1]), ' ' * 8)


    # --- 2. Observation Space ---
    observation_vars = [v.name for v in variables.state_variables]
    observation_space_code = []
    low_bounds = []
    high_bounds = []
    for var in variables.state_variables:
        # Bounds are evaluated in the generated environment's _eval_expr
        low = var.lower_bound if var.lower_bound else "-np.inf"
        high = var.upper_bound if var.upper_bound else "np.inf"
        low_bounds.append(f"self._eval_expr('{low}', {{}})")
        high_bounds.append(f"self._eval_expr('{high}', {{}})")

    observation_space_code.append(f"        self.observation_space = spaces.Box(low=np.array([{', '.join(low_bounds)}], dtype=np.float32),")
    observation_space_code.append(f"                                            high=np.array([{', '.join(high_bounds)}], dtype=np.float32),")
    observation_space_code.append(f"                                            dtype=np.float32)")
    observation_space_code.append(f"        self.observation_vars = {observation_vars}")

    # --- 3. Action Space & Action Application Logic ---
    action_space_definition = {}
    action_mapping_logic = [] # Code to apply agent's action to self.state

    def _process_action_node(node: Union[SequenceActionConfig, ChoiceActionConfig], current_level: str = "") -> Tuple[Dict, List[str]]:
        """Recursively processes action structure to build action space and mapping logic."""
        space_part = {}
        logic_part = []
        if isinstance(node, ChoiceActionConfig):
            # This level represents a discrete choice
            level_key = f"choice_{node.level_name}"
            space_part[level_key] = f"spaces.Discrete({len(node.choices)})" # Store as string for code gen
            logic_part.append(f"        # Process choice for level {node.level_name}")
            logic_part.append(f"        chosen_idx = action['{level_key}']")
            logic_part.append(f"        if not (0 <= chosen_idx < {len(node.choices)}):")
            logic_part.append(f"            raise ValueError(f'Invalid choice index {{chosen_idx}} for level {node.level_name}')")

            for i, choice_seq in enumerate(node.choices):
                logic_part.append(f"        if chosen_idx == {i}:")
                seq_space, seq_logic = _process_action_node(choice_seq, current_level=f"{level_key}_{i}")
                if seq_space:
                    space_part.update(seq_space) # Merge sub-spaces
                logic_part.extend([textwrap.indent(line, ' ' * 4) for line in seq_logic])
            return space_part, logic_part

        elif isinstance(node, SequenceActionConfig):
            # This level is a sequence of assignments/constraints
            for action_item in node.actions:
                if isinstance(action_item, NondeterministicAssignmentActionConfig):
                    var_name = action_item.variable
                    # Find bounds for this variable
                    var_config = next((v for v in variables.state_variables + variables.auxiliary_variables if v.name == var_name), None)
                    low_bound_str = var_config.lower_bound if var_config and var_config.lower_bound else "-np.inf"
                    high_bound_str = var_config.upper_bound if var_config and var_config.upper_bound else "np.inf"

                    action_key = f"assign_{var_name}"
                    space_part[action_key] = f"spaces.Box(low=self._eval_expr('{low_bound_str}', {{}}), high=self._eval_expr('{high_bound_str}', {{}}), shape=(), dtype=np.float32)"
                    logic_part.append(f"        self.state['{var_name}'] = action['{action_key}']")

                elif isinstance(action_item, DeterministicAssignmentActionConfig):
                    var_name = action_item.variable
                    value_expr = action_item.value
                    logic_part.append(f"        self.state['{var_name}'] = float(self._eval_expr('{value_expr}', self.state))")

                elif isinstance(action_item, ActionConstraintConfig):
                    formula = action_item.formula
                    logic_part.append(f"        if not self._eval_expr('{formula}', self.state):")
                    logic_part.append(f"            self.terminated = True # Invalid action leads to termination")
                    logic_part.append(f"            self.reward -= 100.0 # Penalty for invalid action")
                    logic_part.append(f"            return False # Indicate invalid action")
            return space_part, logic_part
        else:
            raise ValueError(f"Unsupported action node type: {type(node)}")

    # Start processing from the root action structure
    action_space_dict_str, action_mapping_logic = _process_action_node(control_func.action_structure)

    # Convert action_space_dict_str to a string representation for code generation
    action_space_str_parts = []
    for k, v in action_space_dict_str.items():
        action_space_str_parts.append(f"'{k}': {v}")
    action_space_code_gen_str = "{" + ", ".join(action_space_str_parts) + "}"


    action_space_code = []
    action_space_code.append(f"        self.action_space = spaces.Dict({action_space_code_gen_str})")
    action_space_code.append("")
    action_space_code.append("    def _apply_agent_action(self, action):")
    action_space_code.append("        # Apply agent's chosen action to the environment state")
    action_space_code.append("        # Returns True if action was valid, False otherwise (e.g., violated constraint)")
    action_space_code.append("        self.terminated = False # Reset termination flag for action application")
    action_space_code.append("        self.reward = 0.0 # Reset reward for action application")
    action_space_code.extend(action_mapping_logic)
    action_space_code.append("        return not self.terminated # Return True if no termination occurred due to action")


    # --- 4. ODE Integration Logic ---
    ode_integration_code = []
    ode_integration_code.append("    def _ode_system(self, t, y):")
    ode_integration_code.append(f"        state_dict = {{k: v for k, v in zip(self.observation_vars, y)}}")
    ode_integration_code.append(f"        # Add constants to state_dict for evaluation context")
    ode_integration_code.append(f"        state_dict.update({{k: getattr(self, k) for k in self.constants_names}})")
    ode_integration_code.append(f"        # Add functions/predicates to state_dict for evaluation context")
    ode_integration_code.append(f"        state_dict.update(self.kx_functions)")
    ode_integration_code.append(f"        state_dict.update(self.kx_predicates)")
    ode_integration_code.append(f"        return {sympy_evaluator.get_ode_function_body_str(step_func.odes, variables.state_variables)}")
    ode_integration_code.append("")

    # Generate Event function
    ode_integration_code.append("    def _ode_event(self, t, y):")
    ode_integration_code.append(f"        state_dict = {{k: v for k, v in zip(self.observation_vars, y)}}")
    ode_integration_code.append(f"        state_dict.update({{k: getattr(self, k) for k in self.constants_names}})")
    ode_integration_code.append(f"        state_dict.update(self.kx_functions)")
    ode_integration_code.append(f"        state_dict.update(self.kx_predicates)")
    ode_integration_code.append(f"        # Event occurs when evolution domain becomes false (crosses 0 from positive)")
    ode_integration_code.append(f"        return {sympy_evaluator.get_event_function_body_str(step_func.evolution_domain)}")
    ode_integration_code.append("    _ode_event.terminal = True")
    ode_integration_code.append("    _ode_event.direction = -1") # Event when value goes from positive to negative
    ode_integration_code.append("")

    ode_integration_code.append("    def _integrate_dynamics(self, dt: float):")
    ode_integration_code.append(f"        y0 = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)")
    ode_integration_code.append("        t_span = (0, dt)")
    ode_integration_code.append("        sol = solve_ivp(self._ode_system, t_span, y0, events=self._ode_event, dense_output=True, rtol=1e-6, atol=1e-9)")
    ode_integration_code.append("        if sol.status == 0: # Integration successful, no event")
    ode_integration_code.append("            final_y = sol.y[:, -1]")
    ode_integration_code.append("            self.current_time += dt")
    ode_integration_code.append("        elif sol.status == 1: # Event occurred")
    ode_integration_code.append("            # When an event occurs, sol.y_events[0] contains the state at the event time.")
    ode_integration_code.append("            # It's a list of arrays, so we take the last element if it exists.")
    ode_integration_code.append("            final_y = sol.y_events[0][-1] if sol.y_events[0].size > 0 else y0")
    ode_integration_code.append("            # Similarly for time, use sol.t_events[0]")
    ode_integration_code.append("            self.current_time += sol.t_events[0][-1] if sol.t_events[0].size > 0 else 0.0") # Ensure float
    ode_integration_code.append("            self.terminated = True # Environment terminates on evolution domain violation")
    ode_integration_code.append("        else: # Integration failed")
    ode_integration_code.append("            final_y = y0 # Revert to previous state or handle as error")
    ode_integration_code.append("            self.terminated = True # Treat as termination")
    ode_integration_code.append("            print(f\"ODE integration failed: {{sol.message}}\")")
    ode_integration_code.append("")
    ode_integration_code.append("        # Update state with integrated values")
    ode_integration_code.append(f"        for i, var_name in enumerate(self.observation_vars):")
    ode_integration_code.append(f"            self.state[var_name] = final_y[i]")
    ode_integration_code.append("")

    # --- 5. Problem Configuration (reset, step logic) ---
    reset_code = []
    reset_code.append("        # Initialize state variables")
    for var in variables.state_variables + variables.auxiliary_variables:
        # Use _eval_expr to get initial value from bounds or default to 0.0
        initial_value_str = var.lower_bound if var.lower_bound else "0.0"
        reset_code.append(f"        self.state['{var.name}'] = float(self._eval_expr('{initial_value_str}', {{}}))")
    reset_code.append("")
    reset_code.append("        # Apply precondition (e.g., sample initial state satisfying it)")
    precondition_formula = problem.precondition.formula
    reset_code.append(f"        # For now, we assume precondition defines a fixed initial state or is satisfied by default init.")
    reset_code.append(f"        # A more advanced generator would sample until self._eval_expr('{precondition_formula}', self.state) is True.")
    reset_code.append(f"        if not self._eval_expr('{precondition_formula}', self.state):")
    reset_code.append(f"            print(f\"Warning: Initial state does not satisfy precondition: {precondition_formula}\")")
    reset_code.append(f"            # Potentially re-sample or raise error, or terminate immediately")
    reset_code.append(f"            # self.terminated = True # Example: Terminate if precondition not met")
    reset_code.append("")
    reset_code.append("        self.current_time = 0.0")
    reset_code.append("        self.terminated = False")
    reset_code.append("        self.truncated = False")
    reset_code.append("        self.reward = 0.0")
    reset_code.append(f"        observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)")
    reset_code.append("        info = {}")
    reset_code.append("        return observation, info")

    step_postcondition_code = []
    postcondition_formula = problem.postcondition.formula
    # The postcondition can define reward and termination.
    # We'll assume a simple structure where it evaluates to a boolean for termination
    # and a separate reward expression might be present or implied.
    # For now, let's assume the formula itself is the termination condition,
    # and reward is based on reaching a goal or avoiding termination.
    step_postcondition_code.append(f"        # Evaluate postcondition for reward and termination")
    step_postcondition_code.append(f"        post_eval_result = self._eval_expr('{postcondition_formula}', self.state)")
    step_postcondition_code.append(f"        if post_eval_result:")
    step_postcondition_code.append(f"            self.reward += 1.0 # Example: reward for satisfying postcondition")
    step_postcondition_code.append(f"            self.terminated = True # Example: termination upon satisfying postcondition")
    step_postcondition_code.append(f"        else:")
    step_postcondition_code.append(f"            self.reward -= 0.01 # Example: small penalty per step")
    step_postcondition_code.append("")
    step_postcondition_code.append(f"        # Check for truncation (e.g., max steps, max time)")
    # Assuming a default max_episode_steps or max_episode_time
    max_duration_str = step_func.duration if step_func.duration else "10.0" # Default max duration
    # Evaluate max_duration using _eval_expr in the generated environment
    step_postcondition_code.append(f"        if self.current_time >= self._eval_expr('{max_duration_str}', {{}}):")
    step_postcondition_code.append(f"            self.truncated = True")
    step_postcondition_code.append(f"            self.reward -= 10.0 # Penalty for truncation")
    step_postcondition_code.append("")
    step_postcondition_code.append(f"        observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)")
    step_postcondition_code.append("        info = {}")
    step_postcondition_code.append("        return observation, self.reward, self.terminated, self.truncated, info")


    # --- Assemble the full class code ---
    class_template = f"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import sympy
from typing import List, Dict, Any, Optional, Union, Tuple # Re-added for the final generated code

class {env_name}(gym.Env):
    metadata = {{"render_modes": ["human"], "render_fps": 30}}

    def __init__(self):
        super().__init__()
        self.state = {{}} # Internal state dictionary
        self.current_time = 0.0
        self.terminated = False
        self.truncated = False
        self.reward = 0.0

        # --- Generated Definitions ---
{textwrap.indent("\\n".join(definitions_code), ' ' * 8)}
        self.constants_names = [{', '.join([f"'{c.name}'" for c in definitions.constants])}]

        # Store generated functions/predicates as callables
        self.kx_functions = {{}}
        for name in [{', '.join([f"'{f.name}'" for f in definitions.functions])}]:
            self.kx_functions[name] = getattr(self, name)

        self.kx_predicates = {{}}
        for name in [{', '.join([f"'{p.name}'" for p in definitions.predicates])}]:
            self.kx_predicates[name] = getattr(self, name)

        # --- Generated Observation Space ---
{textwrap.indent("\\n".join(observation_space_code), ' ' * 8)}

        # --- Generated Action Space ---
{textwrap.indent("\\n".join(action_space_code), ' ' * 8)}

    def _eval_expr(self, expr_str: str, context: dict) -> Any:
        # Helper to evaluate symbolic expressions using sympy
        # Combine current state, constants, and generated functions/predicates for context
        full_context = {{**context}}
        for k_name in self.constants_names:
            full_context[k_name] = getattr(self, k_name)
        full_context.update(self.kx_functions)
        full_context.update(self.kx_predicates)

        # Create sympy symbols for all variables and constants in the context
        local_symbols = {{k: sympy.symbols(k) for k in full_context}}

        # Replace common operators if necessary
        expr_str = expr_str.replace('^', '**')

        try:
            sympy_expr = sympy.sympify(expr_str, locals=local_symbols)
            # Substitute numerical values
            subs_dict = {{local_symbols[k]: v for k, v in full_context.items() if isinstance(v, (int, float, np.number))}}
            # If the expression is a boolean (e.g., 'x > 0'), evalf() might return True/False
            # If it's a number, it will return the number.
            return sympy_expr.subs(subs_dict).evalf()
        except (sympy.SympifyError, SyntaxError, TypeError) as e:
            print(f"Error evaluating expression '{{expr_str}}' with context '{{full_context}}': {{e}}")
            return False # Default to false for boolean expressions on error, or raise an exception

    # --- Generated ODE System and Event Functions ---
{textwrap.indent("\\n".join(ode_integration_code), ' ' * 4)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
{textwrap.indent("\\n".join(reset_code), ' ' * 8)}

    def step(self, action):
        # Apply agent's action
        action_valid = self._apply_agent_action(action)
        if not action_valid:
            # Action was invalid, environment terminated or heavily penalized
            observation = np.array([self.state[var] for var in self.observation_vars], dtype=np.float32)
            info = {{"action_invalid": True}}
            return observation, self.reward, self.terminated, self.truncated, info

        # Integrate continuous dynamics
        # The duration of one step might be fixed or dynamic based on problem config
        # For now, let's assume a fixed dt for the integration step
        # This 'integration_dt' is the duration of a single 'stepFunc' execution.
        integration_dt = 0.1 # Example fixed integration time step, could be configurable

        # Loop execution from ProblemConfig (e.g., how many stepFuncs per Gym step)
        # For simplicity, assume loop_execution.iterations is '1' or a fixed number
        # Evaluate iterations in the generated environment's context
        num_step_funcs_per_gym_step_str = '{problem.loop_execution.iterations}'
        num_step_funcs_per_gym_step = int(self._eval_expr(num_step_funcs_per_gym_step_str, self.state))
        if num_step_funcs_per_gym_step <= 0: num_step_funcs_per_gym_step = 1 # Default

        for _ in range(num_step_funcs_per_gym_step):
            self._integrate_dynamics(integration_dt)
            if self.terminated: # If ODE integration terminated the environment
                break

        # Evaluate postcondition for reward and termination
{textwrap.indent("\\n".join(step_postcondition_code), ' ' * 8)}

    def render(self):
        # Implement rendering logic here (e.g., matplotlib, pygame)
        pass

    def close(self):
        # Clean up resources
        pass
"""
    return class_template