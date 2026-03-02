import textwrap
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import z3 # <--- ADDED Z3 IMPORT

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

# --- Generic AST Node Serialization Helper ---
def _serialize_ast_node(node: Any, indent_level: int = 0) -> str:
    """
    Recursively serializes an AST node into its Python constructor call string.
    This is used to embed the full AST configuration objects into the generated
    environment's __init__ method for later runtime access (e.g., by Z3 helpers).
    """
    indent_str = "    " * indent_level
    next_indent_str = "    " * (indent_level + 1)

    if isinstance(node, (str, int, float, bool)):
        return repr(node) # Use repr for primitives (handles strings, numbers, bools)

    if node is None:
        return "None"
    
    if isinstance(node, list):
        if not node:
            return "[]"
        items_str = ",\n".join([f"{next_indent_str}{_serialize_ast_node(item, indent_level + 1)}" for item in node])
        return f"[\n{items_str}\n{indent_str}]"

    if isinstance(node, tuple):
        items_str = ", ".join([repr(item) for item in node])
        return f"({items_str})"

    class_name = node.__class__.__name__
    attributes = []
    
    # Manually map attributes for each config type to control formatting and recursion
    if isinstance(node, GymEnvironmentConfig):
        attributes.append(f"env_name={_serialize_ast_node(node.env_name)}")
        attributes.append(f"definitions={_serialize_ast_node(node.definitions, indent_level + 1)}")
        attributes.append(f"variables={_serialize_ast_node(node.variables, indent_level + 1)}")
        attributes.append(f"problem={_serialize_ast_node(node.problem, indent_level + 1)}")
    elif isinstance(node, DefinitionsConfig):
        attributes.append(f"constants={_serialize_ast_node(node.constants, indent_level + 1)}")
        attributes.append(f"functions={_serialize_ast_node(node.functions, indent_level + 1)}")
        attributes.append(f"predicates={_serialize_ast_node(node.predicates, indent_level + 1)}")
        attributes.append(f"hybrid_programs={_serialize_ast_node(node.hybrid_programs, indent_level + 1)}")
    elif isinstance(node, VariablesConfig):
        attributes.append(f"state_variables={_serialize_ast_node(node.state_variables, indent_level + 1)}")
    elif isinstance(node, ProblemConfig):
        attributes.append(f"precondition={_serialize_ast_node(node.precondition, indent_level + 1)}")
        attributes.append(f"loop_execution={_serialize_ast_node(node.loop_execution, indent_level + 1)}")
        attributes.append(f"postcondition={_serialize_ast_node(node.postcondition, indent_level + 1)}")
    elif isinstance(node, ConstantDefinitionConfig):
        value_str = repr(node.value) if node.value is not None else "None"
        attributes.append(f"name={_serialize_ast_node(node.name)}")
        attributes.append(f"type={_serialize_ast_node(node.type)}")
        attributes.append(f"value={value_str}")
    elif isinstance(node, FunctionDefinitionConfig):
        # FIX: Use repr() for body to handle newlines and quotes correctly
        attributes.append(f"name={_serialize_ast_node(node.name)}")
        attributes.append(f"return_type={_serialize_ast_node(node.return_type)}")
        attributes.append(f"parameters={_serialize_ast_node(node.parameters)}")
        attributes.append(f"body={repr(node.body)}") # Changed here
    elif isinstance(node, PredicateDefinitionConfig):
        # FIX: Use repr() for body to handle newlines and quotes correctly
        attributes.append(f"name={_serialize_ast_node(node.name)}")
        attributes.append(f"parameters={_serialize_ast_node(node.parameters)}")
        attributes.append(f"body={repr(node.body)}") # Changed here
    elif isinstance(node, ControlFuncConfig):
        attributes.append(f"name={_serialize_ast_node(node.name)}")
        attributes.append(f"action_space_config={_serialize_ast_node(node.action_space_config, indent_level + 1)}")
    elif isinstance(node, StepFuncConfig):
        attributes.append(f"name={_serialize_ast_node(node.name)}")
        attributes.append(f"odes={_serialize_ast_node(node.odes, indent_level + 1)}")
        attributes.append(f"evolution_domain={_serialize_ast_node(node.evolution_domain, indent_level + 1)}")
        attributes.append(f"duration={_serialize_ast_node(node.duration)}")
    elif isinstance(node, ODEConfig):
        # FIX: Use repr() for equation
        attributes.append(f"variable={_serialize_ast_node(node.variable)}")
        attributes.append(f"equation={repr(node.equation)}") # Changed here
    elif isinstance(node, EvolutionDomainConfig):
        # FIX: Use repr() for constraints
        constraints_str = ", ".join([repr(c) for c in node.constraints]) # Changed here
        attributes.append(f"constraints=[{constraints_str}]")
    elif isinstance(node, VariableDeclarationConfig):
        attributes.append(f"name={_serialize_ast_node(node.name)}")
        attributes.append(f"type={_serialize_ast_node(node.type)}")
    elif isinstance(node, PreconditionConfig):
        # FIX: Use repr() for formula
        attributes.append(f"formula={repr(node.formula)}") # Changed here
    elif isinstance(node, PostconditionConfig):
        # FIX: Use repr() for formula
        attributes.append(f"formula={repr(node.formula)}") # Changed here
    elif isinstance(node, LoopExecutionConfig):
        attributes.append(f"iterations={_serialize_ast_node(node.iterations)}")
        # FIX: Use repr() for loop_invariant_formula
        attributes.append(f"loop_invariant_formula={repr(node.loop_invariant_formula)}") # Changed here
        attributes.append(f"loop_body={_serialize_ast_node(node.loop_body, indent_level + 1)}")
    elif isinstance(node, HybridProgramStepConfig):
        # FIX: Use repr() for invariant_formula
        attributes.append(f"hp_name={_serialize_ast_node(node.hp_name)}")
        attributes.append(f"invariant_formula={repr(node.invariant_formula)}") # Changed here
    elif isinstance(node, ChoiceActionConfig):
        attributes.append(f"level_name={_serialize_ast_node(node.level_name)}")
        attributes.append(f"choices={_serialize_ast_node(node.choices, indent_level + 1)}")
    elif isinstance(node, SequenceActionConfig):
        attributes.append(f"actions={_serialize_ast_node(node.actions, indent_level + 1)}")
    elif isinstance(node, DeterministicAssignmentActionConfig):
        # FIX: Use repr() for value
        attributes.append(f"variable={_serialize_ast_node(node.variable)}")
        attributes.append(f"value={repr(node.value)}") # Changed here
    elif isinstance(node, NondeterministicAssignmentActionConfig):
        attributes.append(f"variable={_serialize_ast_node(node.variable)}")
    elif isinstance(node, ActionConstraintConfig):
        # FIX: Use repr() for formula
        attributes.append(f"formula={repr(node.formula)}") # Changed here
    elif isinstance(node, CallHybridProgramConfig):
        attributes.append(f"hp_name={_serialize_ast_node(node.hp_name)}")
    else:
        raise TypeError(f"Cannot serialize unknown AST node type: {type(node)}")

    if not attributes:
        return f"{class_name}()"

    attributes_str = ",\n".join([f"{next_indent_str}{attr}" for attr in attributes])
    return f"{class_name}(\n{attributes_str}\n{indent_str})"

# --- Helper Functions for Expression Formatting (Generator-side) ---

def _format_kx_expr_for_python_lambda_body_generator(expr_str: str, lambda_param_names: Optional[List[str]] = None, all_constants: List[str] = None, all_state_vars: List[str] = None, all_functions: List[str] = None, all_predicates: List[str] = None) -> str:
    """
    Converts a KeYmaera X expression string into a Python expression string
    suitable for direct use in a lambda body (e.g., for self.functions or self.predicates).
    This function is used by the GENERATOR to produce code.
    It handles identifier prefixing (e.g., 'x' -> 'self.state['x']') and basic arithmetic.
    For relational/logical operators, it uses the same robust parsing logic as the runtime
    _evaluate_expression, but applies it at generation time with prefixed identifiers.
    """
    if lambda_param_names is None: lambda_param_names = []
    if all_constants is None: all_constants = []
    if all_state_vars is None: all_state_vars = []
    if all_functions is None: all_functions = []
    if all_predicates is None: all_predicates = []

    expr_py = expr_str.strip()

    # 1. Basic syntax replacements (power, abs)
    expr_py = expr_py.replace('^', '**').replace('abs(', 'np.abs(')

    # 2. Identify and replace identifiers with their correct scope prefix
    # Build a list of all known identifiers (excluding lambda params, and global modules like 'np', 'math')
    all_env_identifiers_for_prefixing = sorted(
        list(set(all_constants + all_state_vars + all_functions + all_predicates)), # Removed 'np' and 'math'
        key=len, reverse=True
    )

    identifiers_to_replace_for_prefixing = [
        ident for ident in all_env_identifiers_for_prefixing if ident not in lambda_param_names
    ]
    
    if identifiers_to_replace_for_prefixing:
        pattern_str = r'\b(' + '|'.join(re.escape(ident) for ident in identifiers_to_replace_for_prefixing) + r')\b'
        pattern = re.compile(pattern_str)

        def replacer(match):
            ident = match.group(1)
            if ident in all_constants:
                return f"self.constants['{ident}']"
            elif ident in all_state_vars:
                return f"self.state['{ident}']"
            elif ident in all_functions:
                return f"self.functions['{ident}']"
            elif ident in all_predicates:
                return f"self.predicates['{ident}']"
            # No 'np' or 'math' case here, they are assumed to be directly accessible or prefixed in the original string (e.g., np.abs)
            return ident # Should not be reached for identifiers in the list

        expr_py = pattern.sub(replacer, expr_py)

    # 3. Recursive parsing for operator precedence (relational then logical)
    # This logic is similar to the runtime _evaluate_expression, but applies it at generation time.
    def _parse_expression_recursive_generator(sub_expr: str) -> str:
        sub_expr = sub_expr.strip()
        if not sub_expr:
            return ""

        # Handle outermost balanced parentheses
        if sub_expr.startswith('(') and sub_expr.endswith(')'):
            balance = 0
            is_fully_enclosed = True
            for i in range(len(sub_expr) - 1): # Check up to second to last char
                char = sub_expr[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                if balance == 0:
                    is_fully_enclosed = False
                    break
            if is_fully_enclosed:
                return _parse_expression_recursive_generator(sub_expr[1:-1]) # Recursively parse inner part

        # Find lowest precedence operator at top level
        
        # Implication (lowest logical precedence) - right-associative
        balance = 0
        implication_idx = -1
        for i in range(len(sub_expr) - 2, -1, -1): # Scan right-to-left
            char_pair = sub_expr[i:i+2]
            if char_pair == '->' and balance == 0:
                implication_idx = i
                break
            if sub_expr[i] == ')': balance += 1
            elif sub_expr[i] == '(': balance -= 1
        
        if implication_idx != -1:
            left = _parse_expression_recursive_generator(sub_expr[:implication_idx].strip())
            right = _parse_expression_recursive_generator(sub_expr[implication_idx+2:].strip())
            return f"((not ({left})) or ({right}))"

        # Disjunction (next lowest logical) - left-associative
        balance = 0
        disjunction_idx = -1
        for i in range(len(sub_expr)): # Scan left-to-right
            if sub_expr[i] == '|' and balance == 0:
                disjunction_idx = i
                break
            if sub_expr[i] == '(': balance += 1
            elif sub_expr[i] == ')': balance -= 1
        
        if disjunction_idx != -1:
            left = _parse_expression_recursive_generator(sub_expr[:disjunction_idx].strip())
            right = _parse_expression_recursive_generator(sub_expr[disjunction_idx+1:].strip())
            return f"({left}) or ({right})"

        # Conjunction (next lowest logical) - left-associative
        balance = 0
        conjunction_idx = -1
        for i in range(len(sub_expr)): # Scan left-to-right
            if sub_expr[i] == '&' and balance == 0:
                conjunction_idx = i
                break
            if sub_expr[i] == '(': balance += 1
            elif sub_expr[i] == ')': balance -= 1
        
        if conjunction_idx != -1:
            left = _parse_expression_recursive_generator(sub_expr[:conjunction_idx].strip())
            right = _parse_expression_recursive_generator(sub_expr[conjunction_idx+1:].strip())
            return f"({left}) and ({right})"

        # Negation (highest logical precedence) - prefix operator
        if sub_expr.startswith('!'):
            operand = _parse_expression_recursive_generator(sub_expr[1:].strip())
            return f"(not ({operand}))"

        # Relational operators (including '=')
        relational_ops = [('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>'), ('=', 'self._approx_eq')]
        
        balance = 0
        rel_op_found_kx = None
        rel_op_found_py = None
        rel_op_idx = -1
        
        # Scan left-to-right for relational operators
        for i in range(len(sub_expr)):
            char = sub_expr[i]
            if char == '(': balance += 1
            elif char == ')': balance -= 1
            elif balance == 0:
                # Check for multi-character operators first
                for kx_op, py_op_format in relational_ops:
                    if len(kx_op) > 1 and sub_expr[i:i+len(kx_op)] == kx_op:
                        rel_op_found_kx = kx_op
                        rel_op_found_py = py_op_format
                        rel_op_idx = i
                        break
                if rel_op_found_kx: break 

                # Check for single-character operators if no multi-char op found yet
                for kx_op, py_op_format in relational_ops:
                    if len(kx_op) == 1 and sub_expr[i] == kx_op:
                        # Special check for '=', ensure it's not part of '==', '!=', '<=', '>='
                        if kx_op == '=' and (i > 0 and sub_expr[i-1] in '<>=!') or \
                                        (i < len(sub_expr) - 1 and sub_expr[i+1] == '='):
                            continue 
                        rel_op_found_kx = kx_op
                        rel_op_found_py = py_op_format
                        rel_op_idx = i
                        break
                if rel_op_found_kx: break 

        if rel_op_found_kx:
            left = _parse_expression_recursive_generator(sub_expr[:rel_op_idx].strip())
            right = _parse_expression_recursive_generator(sub_expr[rel_op_idx+len(rel_op_found_kx):].strip())
            if rel_op_found_kx == '=':
                return f"{rel_op_found_py}({left}, {right})"
            else:
                return f"({left}) {rel_op_found_py} ({right})"
        
        # If no logical or relational operators, it's an arithmetic expression, literal, or function/predicate call
        return sub_expr

    expr_py = _parse_expression_recursive_generator(expr_py)
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

# --- End Helper Functions (Generator-side) ---


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
        # Check if the value is a simple numeric literal (integer or float) or boolean
        if const.value is not None and (re.fullmatch(r'^-?\d+(\.\d+)?$', const.value) or re.fullmatch(r'^(True|False)$', const.value)):
            formatted_value = const.value
        elif const.value is not None:
             # It's an expression involving other constants/variables, so format it for lambda body
            formatted_value = _format_kx_expr_for_python_lambda_body_generator(
                const.value, [], all_constant_names, state_variable_names, all_function_names, all_predicate_names
            )
        else: # const.value is None (not provided in AST)
            # If value is None, initialize to 0.0. Z3 will solve for it in reset().
            # This prevents hardcoding specific defaults in __init__ for unspecified constants.
            formatted_value = '0.0' # A neutral default, which Z3 will then override.

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
        formatted_body = _format_kx_expr_for_python_lambda_body_generator(
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
        formatted_body = _format_kx_expr_for_python_lambda_body_generator(
            body_single_line, param_names_list, all_constant_names, state_variable_names, all_function_names, all_predicate_names
        )
        predicates_init_code_lines.append(f"self.predicates['{pred.name}'] = lambda {params_str}: {formatted_body}")
    # Pre-indent this block
    indented_predicates_block = textwrap.indent("\n".join(predicates_init_code_lines), '        ')

    # --- Generate Action Space ---
    action_space_definition, control_func_ast_object, action_space_keys = _generate_action_space_and_logic(config.definitions.hybrid_programs)
    
    # Convert the control_func_ast_object to its Python code string representation
    # This will embed raw KeYmaera X strings for action values/formulas
    control_func_ast_code_str = _serialize_ast_node(
        control_func_ast_object, indent_level=3
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
    indented_reset_block = textwrap.indent("\n".join(reset_initial_state_code_lines), '    ') # Adjusted indentation for reset body

    # --- Generate Problem Logic (Postcondition, Loop Invariants) ---
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
    typing_imports = ["Dict", "Any", "Tuple", "List", "Optional", "Union"] # Added Optional, Union
    typing_import_statement = f"from typing import ({', '.join(typing_imports)})"

    # New: Code for _split_top_level_and_local method, to be inserted as a class method
    _split_top_level_and_local_method_code = textwrap.dedent("""
    def _split_top_level_and_local(self, formula_str: str) -> List[str]:
        parts = []
        balance = 0
        start = 0
        # Normalize spaces around '&'
        formula_str = re.sub(r'\\s*&\\s*', '&', formula_str)
        
        for i, char in enumerate(formula_str):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            elif char == '&' and balance == 0:
                part = formula_str[start:i].strip()
                if part:
                    parts.append(part)
                start = i + 1
        
        last_part = formula_str[start:].strip()
        if last_part:
            parts.append(last_part)
            
        return parts if parts else [formula_str.strip()]
    """)

    # Template for the _evaluate_expression method, with robust parsing logic at runtime
    _evaluate_expression_method_template = textwrap.dedent("""
    def _evaluate_expression(self, expr_str: str, local_vars: Optional[Dict[str, Any]] = None, is_constraint: bool = False) -> Any:
        # Safe evaluation of expressions using environment's context.
        # This method parses KeYmaera X expressions at runtime.
        # If is_constraint is True, returns False on evaluation error.
        
        if local_vars is None:
            local_vars = {}
        
        # Strip KeYmaera X constraint prefix '?' if present
        if expr_str.startswith('?'):
            expr_str = expr_str[1:].strip()

        # --- NEW PRE-PROCESSING STEP: Prefix identifiers ---
        # This step converts bare KeYmaera X identifiers (like 'x', 'A', 'stopDist')
        # into Python-compatible references (like 'self.state["x"]', 'self.constants["A"]', 'self.functions["stopDist"]')
        # before the expression is further parsed and evaluated.
        
        processed_expr = expr_str.strip()

        # Combine all known identifiers from the environment that need prefixing
        # Exclude 'np' and 'math' as they are global modules and should not be prefixed.
        all_env_identifiers = list(self.state.keys()) + \\
                              list(self.constants.keys()) + \\
                              list(self.functions.keys()) + \\
                              list(self.predicates.keys())
        
        # Sort by length descending to handle cases where one identifier is a substring of another (e.g., 'dx' before 'x')
        all_env_identifiers_sorted = sorted(all_env_identifiers, key=len, reverse=True)

        # Create a temporary list of identifiers that are *local* to this evaluation (e.g., lambda parameters passed in local_vars)
        # These should NOT be prefixed.
        local_identifiers_in_scope = list(local_vars.keys()) if local_vars else []

        # Perform the prefixing
        for ident_name in all_env_identifiers_sorted:
            if ident_name in local_identifiers_in_scope:
                continue # Skip if it's a local variable (e.g., 'v' in 'stopDist(v)' if 'v' is in local_vars)

            replacement = ""
            if ident_name in self.state:
                replacement = f"self.state['{ident_name}']"
            elif ident_name in self.constants:
                replacement = f"self.constants['{ident_name}']"
            elif ident_name in self.functions:
                replacement = f"self.functions['{ident_name}']"
            elif ident_name in self.predicates:
                replacement = f"self.predicates['{ident_name}']"
            
            if replacement:
                # Use regex with word boundaries to ensure we replace whole identifiers only
                processed_expr = re.sub(r'\\b' + re.escape(ident_name) + r'\\b', replacement, processed_expr)

        # --- END NEW PRE-PROCESSING STEP ---

        expr_py_transformed = processed_expr # Now use the processed_expr

        # 1. Basic syntax replacements (power, abs)
        expr_py_transformed = expr_py_transformed.replace('^', '**').replace('abs(', 'np.abs(')

        # 2. Recursive parsing for operator precedence (relational then logical)
        def _parse_expression_recursive_runtime(sub_expr: str) -> str:
            sub_expr = sub_expr.strip()
            if not sub_expr:
                return ""

            # Handle outermost balanced parentheses
            if sub_expr.startswith('(') and sub_expr.endswith(')'):
                balance = 0
                is_fully_enclosed = True
                for i in range(len(sub_expr) - 1): # Check up to second to last char
                    char = sub_expr[i]
                    if char == '(': balance += 1
                    elif char == ')': balance -= 1
                    if balance == 0:
                        is_fully_enclosed = False
                        break
                if is_fully_enclosed:
                    return _parse_expression_recursive_runtime(sub_expr[1:-1]) # Recursively parse inner part

            # Find lowest precedence operator at top level
            
            # Implication (lowest logical precedence) - right-associative
            balance = 0
            implication_idx = -1
            for i in range(len(sub_expr) - 2, -1, -1): # Scan right-to-left
                char_pair = sub_expr[i:i+2]
                if char_pair == '->' and balance == 0:
                    implication_idx = i
                    break
                if sub_expr[i] == ')': balance += 1
                elif sub_expr[i] == '(': balance -= 1
            
            if implication_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:implication_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[implication_idx+2:].strip())
                return f"((not ({left})) or ({right}))"

            # Disjunction (next lowest logical) - left-associative
            balance = 0
            disjunction_idx = -1
            for i in range(len(sub_expr)): # Scan left-to-right
                if sub_expr[i] == '|' and balance == 0:
                    disjunction_idx = i
                    break
                if sub_expr[i] == '(': balance += 1
                elif sub_expr[i] == ')': balance -= 1
            
            if disjunction_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:disjunction_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[disjunction_idx+1:].strip())
                return f"({left}) or ({right})"

            # Conjunction (next lowest logical) - left-associative
            balance = 0
            conjunction_idx = -1
            for i in range(len(sub_expr)): # Scan left-to-right
                if sub_expr[i] == '&' and balance == 0:
                    conjunction_idx = i
                    break
                if sub_expr[i] == '(': balance += 1
                elif sub_expr[i] == ')': balance -= 1
            
            if conjunction_idx != -1:
                left = _parse_expression_recursive_runtime(sub_expr[:conjunction_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[conjunction_idx+1:].strip())
                return f"({left}) and ({right})"

            # Negation (highest logical precedence) - prefix operator
            if sub_expr.startswith('!'):
                operand = _parse_expression_recursive_runtime(sub_expr[1:].strip())
                return f"(not ({operand}))"

            # Relational operators (including '=')
            relational_ops = [('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>'), ('=', '_approx_eq')]
            
            balance = 0
            rel_op_found_kx = None
            rel_op_found_py = None
            rel_op_idx = -1
            
            # Scan left-to-right for relational operators
            for i in range(len(sub_expr)):
                char = sub_expr[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0:
                    # Check for multi-character operators first
                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) > 1 and sub_expr[i:i+len(kx_op)] == kx_op:
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break 

                    # Check for single-character operators if no multi-char op found yet
                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) == 1 and sub_expr[i] == kx_op:
                            # Special check for '=', ensure it's not part of '==', '!=', '<=', '>='
                            if kx_op == '=' and (i > 0 and sub_expr[i-1] in '<>=!') or \
                                            (i < len(sub_expr) - 1 and sub_expr[i+1] == '='):
                                continue 
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break 

            if rel_op_found_kx:
                left = _parse_expression_recursive_runtime(sub_expr[:rel_op_idx].strip())
                right = _parse_expression_recursive_runtime(sub_expr[rel_op_idx+len(rel_op_found_kx):].strip())
                if rel_op_found_kx == '=':
                    return f"{rel_op_found_py}({left}, {right})"
                else:
                    return f"({left}) {rel_op_found_py} ({right})"
            
            # If no logical or relational operators, it's an arithmetic expression, literal, or function/predicate call
            return sub_expr

        expr_py_final = _parse_expression_recursive_runtime(expr_py_transformed)
        # --- End KeYmaera X to Python conversion logic ---

        try:
            # With explicit prefixing, eval_locals can be much simpler.
            # Only local_vars are truly 'local' to the eval call.
            # All other references are now explicit (self.state['x'], self.constants['A'], etc.)
            eval_globals_for_eval = {'np': np, 'math': math, '_approx_eq': self._approx_eq, 'self': self}
            result = eval(expr_py_final, eval_globals_for_eval, local_vars)
            return result
        except Exception as e:
            print(f"Error evaluating expression '{expr_str}' (transformed: '{expr_py_final}'): {e}")
            if is_constraint:
                return False # For constraints, return False on error
            raise # Re-raise for non-constraint expressions
    """)

    # Embed _kx_expr_to_z3_formula method
    _kx_expr_to_z3_formula_method_code = textwrap.dedent("""
    def _kx_expr_to_z3_formula(self, kx_expr: str, z3_symbol_map: Dict[str, Any], fixed_constants: Dict[str, Union[int, float]]) -> Any:
        \"\"\"
        Converts a KeYmaera X expression string into a Z3 formula.
        This function performs string replacements to convert KeYmaera X syntax
        to a Python string that, when evaluated, constructs a Z3 expression.
        It processes operators respecting precedence: arithmetic, relational, then logical.
        For logical operators, it uses a recursive approach to handle nested parentheses.
        \"\"\"
        expr_transformed = kx_expr.strip()

        # 1. Replace identifiers with their Z3 symbol map reference.
        # Fixed constants will be handled by being added to eval_globals as Z3 literals.
        # Sort by length descending to handle cases where one identifier is a substring of another (e.g., 'dx' before 'x')
        sorted_z3_symbols_and_constants = sorted(
            list(z3_symbol_map.keys()) + list(fixed_constants.keys()), 
            key=len, reverse=True
        )

        for ident_name in sorted_z3_symbols_and_constants:
            replacement = ""
            if ident_name in z3_symbol_map:
                replacement = f"z3_symbol_map['{ident_name}']"
            elif ident_name in fixed_constants:
                # Fixed constants are added to eval_globals directly, so they should be referenced by name
                replacement = ident_name 
            
            if replacement:
                # Use word boundaries to avoid partial replacements (e.g., 'x' in 'dx')
                expr_transformed = re.sub(r'\\b' + re.escape(ident_name) + r'\\b', replacement, expr_transformed)

        # 2. Basic arithmetic and math functions
        expr_transformed = expr_transformed.replace('^', '**')
        expr_transformed = expr_transformed.replace('abs(', 'z3.Abs(') # Use z3.Abs for Z3 context

        # 3. Handle logical and relational operators respecting precedence and parentheses
        # This recursive function will parse the logical and relational structure.
        def _parse_logical_and_relational_expression_z3(expression_str: str) -> str:
            expression_str = expression_str.strip()
            if not expression_str:
                return ""

            # 1. Strip outermost balanced parentheses
            while expression_str.startswith('(') and expression_str.endswith(')'):
                balance = 0
                is_fully_enclosed = True
                for i in range(len(expression_str) - 1): 
                    char = expression_str[i]
                    if char == '(': balance += 1
                    elif char == ')': balance -= 1
                    if balance == 0:
                        is_fully_enclosed = False
                        break
                if is_fully_enclosed:
                    expression_str = expression_str[1:-1].strip()
                else:
                    break 

            # --- Logical Operators (highest level of this parser) ---
            
            # Implication (lowest logical precedence, right-associative)
            balance = 0
            implication_idx = -1
            for i in range(len(expression_str) - 2, -1, -1): # Scan right-to-left for implication
                char = expression_str[i]
                if char == ')': balance += 1
                elif char == '(': balance -= 1
                elif balance == 0 and expression_str[i:i+2] == '->':
                    implication_idx = i
                    break
            if implication_idx != -1:
                left = _parse_logical_and_relational_expression_z3(expression_str[:implication_idx].strip())
                right = _parse_logical_and_relational_expression_z3(expression_str[implication_idx+2:].strip())
                return f"z3.Implies({left}, {right})"

            # Disjunction (medium precedence, left-associative)
            balance = 0
            disjunction_idx = -1
            for i in range(len(expression_str)): # Scan left-to-right for disjunction
                char = expression_str[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0 and char == '|':
                    disjunction_idx = i
                    break 
            if disjunction_idx != -1:
                left = _parse_logical_and_relational_expression_z3(expression_str[:disjunction_idx].strip())
                right = _parse_logical_and_relational_expression_z3(expression_str[disjunction_idx+1:].strip())
                return f"z3.Or({left}, {right})"

            # Conjunction (highest precedence, left-associative)
            balance = 0
            conjunction_idx = -1
            for i in range(len(expression_str)): # Scan left-to-right for conjunction
                char = expression_str[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0 and char == '&':
                    conjunction_idx = i
                    break 
            if conjunction_idx != -1:
                left = _parse_logical_and_relational_expression_z3(expression_str[:conjunction_idx].strip())
                right = _parse_logical_and_relational_expression_z3(expression_str[conjunction_idx+1:].strip())
                return f"z3.And({left}, {right})"
                
            # Logical NOT (prefix operator)
            if expression_str.startswith('!'):
                operand = _parse_logical_and_relational_expression_z3(expression_str[1:].strip())
                return f"z3.Not({operand})"

            # --- Relational Operators (next level down) ---
            # Note: '=' in KeYmaera X maps to '==' for Z3 exact equality
            relational_ops = [('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('<', '<'), ('=', '==')]
            
            balance = 0
            rel_op_found_kx = None
            rel_op_found_py = None
            rel_op_idx = -1
            
            # Scan left-to-right for relational operators
            for i in range(len(expression_str)):
                char = expression_str[i]
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                elif balance == 0:
                    # Check for multi-character operators first
                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) > 1 and expression_str[i:i+len(kx_op)] == kx_op:
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break 

                    # Check for single-character operators if no multi-char op found yet
                    for kx_op, py_op_format in relational_ops:
                        if len(kx_op) == 1 and expression_str[i] == kx_op:
                            # Special check for '=', ensure it's not part of '==', '!=', '<=', '>='
                            if kx_op == '=' and ((i > 0 and expression_str[i-1] in '<>=!') or \
                                            (i < len(expression_str) - 1 and expression_str[i+1] == '=')):
                                continue 
                            rel_op_found_kx = kx_op
                            rel_op_found_py = py_op_format
                            rel_op_idx = i
                            break
                    if rel_op_found_kx: break 

            if rel_op_found_kx:
                left = _parse_logical_and_relational_expression_z3(expression_str[:rel_op_idx].strip())
                right = _parse_logical_and_relational_expression_z3(expression_str[rel_op_idx+len(rel_op_found_kx):].strip())
                return f"({left}) {rel_op_found_py} ({right})" 

            # Base case: no logical or relational operators found, return the expression as is.
            # This should now be an arithmetic expression or a literal.
            return expression_str

        # Apply the combined logical and relational expression parser
        expr_transformed = _parse_logical_and_relational_expression_z3(expr_transformed)

        # Final evaluation in a Z3-aware context
        eval_globals = {'z3': z3, 'z3_symbol_map': z3_symbol_map}
        for const_name, const_value in fixed_constants.items():
            if isinstance(const_value, int):
                eval_globals[const_name] = z3.IntVal(const_value)
            else: # float or other numeric
                eval_globals[const_name] = z3.RealVal(const_value)
        
        try:
            return eval(expr_transformed, eval_globals)
        except Exception as e:
            raise ValueError(f"Failed to evaluate Z3 expression '{kx_expr}' (transformed: '{expr_transformed}'): {e}")
    """)

    # Embed _solve_initial_state_with_z3 method
    _solve_initial_state_with_z3_method_code = textwrap.dedent("""
    def _solve_initial_state_with_z3(self, precondition_kx: str) -> Optional[Dict[str, float]]:
        \"\"\"
        Uses Z3 to find a satisfying assignment for initial state variables and constants
        based on the expanded KeYmaera X precondition.
        Returns a dictionary of solved values (variable/constant name -> float) or None if unsatisfiable.
        \"\"\"
        # 1. Prepare Z3 symbols (Real for all variables and constants that need to be solved)
        z3_symbol_map = {}
        
        # Store fixed constants separately to avoid creating Z3 variables for them
        fixed_constants = {}

        # State variables (these are the ones we want Z3 to solve for)
        for var_decl in self._variables_config.state_variables:
            z3_symbol_map[var_decl.name] = z3.Real(var_decl.name)
        
        # Constants: If they have a fixed numeric value, use that value directly in the expression.
        # Otherwise, treat them as Z3 Real variables to be solved for.
        for const_decl in self._definitions_config.constants:
            if const_decl.value is not None and re.fullmatch(r'^-?\\d+(\\.\\d+)?$', const_decl.value):
                # This constant has a fixed numeric value. Store it as a Python float.
                fixed_constants[const_decl.name] = float(const_decl.value)
            else:
                # This constant either has no value, or its value is an expression.
                # For Z3, we treat it as a variable that needs to be solved.
                z3_symbol_map[const_decl.name] = z3.Real(const_decl.name)

        # 2. Expand predicates in the precondition
        expanded_precondition = self._expand_formula(precondition_kx, self._definitions_config)
        # print(f"Expanded precondition for Z3 solving: '{expanded_precondition}'")

        # 3. Convert expanded KeYmaera X formula to Z3 expression
        try:
            # Pass both z3_symbol_map and fixed_constants
            z3_formula = self._kx_expr_to_z3_formula(expanded_precondition, z3_symbol_map, fixed_constants)
        except ValueError as e:
            print(f"Z3 Solver Error: Failed to convert precondition to Z3 formula: {e}")
            return None
        
        # print(f"Z3 formula for precondition: {z3_formula}")

        # 4. Create Z3 solver and add the formula
        s = z3.Solver()
        s.add(z3_formula)

        # 5. Check for satisfiability
        check_result = s.check()
        if check_result == z3.sat:
            model = s.model()
            solved_values = {}
            # Add values for state variables and constants that were Z3 variables
            for k, v in z3_symbol_map.items():
                evaluated_val = model.evaluate(v, model_completion=True)
                if evaluated_val is not None:
                    if hasattr(evaluated_val, 'as_fraction'): # For Z3 Real values
                        solved_values[k] = float(evaluated_val.as_fraction())
                    elif hasattr(evaluated_val, 'as_long'): # For Z3 Int values
                        solved_values[k] = float(evaluated_val.as_long())
                    else: # Fallback for other Z3 types or if as_fraction/as_long not present
                        solved_values[k] = float(str(evaluated_val)) # Convert to string and then float
            # Add values for fixed constants
            # These are already numeric, no need for Z3 evaluation
            solved_values.update(fixed_constants)
            return solved_values
        elif check_result == z3.unsat:
            print(f"Z3 Solver: Precondition is UNSAT. No initial state exists for: '{precondition_kx}'")
            return None
        else: # z3.unknown
            print(f"Z3 Solver: Precondition satisfiability UNKNOWN for: '{precondition_kx}'")
            return None
    """)

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
        self._definitions_config = {_serialize_ast_node(config.definitions, indent_level=2)}
        self._variables_config = {_serialize_ast_node(config.variables, indent_level=2)}
        self._problem_config = {_serialize_ast_node(config.problem, indent_level=2)}

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

{textwrap.indent(_kx_expr_to_z3_formula_method_code, '    ')}

{textwrap.indent(_solve_initial_state_with_z3_method_code, '    ')}

{textwrap.indent(_evaluate_expression_method_template, '    ')}

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
            # Formula is a raw KeYmaera X string, parsed at runtime by _evaluate_expression
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

{textwrap.indent(_split_top_level_and_local_method_code, '    ')}

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
                    # Note: ODE events themselves use floating-point comparisons internally.
                    # This check here is a re-evaluation after the fact.
                    if self.state.get('v', 0.0) < -self.equality_tolerance: # Use tolerance for v < 0 check
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
        # Postcondition is a raw KeYmaera X string, parsed at runtime by _evaluate_expression
        postcondition_satisfied = self._evaluate_expression(f'{postcondition_formula_py}', is_constraint=True)
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
    # We pass all_constants, all_function_names, all_predicate_names to _format_kx_expr_for_python_lambda_body_generator
    # so it can correctly prefix them with `self.constants['...']`, `self.functions['...']`, etc.
    # The state variables (x, y, v, etc.) are directly in the local scope of ode_func, so they are not prefixed.
    ode_params_in_scope = list(ode_state_map.keys()) + ['t_val'] # These are direct local variables

    for ode in step_func_config.odes:
        idx = ode_state_map.get(ode.variable)
        if idx is not None:
            # Ensure ODE equation is a single line
            equation_single_line = ode.equation.replace('\n', ' ').strip()
            # Format the ODE equation for direct Python evaluation within the ODE function's scope
            formatted_equation = _format_kx_expr_for_python_lambda_body_generator(
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
    reset_code_lines.append(f"    z3_base_solution = self._solve_initial_state_with_z3(self._problem_config.precondition.formula)")
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
    reset_code_lines.append(f"                if var_name in self._state_keys:") # Only perturb state variables
    reset_code_lines.append(f"                    perturb = self.np_random.uniform(-perturbation_magnitude, perturbation_magnitude)")
    reset_code_lines.append(f"                    trial_state[var_name] = value + perturb") # Apply perturbation to base value
    reset_code_lines.append(f"            ")
    
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
    
    # Escape the precondition formula for embedding into the f-string
    precondition_formula_escaped_for_eval = precondition.formula.replace("'", "\\'").replace('\\', '\\\\').replace('\n', ' ')
    
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
    expanded_precondition_kx = _expand_formula(precondition.formula, definitions)
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
    if re.search(r'dx\^2\s*\+\s*dy\^2', expanded_precondition_kx):
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
    if re.search(r'\(x\s*-\s*xo\)\^2\s*\+\s*\(y\s*-\s*yo\)\^2\s*>\s*0', expanded_precondition_kx):
        reset_code_lines.append(f"        if not z3_base_solution:") # Only run if Z3 failed
        reset_code_lines.append(f"            # Ensure robot and obstacle are not at the same position (fallback)")
        reset_code_lines.append(f"            if self._approx_eq(self.state['x'], self.state['xo']) and self._approx_eq(self.state['y'], self.state['yo']):")
        reset_code_lines.append(f"                self.state['x'] += 0.1 # Slightly perturb x to ensure separation")
        reset_code_lines.append("")

    # 5. Final verification (always runs)
    # The _evaluate_expression method expects KeYmaera X syntax (with '=' for equality).
    # It will internally convert '=' to self._approx_eq.
    final_precondition_for_eval = _expand_formula(precondition.formula, definitions)
    # Remove extra spaces for cleaner embedding
    normalized_string = " ".join(final_precondition_for_eval.split())
    
    print(f"Debug: Final expanded precondition for evaluation: '{final_precondition_for_eval}'") # Debug print to check the final precondition formula after expansion
    
    # Escape strings for embedding into the generated Python code's f-string
    original_precondition_formula_escaped = precondition.formula.replace("'", "\\'").replace('\\', '\\\\').replace('\n', ' ')
    final_precondition_for_eval_escaped = normalized_string.replace("'", "\\'").replace('\\', '\\\\')
    
    print(f"Debug: Escaped original precondition formula for embedding: '{original_precondition_formula_escaped}'") # Debug print to check escaping
    print(f"Debug: Escaped final precondition for evaluation: '{final_precondition_for_eval_escaped}'") # Debug print to check escaping

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


# --- Main Execution (for testing the generator) ---
if __name__ == "__main__":
    # This part is for testing the generator directly with the provided PyAST structure
    
    # Create a dummy GymEnvironmentConfig based on the example PyAST structure
    # This mimics the output of convert_keymaerax_ast_to_intermediate_pyast
    
    # Constants
    # Set ep_const.value to None to demonstrate Z3 solving for it
    ep_const = ConstantDefinitionConfig(name='ep', type='Real', value=None) 
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
    # Note: 'isWellformedDir' and 'initialState' use '=', which will now be approx.
    # Changed isWellformedDir to use equality for testing the fix
    isWellformedDir_pred = PredicateDefinitionConfig(name='isWellformedDir', parameters=[('Real', 'dx'), ('Real', 'dy')], body='dx^2 + dy^2 = 1')
    # FIX: Corrected bounds predicate to match 0 <= Ud < 1
    bounds_pred = PredicateDefinitionConfig(name='bounds', parameters=[], body='A >= 0\n    & b > 0\n    & ep > 0\n    & V >= 0\n    & 0<=Ud\n    & Ud < 1') # Use & for AND, added V, Ud bounds
    initialState_pred = PredicateDefinitionConfig(name='initialState', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'dx'), ('Real', 'dy'), ('Real', 'xo'), ('Real', 'yo')], body='v = 0 & (x-xo)^2 + (y-yo)^2 > 0 & isWellformedDir(dx,dy)') # Use &
    loopinv_pred = PredicateDefinitionConfig(name='loopinv', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'dx'), ('Real', 'dy'), ('Real', 'xo'), ('Real', 'yo')], body='v >= 0 & isWellformedDir(dx,dy) & (abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v))')
    assumptions_pred = PredicateDefinitionConfig(name='assumptions', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'dx'), ('Real', 'dy'), ('Real', 'xo'), ('Real', 'yo')], body='bounds() & initialState(x,y,v,dx,dy,xo,yo)')


    # controlFunc Action Config
    # This structure directly reflects the problem description's controlFunc
    # Choice 0: a := -b
    # Choice 1: v = 0; a := 0; w := 0
    # Choice 2: a := A; w := *; -W<=w & w<=W; r := *; xo := *; yo := *; r!=0 & r*w = v; abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparable(v)
    choice0_actions_branch0 = SequenceActionConfig(actions=[
        DeterministicAssignmentActionConfig(variable='a', value='-b')
    ])
    choice0_actions_branch1 = SequenceActionConfig(actions=[
        ActionConstraintConfig(formula='v = 0'), # This '=' will now be approximate
        DeterministicAssignmentActionConfig(variable='a', value='0'),
        DeterministicAssignmentActionConfig(variable='w', value='0')
    ])
    choice0_actions_branch2 = SequenceActionConfig(actions=[
        DeterministicAssignmentActionConfig(variable='a', value='A'),
        NondeterministicAssignmentActionConfig(variable='w'),
        ActionConstraintConfig(formula='-W<=w & w<=W'), # Use &
        NondeterministicAssignmentActionConfig(variable='r'),
        NondeterministicAssignmentActionConfig(variable='xo'),
        NondeterministicAssignmentActionConfig(variable='yo'), # FIX: Corrected class name here
        ActionConstraintConfig(formula='r!=0 & r*w = v'), # This '=' will now be approximate
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
        constants=[ep_const, b_const, A_const, W_const, V_const, Ud_const], # Added V, Ud_const
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
        HybridProgramStepConfig(hp_name='stepFunc', invariant_formula='t>=0 & isWellformedDir(dx, dy) & (v>0 -> (abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v)))') # Corrected invariant for example
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
    sanitized_example_env_name = re.sub(r'_+', '_', sanitized_example_env_name).strip('_')
    if sanitized_example_env_name and sanitized_example_env_name[0].isdigit():
        sanitized_example_env_name = '_' + sanitized_example_env_name
    if not sanitized_example_env_name:
        sanitized_example_env_name = "GeneratedEnv"

    print(f"""
To use the generated environment:

1.  Make sure `gymnasium`, `numpy`, `scipy`, and `z3-solver` are installed (`pip install gymnasium numpy scipy z3-solver`).
2.  **CRITICAL:** Save the generated code above as `{output_filename}`. **Ensure you completely replace your existing generator file with this one.**
3.  Ensure `pyast_config.py` is in the same directory.
4.  You can then import and use the generated environment in another Python script or interactive session:

```python
import gymnasium as gym
from generated_kx_env import {sanitized_example_env_name} 
import numpy as np

# Instantiate the environment
# The __init__ method now calls reset(), so the environment starts in a valid state.
env = {sanitized_example_env_name}()

print("\\n--- Initial state after __init__ (from first reset call) ---")
# The first reset call is done internally by __init__
observation, info = env._get_obs(), env._get_info() # Manually get obs and info as reset() was called internally
print("Initial observation:", observation)
print("Initial state (info):", info)
if env.last_failing_precondition_formula:
    print(f"Precondition not fully satisfied: {{env.last_failing_precondition_formula}}")
print(f"Constants after first reset: {{env.constants}}") # Check if ep was solved by Z3

print("\\n--- Second Reset (should be different and valid) ---")
observation, info = env.reset() # Explicitly call reset again for a new episode
print("Initial observation:", observation)
print("Initial state (info):", info)
if env.last_failing_precondition_formula:
    print(f"Precondition not fully satisfied: {{env.last_failing_precondition_formula}}")
print(f"Constants after second reset: {{env.constants}}") # Check if ep was solved by Z3

# Example 1: Choose the first branch of ChoiceLevel_0 (a := -b)
print("\\nTaking 5 steps with action_branch_0 (a := -b):")
env.reset() # Reset for a fresh start for this example
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