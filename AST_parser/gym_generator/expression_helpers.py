import re
from typing import List, Optional, Tuple, Union, Dict, Any

# Assuming pyast_config.py is in the same directory or importable
from pyast_config import DefinitionsConfig, PredicateDefinitionConfig

def format_kx_expr_for_python_lambda_body_generator(expr_str: str, lambda_param_names: Optional[List[str]] = None, all_constants: List[str] = None, all_state_vars: List[str] = None, all_functions: List[str] = None, all_predicates: List[str] = None) -> str:
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
                        if kx_op == '=' and ( (i > 0 and sub_expr[i-1] in '<>=!') or \
                                        (i < len(sub_expr) - 1 and sub_expr[i+1] == '=') ):
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

def expand_formula(formula_str: str, definitions: DefinitionsConfig) -> str:
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