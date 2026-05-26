# embedded_methods.py
# This module contains the embedded method templates that will be injected into the generated Gymnasium environment class
import textwrap

# --- Embedded Method Templates ---
# These are multi-line strings that will be injected into the generated Gymnasium environment class.

SPLIT_TOP_LEVEL_AND_LOCAL_METHOD_CODE = textwrap.dedent("""
    def _split_top_level_and_local(self, formula_str: str) -> List[str]:
        \"\"\"
        Splits a formula string by top-level '&' operators, respecting brace nesting.
        This is a simplified version of the parser's split_by_operator_outside_braces
        tailored for logical conjunctions.
        \"\"\"
        parts = []
        brace_level = 0
        current_part_start = 0
        
        # Normalize spaces around '&' for consistent splitting
        formula_str = re.sub(r'\\s*&\\s*', '&', formula_str)

        i = 0
        while i < len(formula_str):
            if formula_str[i] == '(':
                brace_level += 1
            elif formula_str[i] == ')':
                brace_level -= 1

            if formula_str[i] == '&' and brace_level == 0:
                part = formula_str[current_part_start:i].strip()
                if part:
                    parts.append(part)
                current_part_start = i + 1
            i += 1

        # Add the last part after the loop finishes
        last_part = formula_str[current_part_start:].strip()
        if last_part:
            parts.append(last_part)
        return parts if parts else [formula_str.strip()]
    """)


EVALUATE_EXPRESSION_METHOD_TEMPLATE = textwrap.dedent("""
    def _evaluate_expression(self, expr_str: str, local_vars: Optional[Dict[str, Any]] = None, is_constraint: bool = False) -> Any:
        \"\"\"
        Safe evaluation of expressions using environment's context.
        This method parses KeYmaera X expressions at runtime.
        If is_constraint is True, returns False on evaluation error.
        \"\"\"
        
        if local_vars is None: # FIX: Changed '===' to 'is'
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
                            if kx_op == '=' and ((i > 0 and sub_expr[i-1] in '<>=!') or \\
                                            (i < len(sub_expr) - 1 and sub_expr[i+1] == '=')):
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
            eval_globals_for_eval = {{'np': np, 'math': math, '_approx_eq': self._approx_eq, 'self': self}}
            result = eval(expr_py_final, eval_globals_for_eval, local_vars)
            return result
        except Exception as e:
            print(f"Error evaluating expression '{{expr_str}}' (transformed: '{{expr_py_final}}'): {{e}}")
            if is_constraint:
                return False # For constraints, return False on error
            raise # Re-raise for non-constraint expressions
    """)


KX_EXPR_TO_Z3_FORMULA_METHOD_CODE = textwrap.dedent("""
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
            relational_ops = [('!=', '!='), ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>'), ('=', '==')]
            
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
                            if kx_op == '=' and ((i > 0 and expression_str[i-1] in '<>=!') or \\
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
                if rel_op_found_kx == '=':
                    return f"{left} {py_op_format} {right}" # Use py_op_format directly for Z3 '=='
                else:
                    return f"({left}) {py_op_format} ({right})" 

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
            raise ValueError(f"Failed to evaluate Z3 expression '{{kx_expr}}' (transformed: '{{expr_transformed}}'): {{e}}")
    """)


SOLVE_INITIAL_STATE_WITH_Z3_METHOD_CODE = textwrap.dedent("""
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
                # This constant either has no value, or its value is an an expression.
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
            print(f"Z3 Solver Error: Failed to convert precondition to Z3 formula: {{e}}")
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
            print(f"Z3 Solver: Precondition is UNSAT. No initial state exists for: '{{precondition_kx}}'")
            return None
        else: # z3.unknown
            print(f"Z3 Solver: Precondition satisfiability UNKNOWN for: '{{precondition_kx}}'")
            return None
    """)