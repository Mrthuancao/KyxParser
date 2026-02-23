Okay, let's break down the code snippet by snippet, focusing on the Z3 integration and how it enhances the generation of Gymnasium environments.

### **1. Imports and Setup**

```python
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
```

**Explanation:**
*   **`import z3`**: This is the most significant addition. It imports the Z3 Python API, which allows us to interact with the Z3 SMT (Satisfiability Modulo Theories) solver. Z3 is used here to find an initial state for the environment that satisfies the complex logical conditions specified in the KeYmaera X precondition.
*   **`textwrap`, `typing`, `re`**: These are standard Python libraries used for text formatting (indentation), type hinting, and regular expressions, respectively. They were already present and their roles haven't changed.
*   **`from pyast_config import (...)`**: This imports the Abstract Syntax Tree (AST) node classes that define the structure of the KeYmaera X environment configuration. These classes are the input to our generator.

---

### **2. Helper Functions for Expression Formatting (`_format_kx_expr_for_python_lambda_body`, `_format_kx_expr_for_python_eval_context`, `_expand_formula`)**

These functions remain largely unchanged from the previous version.

```python
# --- Helper Functions for Expression Formatting ---

def _format_kx_expr_for_python_lambda_body(expr_str: str, lambda_param_names: List[str], all_constants: List[str], all_state_vars: List[str], all_functions: List[str], all_predicates: List[str]) -> str:
    # ... (unchanged logic) ...
    return expr_py

def _format_kx_expr_for_python_eval_context(expr_str: str) -> str:
    # ... (unchanged logic) ...
    return expr_py

def _expand_formula(formula_str: str, definitions: DefinitionsConfig) -> str:
    # ... (unchanged logic) ...
    return expanded_formula
```

**Explanation:**
*   **`_format_kx_expr_for_python_lambda_body`**: This function takes a KeYmaera X expression string and converts it into a Python-compatible string suitable for use within a Python `lambda` function body. It handles operator conversions (e.g., `^` to `**`, `&` to `and`, `|` to `or`, `=` to `==`, `->` to `(not A or B)`) and prefixes identifiers with `self.constants['...']` or `self.state['...']` so they can be correctly evaluated within the generated environment's methods.
*   **`_format_kx_expr_for_python_eval_context`**: A simpler version of the above, used for expressions that will be evaluated at runtime by the environment's `_evaluate_expression` method. It performs basic operator conversions but doesn't prefix identifiers, as `_evaluate_expression` handles the scope resolution dynamically.
*   **`_expand_formula`**: This function recursively expands predicate calls within a KeYmaera X formula. For instance, if you have `assumptions(x,y)` and `assumptions` is defined as `x > 0 & y > 0`, it will replace `assumptions(x,y)` with `(x > 0 & y > 0)`. This is crucial because Z3 and Python evaluation need the fully expanded logical expression, not just the predicate call.

---

### **3. Z3 Integration Helpers (`_kx_expr_to_z3_formula`, `_solve_initial_state_with_z3`)**

These are entirely new functions responsible for the Z3 integration.

#### **`_kx_expr_to_z3_formula(kx_expr: str, z3_symbol_map: Dict[str, Any]) -> Any`**

```python
# --- Z3 Integration Helpers ---
def _kx_expr_to_z3_formula(kx_expr: str, z3_symbol_map: Dict[str, Any]) -> Any:
    """
    Converts a KeYmaera X expression string into a Z3 formula.
    This function performs string replacements to convert KeYmaera X syntax
    to a Python string that, when evaluated, constructs a Z3 expression.
    It assumes logical operations are parenthesized binary or unary operations
    (e.g., (A & B), (A | B), (!A), (A -> B)) for robust parsing of logical structure.
    Arithmetic and relational operators are handled by Python's eval
    on Z3-overloaded types.
    """
    expr_transformed = kx_expr.strip()

    # 1. Replace identifiers with their Z3 symbol map reference
    # Sort keys by length descending to handle 'dx' before 'x'
    sorted_keys = sorted(z3_symbol_map.keys(), key=len, reverse=True)
    for key in sorted_keys:
        # Use word boundaries to ensure whole word replacement
        expr_transformed = re.sub(r'\b' + re.escape(key) + r'\b', f"z3_symbol_map['{key}']", expr_transformed)

    # 2. Handle KeYmaera X implication '->' (A -> B becomes z3.Implies(A, B))
    # This must be done iteratively to resolve nested implications.
    while True:
        # Find innermost (A -> B)
        new_expr = re.sub(r'\(([^()]*?)\s*->\s*([^()]*?)\)', r'z3.Implies(\1, \2)', expr_transformed)
        if new_expr == expr_transformed:
            break
        expr_transformed = new_expr

    # 3. Handle logical AND '&' and OR '|'
    # These also need iterative replacement for nesting.
    while True:
        original_expr_transformed = expr_transformed
        
        # Logical AND: (A & B) => z3.And(A, B)
        expr_transformed = re.sub(r'\(([^()]*?)\s*&\s*([^()]*?)\)', r'z3.And(\1, \2)', expr_transformed)
        
        # Logical OR: (A | B) => z3.Or(\1, \2)', expr_transformed)
        expr_transformed = re.sub(r'\(([^()]*?)\s*\|\s*([^()]*?)\)', r'z3.Or(\1, \2)', expr_transformed)

        if expr_transformed == original_expr_transformed:
            break

    # 4. Handle logical NOT '!'
    # This regex attempts to match `! ` followed by an identifier or a parenthesized expression.
    # It avoids matching `!=`.
    # `\b\w+\b` for identifiers, `\([^()]*?\)` for simple parenthesized expressions.
    # This is a heuristic and might need adjustment for very complex cases.
    expr_transformed = re.sub(r'!\s*(\b\w+\b|\([^()]*?\))', r'z3.Not(\1)', expr_transformed)
    
    # 5. Basic arithmetic and math functions
    expr_transformed = expr_transformed.replace('^', '**')
    expr_transformed = expr_transformed.replace('abs(', 'z3.Abs(') # Use z3.Abs for Z3 context

    # 6. Equality operator '=' (KeYmaera X uses '=' for equality, Python uses '==')
    expr_transformed = re.sub(r'(?<![<>=!])=(?![=])', '==', expr_transformed)

    # Remove any remaining outermost parentheses if they were just for grouping the whole expression
    # This is a heuristic to clean up the final expression string for eval.
    if expr_transformed.startswith('(') and expr_transformed.endswith(')'):
        # A simple check to avoid removing essential parentheses, e.g., in `(A & B) | C`
        # This is not a full parser, so it's a best effort.
        if expr_transformed.count('(') == expr_transformed.count(')'):
            expr_transformed = expr_transformed[1:-1].strip()

    # Final evaluation in a Z3-aware context
    eval_globals = {'z3': z3, 'z3_symbol_map': z3_symbol_map}
    
    try:
        return eval(expr_transformed, eval_globals)
    except Exception as e:
        raise ValueError(f"Failed to evaluate Z3 expression '{kx_expr}' (transformed: '{expr_transformed}'): {e}")
```

**Explanation:**
This function is the core of translating KeYmaera X's logical syntax into Z3's Python API.
*   **`expr_transformed = kx_expr.strip()`**: Initializes a working copy of the expression.
*   **1. Replace identifiers with Z3 symbols**: It iterates through `z3_symbol_map` (which contains Z3 `Real` variables for each state variable and constant) and replaces every occurrence of a variable/constant name in the expression with its Z3 symbol reference (e.g., `x` becomes `z3_symbol_map['x']`). This is crucial for Z3 to recognize these as variables it can reason about. Sorting keys by length (`dx` before `x`) prevents partial matches.
*   **2. Handle KeYmaera X implication `->`**: KeYmaera X uses `A -> B` for implication. Z3's Python API uses `z3.Implies(A, B)`. This section uses a regex to find `(A -> B)` patterns and replaces them. It runs in a `while` loop to handle nested implications (e.g., `(A -> (B -> C))`).
*   **3. Handle logical AND `&` and OR `|`**: Similar to implication, KeYmaera X uses `&` and `|`, while Z3 uses `z3.And()` and `z3.Or()`. This section converts these operators, also iteratively to handle nesting.
*   **4. Handle logical NOT `!`**: KeYmaera X uses `!A`, Z3 uses `z3.Not(A)`. A regex replaces `! ` followed by a variable or parenthesized expression. It specifically avoids `!=` (not equals).
*   **5. Basic arithmetic and math functions**:
    *   `^` (power) is replaced with `**` (Python's power operator). Z3's `Real` type overloads `**` to work correctly.
    *   `abs()` is replaced with `z3.Abs()`, which is Z3's absolute value function.
*   **6. Equality operator `=`**: KeYmaera X often uses `=` for equality in logical formulas. Python uses `==`. This regex replaces single `=` signs with `==` but carefully avoids `!=`, `<=`, `>=`.
*   **Remove outermost parentheses**: A heuristic to clean up the string if the entire expression is wrapped in unnecessary parentheses after all transformations.
*   **`eval(expr_transformed, eval_globals)`**: This is the critical step. After all string manipulations, `expr_transformed` becomes a Python string that, when evaluated, constructs a Z3 expression. `eval_globals` provides the `z3` module and the `z3_symbol_map` so `eval` can correctly interpret `z3.Implies`, `z3.And`, `z3.Not`, `z3.Abs`, and the references like `z3_symbol_map['x']`. This dynamically builds the Z3 formula.

#### **`_solve_initial_state_with_z3(...) -> Optional[Dict[str, float]]`**

```python
def _solve_initial_state_with_z3(
    precondition_kx: str,
    definitions: DefinitionsConfig,
    variables: List[VariableDeclarationConfig]
) -> Optional[Dict[str, float]]:
    """
    Uses Z3 to find a satisfying assignment for initial state variables and constants
    based on the expanded KeYmaera X precondition.
    Returns a dictionary of solved values (variable/constant name -> float) or None if unsatisfiable.
    """
    # 1. Prepare Z3 symbols (Real for all variables and constants)
    z3_symbol_map = {}
    
    # State variables (these are the ones we want Z3 to solve for)
    for var_decl in variables:
        z3_symbol_map[var_decl.name] = z3.Real(var_decl.name)
    
    # Constants (include them in the map, Z3 might assign values if they are unconstrained
    # but appear in the formula, or use their fixed value if provided and consistent)
    for const_decl in definitions.constants:
        # If a constant has a specific value, add it as a Z3 constant
        if const_decl.value is not None and re.fullmatch(r'^-?\d+(\.\d+)?$', const_decl.value):
            z3_symbol_map[const_decl.name] = float(const_decl.value) # Use Python float directly
        else:
            # Otherwise, treat it as a Z3 Real variable that can be solved for
            z3_symbol_map[const_decl.name] = z3.Real(const_decl.name)

    # 2. Expand predicates in the precondition
    expanded_precondition = _expand_formula(precondition_kx, definitions)

    # 3. Convert expanded KeYmaera X formula to Z3 expression
    try:
        z3_formula = _kx_expr_to_z3_formula(expanded_precondition, z3_symbol_map)
    except ValueError as e:
        print(f"Z3 Solver Error: Failed to convert precondition to Z3 formula: {e}")
        return None

    # 4. Create Z3 solver and add the formula
    s = z3.Solver()
    s.add(z3_formula)

    # 5. Check for satisfiability
    check_result = s.check()
    if check_result == z3.sat:
        model = s.model()
        solved_values = {}
        for k, v in z3_symbol_map.items():
            # Only extract values that Z3 actually produced a model for
            # (e.g., if it was a Z3 Real variable, not a fixed Python float)
            if isinstance(v, z3.ExprRef): # Check if it's a Z3 expression that was part of the solving
                evaluated_val = model.evaluate(v, model_completion=True)
                if evaluated_val is not None:
                    # Convert Z3 Real values to Python floats
                    solved_values[k] = float(evaluated_val.as_decimal(10))
            elif isinstance(v, (int, float)): # If it was a fixed Python float constant
                solved_values[k] = v
        return solved_values
    elif check_result == z3.unsat:
        print(f"Z3 Solver: Precondition is UNSAT. No initial state exists for: '{precondition_kx}'")
        return None
    else: # z3.unknown
        print(f"Z3 Solver: Precondition satisfiability UNKNOWN for: '{precondition_kx}'")
        return None
```

**Explanation:**
This function orchestrates the Z3 solving process to find a valid initial state.
*   **1. Prepare Z3 symbols**:
    *   It creates a `z3_symbol_map` dictionary.
    *   For each `state_variable` in the `variables` list, it creates a Z3 `Real` variable (e.g., `z3.Real('x')`) and stores it in the map. These are the values Z3 will try to find.
    *   For each `constant` in `definitions.constants`, if it has a fixed numeric `value`, that Python float value is stored. Otherwise (if its value is an expression or not specified), it's also treated as a Z3 `Real` variable, allowing Z3 to assign a value to it if it's part of the precondition.
*   **2. Expand predicates**: Calls `_expand_formula` to get the full logical expression of the precondition.
*   **3. Convert to Z3 expression**: Calls `_kx_expr_to_z3_formula` to transform the expanded KeYmaera X string into an actual Z3 expression object.
*   **4. Create Z3 solver and add formula**:
    *   `s = z3.Solver()`: Initializes a new Z3 solver instance.
    *   `s.add(z3_formula)`: Adds the converted Z3 formula as an assertion to the solver. This tells Z3: "Find values for the variables such that this formula is true."
*   **5. Check for satisfiability**:
    *   `check_result = s.check()`: Asks the solver to find a solution.
    *   **`z3.sat`**: If a solution is found, `s.check()` returns `z3.sat`.
        *   `model = s.model()`: Retrieves the satisfying assignment (the "model").
        *   It then iterates through `z3_symbol_map` to extract the values from the model. For Z3 `ExprRef` (the actual Z3 variables), `model.evaluate()` is used to get their numeric value, which is then converted to a Python `float`. Fixed Python constants are just copied.
        *   Returns a dictionary `solved_values` mapping variable/constant names to their float values.
    *   **`z3.unsat`**: If no solution exists (the precondition is logically impossible), it prints a warning and returns `None`.
    *   **`z3.unknown`**: If Z3 cannot determine satisfiability (e.g., due to complexity or limitations), it prints a warning and returns `None`.

---

### **4. Modified `_generate_reset_logic(...)`**

This function now incorporates the Z3 solver.

```python
def _generate_reset_logic(variables: List[VariableDeclarationConfig], precondition: PreconditionConfig, definitions: DefinitionsConfig) -> List[str]:
    """
    Generates the initial state assignment logic for the reset method,
    leveraging the precondition and predicate definitions.
    Prioritizes Z3 solver for initial state, falls back to heuristics if Z3 fails.
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
    solved_initial_state = _solve_initial_state_with_z3(
        precondition.formula, definitions, variables
    )

    if solved_initial_state:
        reset_code_lines.append("    # Initial state solved by Z3 (overriding defaults):")
        for var_name in [v.name for v in variables]:
            if var_name in solved_initial_state:
                reset_code_lines.append(f"    self.state['{var_name}'] = {solved_initial_state[var_name]}")
        
        # If Z3 solved for constants, ensure they are consistent with __init__ or update them if allowed.
        # For now, we assume constants are primarily set in __init__ and Z3 is for state variables.
        # If a constant's value was determined by Z3, it means it was treated as a variable.
        # This part of the code is executed in `reset()`, constants are typically set in `__init__`.
        # So we only assign to `self.state`.
        reset_code_lines.append("")
    else:
        reset_code_lines.append("    # Z3 could not find a satisfying initial state or encountered an error.")
        reset_code_lines.append("    # Falling back to heuristic defaults and adjustments.")
        # If Z3 failed, keep the heuristic adjustments for dx/dy and obstacle separation.

        # 2. Parse the expanded precondition for explicit assignments (fallback if Z3 fails)
        expanded_precondition_kx = _expand_formula(precondition.formula, definitions)
        assignment_matches = re.findall(r'(\b\w+\b)\s*(?:=|==)\s*([a-zA-Z0-9_.]+)', expanded_precondition_kx)
        
        if assignment_matches:
            reset_code_lines.append("    # Apply explicit assignments from the precondition (fallback)")
            for var_name, value_str in assignment_matches:
                if var_name in [v.name for v in variables]:
                    reset_code_lines.append(f"    self.state['{var_name}'] = {value_str}")
            reset_code_lines.append("")

        # 3. Special handling for dx, dy (isWellformedDir) (fallback if Z3 fails)
        if re.search(r'dx\^2\s*\+\s*dy\^2', expanded_precondition_kx):
            reset_code_lines.append(f"    # Ensure dx, dy satisfy isWellformedDir (approximate unit vector) (fallback)")
            reset_code_lines.append(f"    norm_sq = self.state['dx']**2 + self.state['dy']**2")
            reset_code_lines.append(f"    ud_val = self.constants.get('Ud', 0.0) # Get Ud from constants, default to 0.0 if not found")
            reset_code_lines.append(f"    min_norm_sq_target = (1.0 - ud_val)**2")
            reset_code_lines.append(f"    max_norm_sq_target = (1.0 + ud_val)**2")
            reset_code_lines.append(f"    if not (min_norm_sq_target <= norm_sq <= max_norm_sq_target):")
            reset_code_lines.append(f"        norm = np.sqrt(norm_sq)")
            reset_code_lines.append(f"        if norm > 1e-6: # Avoid division by zero")
            reset_code_lines.append(f"            self.state['dx'] /= norm")
            reset_code_lines.append(f"            self.state['dy'] /= norm")
            reset_code_lines.append(f"        else: # Default to (1,0) if too small")
            reset_code_lines.append(f"            self.state['dx'] = 1.0")
            reset_code_lines.append(f"            self.state['dy'] = 0.0")
            reset_code_lines.append("")

        # 4. Special handling for (x-xo)^2 + (y-yo)^2 > 0 (fallback if Z3 fails)
        if re.search(r'\(x\s*-\s*xo\)\^2\s*\+\s*\(y\s*-\s*yo\)\^2\s*>\s*0', expanded_precondition_kx):
            reset_code_lines.append(f"    # Ensure robot and obstacle are not at the same position (fallback)")
            reset_code_lines.append(f"    if self.state['x'] == self.state['xo'] and self.state['y'] == self.state['yo']:")
            reset_code_lines.append(f"        self.state['x'] += 0.1 # Slightly perturb x to ensure separation")
            reset_code_lines.append("")

    # 5. Final verification (always runs)
    # The _evaluate_expression method expects Python syntax, so format the expanded_precondition_kx now.
    final_precondition_for_eval = _format_kx_expr_for_python_eval_context(_expand_formula(precondition.formula, definitions))
    
    # Escape strings for embedding into the generated Python code's f-string
    original_precondition_formula_escaped = precondition.formula.replace("'", "\\'").replace('\\', '\\\\').replace('\n', ' ')
    final_precondition_for_eval_escaped = final_precondition_for_eval.replace("'", "\\'").replace('\\', '\\\\').replace('\n', ' ')

    reset_code_lines.append(f"    # Verify the full precondition after setting initial states")
    reset_code_lines.append(f"    precondition_satisfied = self._evaluate_expression('{final_precondition_for_eval_escaped}', is_constraint=True)")
    reset_code_lines.append(f"    if not precondition_satisfied:")
    reset_code_lines.append(f"        print(f\"Warning: Initial state does not satisfy precondition: '{original_precondition_formula_escaped}'. Expanded: '{final_precondition_for_eval_escaped}'.\")")
    reset_code_lines.append(f"        # If Z3 failed or heuristics didn't fully satisfy, this warning is crucial.")
    reset_code_lines.append("")

    return reset_code_lines
```

**Explanation:**
This is where the Z3 integration directly impacts the generated environment's `reset` method.
*   **1. Default initial values**: The function still starts by adding lines to initialize all state variables with some basic default values. This serves as a baseline, which will be overridden if Z3 finds a solution.
*   **`solved_initial_state = _solve_initial_state_with_z3(...)`**: This is the key call. It attempts to use the Z3 solver to find a set of initial values for the state variables and constants that satisfy the environment's precondition.
*   **`if solved_initial_state:`**:
    *   If Z3 successfully finds a solution (`solved_initial_state` is not `None`), the generated code will include lines that explicitly set `self.state[var_name]` to the values provided by Z3. This ensures the environment starts in a state guaranteed to satisfy the precondition (assuming Z3's model is correct).
    *   It explicitly states that these Z3-derived values override the defaults.
    *   A note is added about constants: Z3 might solve for constants if they are not explicitly defined with a value but appear in the precondition. However, the generated environment typically sets constants in `__init__`, so `reset()` primarily focuses on `self.state`.
*   **`else:` (Fallback to heuristics)**:
    *   If `_solve_initial_state_with_z3` returns `None` (meaning Z3 found the precondition unsatisfiable, unknown, or encountered an error), the generated code falls back to the previous heuristic-based approach.
    *   This includes:
        *   **Parsing explicit assignments**: It re-expands the precondition and uses regex to find simple `variable = value` assignments.
        *   **Special handling for `isWellformedDir`**: It includes code to normalize `dx` and `dy` to approximately satisfy the `isWellformedDir` predicate, which ensures the direction vector has a reasonable magnitude.
        *   **Special handling for obstacle separation**: It includes code to slightly perturb `x` if the robot and obstacle start at the exact same position, satisfying `(x-xo)^2 + (y-yo)^2 > 0`.
    *   This fallback mechanism ensures that the environment can still be generated and run even if Z3 fails, though the initial state might not be as rigorously guaranteed to satisfy all aspects of the precondition.
*   **5. Final verification**: Regardless of whether Z3 or the heuristics were used, the generated `reset` method always includes a final check. It evaluates the *expanded* precondition using the environment's `_evaluate_expression` method. If this check fails, it prints a warning, indicating that the initial state might be invalid. This acts as a safety net.

---

### **5. Other Functions (`_ast_to_python_code_string`, `generate_gym_environment_code`, `_generate_action_space_and_logic`, `_generate_ode_solver_logic`)**

These functions have not undergone significant changes related to Z3.

```python
def _ast_to_python_code_string(ast_node: ActionType, indent_level: int = 0,
                                all_constants: List[str] = None, all_state_vars: List[str] = None,
                                all_functions: List[str] = None, all_predicates: List[str] = None) -> str:
    # ... (unchanged logic) ...
    return f"ChoiceActionConfig(level_name='{ast_node.level_name}', choices=[\n" \
           f"{choices_str}\n" \
           f"{indent_str}])"

def generate_gym_environment_code(config: GymEnvironmentConfig) -> str:
    # ... (mostly unchanged, calls _generate_reset_logic) ...
    return env_code

def _generate_action_space_and_logic(hps: List[HybridProgramConfig]):
    # ... (unchanged logic) ...
    return action_space_definition, control_func_config.action_space_config, action_space_keys

def _generate_ode_solver_logic(hps: List[HybridProgramConfig], state_variable_names: List[str],
                               all_constants: List[str], all_function_names: List[str], all_predicate_names: List[str]):
    # ... (unchanged logic) ...
    return ode_func_code, ode_event_code, ode_state_map, ode_state_idx_map
```

**Explanation:**
*   **`_ast_to_python_code_string`**: This function converts the AST nodes representing actions (like `ChoiceActionConfig`, `SequenceActionConfig`, `DeterministicAssignmentActionConfig`) into Python code strings that, when executed, reconstruct these AST objects within the generated environment's `__init__` method. This is used to define the control logic. It remains unchanged.
*   **`generate_gym_environment_code`**: This is the main function that orchestrates the entire code generation. It calls all the helper functions to build different parts of the environment class. Its main change is that it now calls the *modified* `_generate_reset_logic` function.
*   **`_generate_action_space_and_logic`**: This function analyzes the `controlFunc` hybrid program to determine the structure of the Gymnasium `action_space` (e.g., `spaces.Dict`, `spaces.Discrete`, `spaces.Box`) and returns the Python code for its definition. It also returns the AST object for the control function itself. It remains unchanged.
*   **`_generate_ode_solver_logic`**: This function processes the `stepFunc` hybrid program to generate the Python code for the ODE system (`ode_func`) and any event functions (`ode_events`) required by `scipy.integrate.solve_ivp`. It remains unchanged.

---

### **6. Main Execution Block (`if __name__ == "__main__":`)**

```python
# --- Main Execution (for testing the generator) ---
if __name__ == "__main__":
    # ... (unchanged example AST construction) ...

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
    
    # ... (unchanged sanitization for example usage) ...

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
```

**Explanation:**
*   This block demonstrates how to use the generator by creating a sample `GymEnvironmentConfig` and then calling `generate_gym_environment_code` to produce the Python environment code.
*   The example usage instructions are updated to reflect the new `z3-solver` dependency.
*   The rest of the example AST construction and environment usage logic remains the same, as the Z3 integration is internal to how the `reset` method is generated, not how the environment is interacted with externally.

In summary, the Z3 integration significantly improves the `reset` functionality by leveraging a formal solver to find initial states that rigorously satisfy the precondition. This makes the generated environments more robust and reliable, especially for complex preconditions that heuristics might struggle with. The fallback mechanism ensures that the generator remains functional even if Z3 cannot find a solution.