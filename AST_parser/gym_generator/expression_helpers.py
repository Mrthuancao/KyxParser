import re
from typing import List, Optional, Tuple, Union, Dict, Any
import pyparsing as pp
import z3 # NEW: Import z3

# Import the AST nodes
from .expression_ast_nodes import (
    ExpressionASTNode,
    ExpressionNumber,
    ExpressionBoolean,
    ExpressionIdentifier,
    ExpressionFunctionCall,
    ExpressionUnaryOp,
    ExpressionBinaryOp,
)

# Enable Packrat parsing for performance
pp.ParserElement.enablePackrat()

# Forward declaration for recursive grammar definition
_kx_expr_forward = pp.Forward()

# --- Helper functions ---

def _node_to_kx(node: Any) -> str:
    """
    Converts an AST node back into a KeYmaera X compatible string.
    This is crucial for predicate expansion substitution.
    """
    if isinstance(node, ExpressionNumber):
        return str(node.value)
    if isinstance(node, ExpressionBoolean):
        return str(node.value).lower()
    if isinstance(node, ExpressionIdentifier):
        return node.name
    if isinstance(node, ExpressionFunctionCall):
        # Handle cases where node.name might be an Identifier node or a string
        name = node.name.name if isinstance(node.name, ExpressionIdentifier) else str(node.name)
        args = ", ".join(_node_to_kx(arg) for arg in node.args)
        return f"{name}({args})"
    if isinstance(node, ExpressionUnaryOp):
        # For unary minus/plus, ensure it's not double-parenthesized if not needed
        if node.op in ['-', '+'] and isinstance(node.operand, (ExpressionNumber, ExpressionIdentifier, ExpressionFunctionCall)):
            return f"{node.op}{_node_to_kx(node.operand)}"
        return f"{node.op}({_node_to_kx(node.operand)})"
    if isinstance(node, ExpressionBinaryOp):
        return f"({_node_to_kx(node.left)} {node.op} {_node_to_kx(node.right)})"
    return str(node)

def _ensure_ast_node(item: Any) -> ExpressionASTNode:
    if isinstance(item, ExpressionASTNode):
        return item
    if isinstance(item, (pp.ParseResults, list)):
        if len(item) == 1:
            return _ensure_ast_node(item[0])
        elif len(item) == 0:
            raise ValueError("Encountered empty structure where an AST node was expected.")
    if item is None:
        raise ValueError("Encountered None where an AST node was expected.")
    raise TypeError(f"Unexpected type during AST conversion: {type(item)}. Value: {item}")

# --- Grammar Definition ---

def make_binary_op_node(tokens):
    # pyparsing returns tokens as a list, often with the actual elements nested
    # We expect a flattened list like [operand, op, operand, op, operand, ...]
    flat_list = tokens[0] # Access the inner list containing the actual parsed elements
    it = iter(flat_list)
    try:
        current_node = _ensure_ast_node(next(it))
    except StopIteration:
        return None # Should not happen for valid expressions

    while True:
        try:
            op = next(it)
            right_operand = _ensure_ast_node(next(it))
            current_node = ExpressionBinaryOp(current_node, op, right_operand)
        except StopIteration:
            break
    return current_node

def make_unary_op_node(tokens):
    # pyparsing for unary ops often returns [op, operand] or [[op, operand]]
    actual_tokens = tokens[0] # Access the inner list
    op = actual_tokens[0]
    operand = _ensure_ast_node(actual_tokens[1])
    return ExpressionUnaryOp(op, operand)


_number = pp.pyparsing_common.number().setParseAction(lambda t: ExpressionNumber(t[0]))
_boolean = (
    pp.Literal('true').setParseAction(lambda t: ExpressionBoolean('true'))
    | pp.Literal('false').setParseAction(lambda t: ExpressionBoolean('false'))
)
_identifier = pp.Word(pp.alphas + '_', pp.alphanums + '_').setParseAction(lambda t: ExpressionIdentifier(t[0]))

func_call_element = (
    _identifier + pp.Suppress('(') + pp.Optional(pp.delimitedList(_kx_expr_forward)) + pp.Suppress(')')
).setParseAction(lambda tokens: ExpressionFunctionCall(
    tokens[0], # The identifier is the first token
    [_ensure_ast_node(arg) for arg in tokens[1:]] # Arguments follow
))

_atomic_expr = (
    func_call_element
    | _number
    | _boolean
    | _identifier
    | (pp.Suppress('(') + _kx_expr_forward + pp.Suppress(')'))
)

_infix_grammar = pp.infixNotation(_atomic_expr,
    [
        (pp.oneOf('! + -'), 1, pp.opAssoc.RIGHT, make_unary_op_node),
        (pp.Literal('^'), 2, pp.opAssoc.RIGHT, make_binary_op_node),
        (pp.oneOf("* /"), 2, pp.opAssoc.LEFT, make_binary_op_node),
        (pp.oneOf("+ -"), 2, pp.opAssoc.LEFT, make_binary_op_node),
        (pp.oneOf("= != <= >= < >"), 2, pp.opAssoc.LEFT, make_binary_op_node),
        (pp.oneOf('& ∧'), 2, pp.opAssoc.LEFT, make_binary_op_node),
        (pp.oneOf('| ∨'), 2, pp.opAssoc.LEFT, make_binary_op_node),
        (pp.oneOf('-> →'), 2, pp.opAssoc.RIGHT, make_binary_op_node),
    ]
)

kx_expr = (_kx_expr_forward << _infix_grammar)

# --- Expansion Logic ---
def expand_formula(node_or_str: Union[ExpressionASTNode, str], definitions: Any = None, debug_mode: bool = False) -> Union[ExpressionASTNode, str]:
    """
    Recursively expands a formula, including user-defined functions and predicates.
    The top-level function handles string input/output, while an internal recursive
    helper handles AST node transformations.
    """
    is_input_string = isinstance(node_or_str, str)

    # 1. Initial Parse: Always work with a Node internally
    if is_input_string:
        try:
            clean_str = " ".join(node_or_str.split())
            parsed = kx_expr.parseString(clean_str, parseAll=True)
            current_node = _ensure_ast_node(parsed)
            if debug_mode:
                print(f"DEBUG_EXPAND_HELPER: Successfully parsed initial string to AST. Node type: {type(current_node)}")
        except Exception as e:
            if debug_mode:
                print(f"DEBUG_EXPAND_HELPER: Initial parsing failed for '{node_or_str}': {e}")
            # If it's not a valid expression, return as is (string)
            # This path means the input string was unparsable.
            return node_or_str
    else:
        current_node = node_or_str

    # 2. Internal recursive expansion function
    def _expand_node_recursive(node: ExpressionASTNode, definitions: Any, debug_mode: bool) -> ExpressionASTNode:
        """
        Recursively expands an AST node. This function always takes and returns an AST node.
        It iteratively applies expansions until no more changes occur for the current node.
        """
        if debug_mode:
            print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Entering _expand_node_recursive with node: {_node_to_kx(node)}")

        while True:
            prev_node_kx_str = _node_to_kx(node) # Store string representation to detect changes

            # Handle Function/Predicate Calls
            if definitions and isinstance(node, ExpressionFunctionCall):
                pred_func_name = node.name.name if isinstance(node.name, ExpressionIdentifier) else str(node.name)
                args = node.args
                
                if debug_mode:
                    print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Found ExpressionFunctionCall: '{pred_func_name}' with args: {[ _node_to_kx(a) for a in args ]}")

                expanded_in_this_pass = False

                # Try to expand as a PREDICATE first
                for pred_def in getattr(definitions, 'predicates', []):
                    if pred_def.name == pred_func_name:
                        if debug_mode:
                            print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Found predicate definition for '{pred_func_name}'. Body: '{pred_def.body}'")
                        
                        # FIX: Changed pred_def.formula to pred_def.body
                        body_str = pred_def.body
                        
                        # Recursively expand arguments first
                        expanded_args_kx_str = []
                        for arg_node in args:
                            expanded_arg_node = _expand_node_recursive(arg_node, definitions, debug_mode)
                            expanded_args_kx_str.append(_node_to_kx(expanded_arg_node))

                        if len(expanded_args_kx_str) != len(pred_def.parameters):
                            if debug_mode:
                                print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Mismatch in argument count for predicate '{pred_func_name}'. Expected {len(pred_def.parameters)}, got {len(expanded_args_kx_str)}. Skipping expansion of this call.")
                            continue # Try next predicate if there are multiple with same name (unlikely but possible)

                        # Apply substitutions to the predicate's body formula
                        substituted_body = body_str
                        for i, param_tuple in enumerate(pred_def.parameters):
                            param_name = param_tuple[1]
                            arg_kx_str = expanded_args_kx_str[i]
                            # Use word boundaries to avoid partial replacements (e.g., 'x' in 'xr')
                            pattern = rf'\b{re.escape(param_name)}\b'
                            substituted_body = re.sub(pattern, arg_kx_str, substituted_body)
                        
                        if debug_mode:
                            print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Substituted body for '{pred_func_name}': '{substituted_body}'")

                        try:
                            # Parse the substituted body string and recursively expand it
                            parsed_body = kx_expr.parseString(substituted_body, parseAll=True)
                            node = _expand_node_recursive(_ensure_ast_node(parsed_body), definitions, debug_mode)
                            expanded_in_this_pass = True
                        except Exception as e:
                            if debug_mode:
                                print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Error parsing expanded predicate body '{substituted_body}': {e}. Keeping original node.")
                            # If body_str is unparsable, keep the original node.
                            pass
                        break # Predicate found and processed

                if not expanded_in_this_pass: # If not a predicate, try functions
                    for func_def in getattr(definitions, 'functions', []):
                        if func_def.name == pred_func_name:
                            if debug_mode:
                                print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Found function definition for '{pred_func_name}'. Body: '{func_def.body}'")
                            
                            body_str = func_def.body

                            # Recursively expand arguments first
                            expanded_args_kx_str = []
                            for arg_node in args:
                                expanded_arg_node = _expand_node_recursive(arg_node, definitions, debug_mode)
                                expanded_args_kx_str.append(_node_to_kx(expanded_arg_node))

                            if len(expanded_args_kx_str) != len(func_def.parameters):
                                if debug_mode:
                                    print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Mismatch in argument count for function '{pred_func_name}'. Expected {len(func_def.parameters)}, got {len(expanded_args_kx_str)}. Skipping expansion of this call.")
                                continue

                            # Apply substitutions to the function's body formula
                            substituted_body = body_str
                            for i, param_tuple in enumerate(func_def.parameters):
                                param_name = param_tuple[1]
                                arg_kx_str = expanded_args_kx_str[i]
                                pattern = rf'\b{re.escape(param_name)}\b'
                                substituted_body = re.sub(pattern, arg_kx_str, substituted_body)
                            
                            if debug_mode:
                                print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Substituted body for '{pred_func_name}': '{substituted_body}'")

                            try:
                                # Parse the substituted body string and recursively expand it
                                parsed_body = kx_expr.parseString(substituted_body, parseAll=True)
                                node = _expand_node_recursive(_ensure_ast_node(parsed_body), definitions, debug_mode)
                                expanded_in_this_pass = True
                            except Exception as e:
                                if debug_mode:
                                    print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Error parsing expanded function body '{substituted_body}': {e}. Keeping original node.")
                                pass
                            break # Function found and processed
                
                if expanded_in_this_pass:
                    if debug_mode:
                        print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Expansion occurred for '{pred_func_name}'. Restarting iteration for new node: {_node_to_kx(node)}")
                    continue # Continue the while loop to re-evaluate the new 'node' for further expansions.
            
            # If no function/predicate call was expanded (or it's not a function call),
            # then recursively expand its children for binary and unary operations.
            children_changed = False
            if isinstance(node, ExpressionBinaryOp):
                old_left_kx_str = _node_to_kx(node.left)
                old_right_kx_str = _node_to_kx(node.right)
                
                node.left = _expand_node_recursive(node.left, definitions, debug_mode)
                node.right = _expand_node_recursive(node.right, definitions, debug_mode)

                if _node_to_kx(node.left) != old_left_kx_str or _node_to_kx(node.right) != old_right_kx_str:
                    children_changed = True

                # Apply logical equivalences for implication
                if node.op in ['->', '→']:
                    # !(A) | B
                    negated_left = ExpressionUnaryOp('!', node.left)
                    node = ExpressionBinaryOp(negated_left, '|', node.right)
                    if debug_mode:
                        print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Applied implication equivalence. New node: {_node_to_kx(node)}")
                    continue # Continue the loop as node has changed

            elif isinstance(node, ExpressionUnaryOp):
                old_operand_kx_str = _node_to_kx(node.operand)
                node.operand = _expand_node_recursive(node.operand, definitions, debug_mode)
                if _node_to_kx(node.operand) != old_operand_kx_str:
                    children_changed = True

                # Apply logical equivalences for negation
                if node.op == '!':
                    # Double Negation: !!A -> A
                    if isinstance(node.operand, ExpressionUnaryOp) and node.operand.op == '!':
                        node = node.operand.operand
                        if debug_mode:
                            print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Applied double negation. New node: {_node_to_kx(node)}")
                        continue # Continue the loop as node has changed
                    # De Morgan's: !(A & B) -> !A | !B
                    elif isinstance(node.operand, ExpressionBinaryOp):
                        if node.operand.op in ['&', '∧']:
                            node = ExpressionBinaryOp(ExpressionUnaryOp('!', node.operand.left), '|', ExpressionUnaryOp('!', node.operand.right))
                            if debug_mode:
                                print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Applied De Morgan's (AND). New node: {_node_to_kx(node)}")
                            continue # Continue the loop as node has changed
                        elif node.operand.op in ['|', '∨']:
                            node = ExpressionBinaryOp(ExpressionUnaryOp('!', node.operand.left), '&', ExpressionUnaryOp('!', node.operand.right))
                            if debug_mode:
                                print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Applied De Morgan's (OR). New node: {_node_to_kx(node)}")
                            continue # Continue the loop as node has changed
                        # For other binary ops, keep as is
            
            if children_changed:
                # If children were expanded, the current node might be eligible for further transformations
                # (e.g., if a child expansion resulted in `!!A` which is then simplified by the parent `!`).
                # So, continue the loop to re-evaluate the current node.
                continue

            # Check if any expansion or transformation occurred in this iteration.
            # If the string representation hasn't changed, we're done with this node.
            current_node_kx_str = _node_to_kx(node)
            if current_node_kx_str == prev_node_kx_str:
                if debug_mode:
                    print(f"DEBUG_EXPAND_HELPER_RECURSIVE: No further expansion/transformation in this iteration. Breaking loop for node: {current_node_kx_str}")
                break
            else:
                if debug_mode:
                    print(f"DEBUG_EXPAND_HELPER_RECURSIVE: Node changed. Previous: {prev_node_kx_str}, Current: {current_node_kx_str}. Continuing loop.")

        return node # Return the fully expanded AST node

    # Call the internal recursive expander
    final_expanded_node = _expand_node_recursive(current_node, definitions, debug_mode)

    # 3. Final Output: Match the input type
    if is_input_string:
        return _node_to_kx(final_expanded_node)
    return final_expanded_node



def format_kx_expr_for_python_lambda_body_generator(
    expr_str: str,
    lambda_param_names: List[str],
    all_constants: List[str],
    all_state_vars: List[str],
    all_functions: List[str],
    all_predicates: List[str],
) -> str:
    expr_cleaned = " ".join(expr_str.strip().split())
    if not expr_cleaned:
        raise ValueError("Cannot format an empty KeYmaera X expression.")

    context = {
        'lambda_param_names': lambda_param_names,
        'all_constants': all_constants,
        'all_state_vars': all_state_vars,
        'all_functions': all_functions,
        'all_predicates': all_predicates,
    }

    try:
        parsed_result = kx_expr.parseString(expr_cleaned, parseAll=True)
        ast_node = _ensure_ast_node(parsed_result)
        return ast_node.to_python(context)
    except Exception as e:
        raise ValueError(f"Error formatting expression '{expr_str}': {e}")


def format_kx_expr_for_z3_formula_generator(
    expr_str: str,
    definitions: Any, # Expects DefinitionsConfig for expansion
    z3_symbol_map: Dict[str, Any],
    fixed_constants: Dict[str, Union[int, float]],
    z3_equality_tolerance: float,
    debug_mode: bool = False
) -> Any:
    """
    Converts a KeYmaera X expression string into a Z3 formula using the AST.
    This function first expands KeYmaera X predicates, then parses the resulting
    string into an AST, and finally converts the AST to a Z3 expression.
    """
    expr_cleaned = " ".join(expr_str.strip().split())
    if not expr_cleaned:
        raise ValueError("Cannot format an empty KeYmaera X expression for Z3.")

    if debug_mode:
        print(f"DEBUG_CISB_HELPER: Calling expand_formula for Z3 conversion on: {expr_cleaned}")
    expanded_kx_expr_node = expand_formula(expr_cleaned, definitions, debug_mode) # Pass debug_mode
    
    # Ensure expanded_kx_expr_node is an AST node for subsequent processing
    if isinstance(expanded_kx_expr_node, str):
        # This means expand_formula failed to parse the initial string.
        # We must try to parse it here, otherwise to_z3 will fail.
        if debug_mode:
            print(f"DEBUG_CISB_HELPER: expand_formula returned a string (initial parse failed). Attempting to parse again for Z3 conversion: {expanded_kx_expr_node}")
        try:
            parsed_result = kx_expr.parseString(expanded_kx_expr_node, parseAll=True)
            ast_node = _ensure_ast_node(parsed_result)
        except Exception as e:
            raise ValueError(f"CRITICAL ERROR: Failed to parse (after expansion attempt) KeYmaera X expression '{expanded_kx_expr_node}' to AST for Z3 conversion: {e}")
    else:
        ast_node = expanded_kx_expr_node

    if debug_mode:
        print(f"DEBUG_CISB_HELPER: Fully expanded KeYmaera X AST for Z3 conversion: {_node_to_kx(ast_node)}")

    try:
        # 3. Convert the AST to a Z3 expression
        z3_context = {
            'z3_symbol_map': z3_symbol_map,
            'fixed_constants': fixed_constants,
            'z3_equality_tolerance': z3_equality_tolerance,
            'debug_mode': debug_mode
        }
        return ast_node.to_z3(z3_context)
    except Exception as e:
        raise ValueError(f"Error converting KeYmaera X expression '{expr_str}' (expanded: '{_node_to_kx(ast_node)}') to Z3 formula: {e}")