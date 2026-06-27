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
def expand_formula(node_or_str: Union[ExpressionASTNode, str], definitions: Any = None) -> Union[ExpressionASTNode, str]:
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
        except Exception as e:
            # If it's not a valid expression, return as is (string)
            # This path means the input string was unparsable.
            return node_or_str
    else:
        current_node = node_or_str

    # 2. Internal recursive expansion function
    def _expand_node_recursive(node: ExpressionASTNode, definitions: Any) -> ExpressionASTNode:
        """
        Recursively expands an AST node. This function always takes and returns an AST node.
        It iteratively applies expansions until no more changes occur for the current node.
        """
        while True:
            prev_node_kx_str = _node_to_kx(node) # Store string representation to detect changes

            # Handle Function/Predicate Calls
            if definitions and isinstance(node, ExpressionFunctionCall):
                pred_func_name = node.name.name if isinstance(node.name, ExpressionIdentifier) else str(node.name)
                args = node.args
                
                expanded_in_this_pass = False

                # Try to expand as a PREDICATE first
                # Note: The provided AST nodes don't have a separate ExpressionPredicateCall,
                # so it's treated as a FunctionCall. The logic below correctly handles both.
                for pred_def in getattr(definitions, 'predicates', []):
                    if pred_def.name == pred_func_name:
                        body_str = pred_def.body
                        for i, param_tuple in enumerate(pred_def.parameters): # param_tuple is ('Real', 'xr')
                            if i < len(args):
                                # CORRECTED: Get parameter name from index 1 of the tuple
                                param_name = param_tuple[1] 
                                arg_kx_str = _node_to_kx(args[i])
                                pattern = rf'\b{re.escape(param_name)}\b'
                                body_str = re.sub(pattern, arg_kx_str, body_str)
                        
                        try:
                            # Parse the body string and recursively expand it
                            parsed_body = kx_expr.parseString(body_str, parseAll=True)
                            node = _expand_node_recursive(_ensure_ast_node(parsed_body), definitions)
                            expanded_in_this_pass = True
                        except Exception:
                            # If body_str is unparsable, keep the original node.
                            pass
                        break # Predicate found and processed

                if not expanded_in_this_pass: # If not a predicate, try functions
                    for func_def in getattr(definitions, 'functions', []):
                        if func_def.name == pred_func_name:
                            body_str = func_def.body
                            for i, param_tuple in enumerate(func_def.parameters): # param_tuple is ('Real', 'v')
                                if i < len(args):
                                    # CORRECTED: Get parameter name from index 1 of the tuple
                                    param_name = param_tuple[1] 
                                    arg_kx_str = _node_to_kx(args[i])
                                    pattern = rf'\b{re.escape(param_name)}\b'
                                    body_str = re.sub(pattern, arg_kx_str, body_str)
                            
                            try:
                                # Parse the body string and recursively expand it
                                parsed_body = kx_expr.parseString(body_str, parseAll=True)
                                node = _expand_node_recursive(_ensure_ast_node(parsed_body), definitions)
                                expanded_in_this_pass = True
                            except Exception:
                                # If body_str is unparsable, keep the original node.
                                pass
                            break # Function found and processed
                
                if expanded_in_this_pass:
                    # If a function/predicate was expanded, continue the while loop
                    # to re-evaluate the new 'node' for further expansions.
                    continue 
            
            # If no function/predicate call was expanded (or it's not a function call),
            # then recursively expand its children for binary and unary operations.
            if isinstance(node, ExpressionBinaryOp):
                node.left = _expand_node_recursive(node.left, definitions)
                node.right = _expand_node_recursive(node.right, definitions)

                # Apply logical equivalences for implication
                if node.op in ['->', '→']:
                    negated_left = ExpressionUnaryOp('!', node.left)
                    node = ExpressionBinaryOp(negated_left, '|', node.right)

            elif isinstance(node, ExpressionUnaryOp):
                node.operand = _expand_node_recursive(node.operand, definitions)

                # Apply logical equivalences for negation
                if node.op == '!':
                    # Double Negation: !!A -> A
                    if isinstance(node.operand, ExpressionUnaryOp) and node.operand.op == '!':
                        node = node.operand.operand
                    # De Morgan's: !(A & B) -> !A | !B
                    elif isinstance(node.operand, ExpressionBinaryOp):
                        if node.operand.op in ['&', '∧']:
                            node = ExpressionBinaryOp(ExpressionUnaryOp('!', node.operand.left), '|', ExpressionUnaryOp('!', node.operand.right))
                        elif node.operand.op in ['|', '∨']:
                            node = ExpressionBinaryOp(ExpressionUnaryOp('!', node.operand.left), '&', ExpressionUnaryOp('!', node.operand.right))
                        # For other binary ops, keep as is
            
            # Check if any expansion or transformation occurred in this iteration.
            # If the string representation hasn't changed, we're done with this node.
            if _node_to_kx(node) == prev_node_kx_str:
                break 
            # Else, continue the while loop to check for further expansions (e.g., De Morgan's could reveal another !!)

        return node # Return the fully expanded AST node

    # Call the internal recursive expander
    final_expanded_node = _expand_node_recursive(current_node, definitions)

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

    # 1. Expand KeYmaera X predicates (and potentially functions if desired) at the string level.
    # This is important because Z3 does not automatically understand KeYmaera X's predicate definitions.
    # The `expand_formula` function is already equipped to do this.
    print(f"DEBUG_CISB: Expanding KeYmaera X expression for Z3 conversion: {expr_cleaned}")
    expanded_kx_expr = expand_formula(expr_cleaned, definitions)
    print(f"DEBUG_CISB: Expanded KeYmaera X expression: {expanded_kx_expr}")
    # Ensure expanded_kx_expr is a string, then parse it into an AST
    if not isinstance(expanded_kx_expr, str):
        # If expand_formula returned an AST node, convert it back to string for consistent parsing
        expanded_kx_expr = _node_to_kx(expanded_kx_expr)

    try:
        # 2. Parse the expanded KeYmaera X expression into an AST
        parsed_result = kx_expr.parseString(expanded_kx_expr, parseAll=True)
        ast_node = _ensure_ast_node(parsed_result)

        # 3. Convert the AST to a Z3 expression
        z3_context = {
            'z3_symbol_map': z3_symbol_map,
            'fixed_constants': fixed_constants,
            'z3_equality_tolerance': z3_equality_tolerance,
            'debug_mode': debug_mode
        }
        return ast_node.to_z3(z3_context)
    except Exception as e:
        raise ValueError(f"Error converting KeYmaera X expression '{expr_str}' (expanded: '{expanded_kx_expr}') to Z3 formula: {e}")