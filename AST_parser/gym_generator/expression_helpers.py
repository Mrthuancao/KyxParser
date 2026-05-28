# expression_helpers.py
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
    Recursively expands a formula.
    """
    is_input_string = isinstance(node_or_str, str)

    # 1. Initial Parse: Always work with a Node internally
    if is_input_string:
        try:
            # Clean string of newlines/excess whitespace which trips up parseAll=True
            clean_str = " ".join(node_or_str.split())
            parsed = kx_expr.parseString(clean_str, parseAll=True)
            current_node = _ensure_ast_node(parsed)
        except Exception as e:
            # If it's not a valid expression, return as is
            return node_or_str
    else:
        current_node = node_or_str

    # 2. Expansion Logic
    result_node = current_node

    if definitions and isinstance(current_node, ExpressionFunctionCall):
        # The name could be an ExpressionIdentifier or a string if init was passed a string
        pred_name = current_node.name.name if isinstance(current_node.name, ExpressionIdentifier) else str(current_node.name)

        # Check predicates first
        for pred_def in getattr(definitions, 'predicates', []):
            if pred_def.name == pred_name:
                body_str = pred_def.body
                args = current_node.args

                # Replace parameters in the body string
                for i, param_obj in enumerate(pred_def.parameters):
                    if i < len(args):
                        param_name = param_obj[0] if isinstance(param_obj, (tuple, list)) else str(param_obj)
                        # Convert argument node to KX string for regex replacement
                        arg_kx_str = _node_to_kx(args[i])
                        pattern = rf'\b{re.escape(param_name)}\b'
                        body_str = re.sub(pattern, arg_kx_str, body_str)

                # Recursively expand the newly built body string
                expanded_body = expand_formula(body_str, definitions)
                # Ensure the result is a node for the rest of this function
                if isinstance(expanded_body, str):
                    parsed_body = kx_expr.parseString(expanded_body, parseAll=True)
                    result_node = _ensure_ast_node(parsed_body)
                else:
                    result_node = expanded_body
                break
        # If not a predicate, check functions (functions are not expanded for Z3 in current design)
        # This part of expansion is primarily for predicates.

    elif isinstance(current_node, ExpressionBinaryOp):
        left_expanded = expand_formula(current_node.left, definitions)
        # Ensure the result is an AST node for subsequent operations
        if isinstance(left_expanded, str):
            left_expanded = _ensure_ast_node(kx_expr.parseString(left_expanded, parseAll=True))

        right_expanded = expand_formula(current_node.right, definitions)
        if isinstance(right_expanded, str):
            right_expanded = _ensure_ast_node(kx_expr.parseString(right_expanded, parseAll=True))

        if current_node.op in ['->', '→']:
            negated_left = ExpressionUnaryOp('!', left_expanded)
            result_node = ExpressionBinaryOp(negated_left, '|', right_expanded)
        else:
            result_node = ExpressionBinaryOp(left_expanded, current_node.op, right_expanded)

    elif isinstance(current_node, ExpressionUnaryOp):
        operand_expanded = expand_formula(current_node.operand, definitions)
        if isinstance(operand_expanded, str):
            operand_expanded = _ensure_ast_node(kx_expr.parseString(operand_expanded, parseAll=True))

        if current_node.op == '!':
            # Double Negation: !!A -> A
            if isinstance(operand_expanded, ExpressionUnaryOp) and operand_expanded.op == '!':
                result_node = operand_expanded.operand
            # De Morgan's: !(A & B) -> !A | !B
            elif isinstance(operand_expanded, ExpressionBinaryOp):
                if operand_expanded.op in ['&', '∧']:
                    result_node = ExpressionBinaryOp(ExpressionUnaryOp('!', operand_expanded.left), '|', ExpressionUnaryOp('!', operand_expanded.right))
                elif operand_expanded.op in ['|', '∨']:
                    result_node = ExpressionBinaryOp(ExpressionUnaryOp('!', operand_expanded.left), '&', ExpressionUnaryOp('!', operand_expanded.right))
                else:
                    result_node = ExpressionUnaryOp(current_node.op, operand_expanded)
            else:
                result_node = ExpressionUnaryOp(current_node.op, operand_expanded)
        else:
            result_node = ExpressionUnaryOp(current_node.op, operand_expanded)

    # 3. Final Output: Match the input type
    if is_input_string:
        return _node_to_kx(result_node)
    return result_node


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
    expanded_kx_expr = expand_formula(expr_cleaned, definitions)

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