# expression_ast_nodes.py
# This module defines the AST node classes for representing KeYmaera X expressions in a structured way.
import numpy as np
from typing import Dict, Any, List, Optional, Union
import pyparsing as pp
import re
import z3 # NEW: Import z3

# Helper function for _node_to_kx used in to_z3 error messages
# This is a local helper to avoid circular imports if expression_helpers also defines it.
def _node_to_kx_internal(node: Any) -> str:
    """Converts an AST node back into a KeYmaera X compatible string for internal use."""
    if isinstance(node, ExpressionNumber):
        return str(node.value)
    if isinstance(node, ExpressionBoolean):
        return str(node.value).lower()
    if isinstance(node, ExpressionIdentifier):
        return node.name
    if isinstance(node, ExpressionFunctionCall):
        name = node.name # Already a string in this conceptual implementation
        args = ", ".join(_node_to_kx_internal(arg) for arg in node.args)
        return f"{name}({args})"
    if isinstance(node, ExpressionUnaryOp):
        return f"{node.op}({_node_to_kx_internal(node.operand)})"
    if isinstance(node, ExpressionBinaryOp):
        return f"({_node_to_kx_internal(node.left)} {node.op} {_node_to_kx_internal(node.right)})"
    return str(node) # Fallback for unexpected types

class ExpressionASTNode:
    """Base class for all Abstract Syntax Tree nodes for expressions."""
    def to_python(self, context: Dict[str, Any]) -> str:
        """Converts the AST node to its Python string representation."""
        raise NotImplementedError(f"to_python not implemented for {self.__class__.__name__}")

    def to_z3(self, z3_context: Dict[str, Any]) -> Any: # NEW method
        """Converts the AST node to its Z3 expression representation."""
        raise NotImplementedError(f"to_z3 not implemented for {self.__class__.__name__}")

class ExpressionNumber(ExpressionASTNode):
    """Represents a numeric literal in the expression."""
    def __init__(self, tokens_or_value: Union[pp.ParseResults, int, float, str]):
        if isinstance(tokens_or_value, (pp.ParseResults, list)):
            if not tokens_or_value:
                raise ValueError("ExpressionNumber received empty tokens list.")
            self.value = tokens_or_value[0]
        else:
            self.value = tokens_or_value

        if not isinstance(self.value, (int, float)):
            try: # Attempt conversion if it's a string that looks like a number
                self.value = float(self.value)
            except (ValueError, TypeError):
                raise TypeError(f"ExpressionNumber expected a number, but received {type(self.value)}: {self.value}")

    def __repr__(self):
        return f"ExpressionNumber({self.value})"
    def to_python(self, context: Dict[str, Any]) -> str:
        return str(self.value)
    def to_z3(self, z3_context: Dict[str, Any]) -> Any: # NEW
        if isinstance(self.value, int):
            return z3.IntVal(self.value)
        return z3.RealVal(self.value)

class ExpressionBoolean(ExpressionASTNode):
    """Represents a boolean literal (true/false) in the expression."""
    def __init__(self, tokens_or_value: Union[pp.ParseResults, str]):
        if isinstance(tokens_or_value, (pp.ParseResults, list)):
            if not tokens_or_value:
                raise ValueError("ExpressionBoolean received empty tokens list.")
            raw_value = tokens_or_value[0]
        else:
            raw_value = tokens_or_value

        if not isinstance(raw_value, str) or raw_value.lower() not in ['true', 'false']:
            raise TypeError(f"ExpressionBoolean expected 'true' or 'false' string, but received {type(raw_value)}: {raw_value}")

        self.value = raw_value.lower() == 'true'

    def __repr__(self):
        return f"ExpressionBoolean({self.value})"
    def to_python(self, context: Dict[str, Any]) -> str:
        return str(self.value) # Python uses True/False
    def to_z3(self, z3_context: Dict[str, Any]) -> Any: # NEW
        return z3.BoolVal(self.value)

class ExpressionIdentifier(ExpressionASTNode):
    """Represents a variable, constant, function name, or predicate name."""
    def __init__(self, tokens_or_value: Union[pp.ParseResults, str]):
        if isinstance(tokens_or_value, (pp.ParseResults, list)):
            if not tokens_or_value:
                raise ValueError("ExpressionIdentifier received empty tokens list.")
            self.name = tokens_or_value[0]
        else:
            self.name = tokens_or_value

        if not isinstance(self.name, str):
            raise TypeError(f"ExpressionIdentifier expected a string, but received {type(self.name)}: {self.name}")

    def __repr__(self):
        return f"ExpressionIdentifier('{self.name}')"
    def to_python(self, context: Dict[str, Any]) -> str:
        """
        Converts the identifier to its Python representation, applying scope prefixing.
        """
        if self.name in context.get('lambda_param_names', []):
            return self.name
        elif self.name in context.get('all_constants', []):
            return f"self.constants['{self.name}']"
        elif self.name in context.get('all_state_vars', []):
            return f"self.state['{self.name}']"
        elif self.name in context.get('all_functions', []):
            return f"self.functions['{self.name}']"
        elif self.name in context.get('all_predicates', []):
            return f"self.predicates['{self.name}']"
        elif self.name == 'np':
            return 'np'
        return self.name
    def to_z3(self, z3_context: Dict[str, Any]) -> Any: # NEW
        # Z3 context contains z3_symbol_map and fixed_constants
        z3_symbol_map = z3_context['z3_symbol_map']
        fixed_constants = z3_context['fixed_constants']

        if self.name in z3_symbol_map:
            return z3_symbol_map[self.name]
        elif self.name in fixed_constants:
            val = fixed_constants[self.name]
            if isinstance(val, int):
                return z3.IntVal(val)
            return z3.RealVal(val)
        else:
            # Fallback for identifiers not explicitly mapped (e.g., function names themselves)
            # This case should ideally be handled by pre-expansion or the calling context.
            # For now, create a Real variable, but this might lead to unexpected behavior
            # if it was meant to be a function name or an unhandled literal.
            if z3_context.get('debug_mode', False):
                print(f"DEBUG_Z3: Warning: Identifier '{self.name}' not found in z3_symbol_map or fixed_constants. Creating a Z3 Real variable.")
            return z3.Real(self.name)


class ExpressionFunctionCall(ExpressionASTNode):
    """
    Represents a function or predicate call (e.g., f(x, y)).
    NOTE: For Z3 conversion, this typically only handles direct Z3 functions like `abs`,
          as KeYmaera X functions/predicates are usually expanded *before* AST conversion to Z3.
    """
    def __init__(self, name: ExpressionIdentifier, args: List[ExpressionASTNode]):
        if not isinstance(name, ExpressionIdentifier):
            # Handle case where name might be passed as string
            self.name = str(name)
        else:
            self.name = name.name
        self.args = args

    def __repr__(self):
        return f"ExpressionFunctionCall('{self.name}', {self.args})"
    def to_python(self, context: Dict[str, Any]) -> str:
        """
        Converts the function call to its Python representation.
        """
        py_args = [arg.to_python(context) for arg in self.args]
        args_str = ', '.join(py_args)

        if self.name == 'abs':
            return f"np.abs({args_str})"
        elif self.name in context.get('all_functions', []):
            return f"self.functions['{self.name}']({args_str})"
        elif self.name in context.get('all_predicates', []):
            return f"self.predicates['{self.name}']({args_str})"

        return f"{self.name}({args_str})"
    def to_z3(self, z3_context: Dict[str, Any]) -> Any: # NEW
        z3_args = [arg.to_z3(z3_context) for arg in self.args]
        if self.name == 'abs':
            if len(z3_args) != 1:
                raise ValueError("Z3 abs() expects exactly one argument.")
            return z3.Abs(z3_args[0])
        else:
            # If a function call reaches here, it implies it was not expanded
            # before AST conversion. This might be a Z3-specific function
            # or an unhandled KeYmaera X function.
            # For now, we'll try to create a Z3 function application.
            if z3_context.get('debug_mode', False):
                print(f"DEBUG_Z3: Warning: Unexpanded function call '{self.name}' encountered during to_z3. Attempting Z3 function application.")
            # This requires defining Z3 functions, which is more complex.
            # For the current CISB usage, KeYmaera X functions/predicates are expanded
            # at the string level before AST parsing, so this branch should ideally not be hit
            # for user-defined functions/predicates.
            # If it is, it means the pre-expansion failed or was skipped.
            # For now, let's raise an error or return a placeholder if not a known Z3 func.
            raise NotImplementedError(f"Unsupported function call '{self.name}' for direct Z3 conversion. Ensure KeYmaera X functions are expanded before AST to Z3 conversion.")


class ExpressionUnaryOp(ExpressionASTNode):
    """Represents a unary operation (e.g., !A)."""
    def __init__(self, op_str: str, operand_node: ExpressionASTNode):
        self.op = op_str
        self.operand = operand_node

    def __repr__(self):
        return f"ExpressionUnaryOp('{self.op}', {self.operand})"
    def to_python(self, context: Dict[str, Any]) -> str:
        py_operand = self.operand.to_python(context)
        if self.op == '!':
            return f"(not ({py_operand}))"
        elif self.op == '+':
            return f"(+({py_operand}))"
        elif self.op == '-':
            return f"(-({py_operand}))"
        return f"({self.op}{py_operand})"
    def to_z3(self, z3_context: Dict[str, Any]) -> Any: # NEW
        z3_operand = self.operand.to_z3(z3_context)
        if self.op == '!':
            return z3.Not(z3_operand)
        elif self.op == '+':
            return z3_operand # Unary plus is identity
        elif self.op == '-':
            return -z3_operand
        raise ValueError(f"Unsupported unary operator for Z3: {self.op}")

class ExpressionBinaryOp(ExpressionASTNode):
    """Represents a binary operation (e.g., A + B)."""
    def __init__(self, left_node: ExpressionASTNode, op_str: str, right_node: ExpressionASTNode):
        self.left = left_node
        self.op = op_str
        self.right = right_node

    def __repr__(self):
        return f"ExpressionBinaryOp({self.left}, '{self.op}', {self.right})"
    def to_python(self, context: Dict[str, Any]) -> str:
        """
        Converts the binary operation to its Python representation, mapping KeYmaera X
        operators to Python equivalents.
        """
        py_left = self.left.to_python(context)
        py_right = self.right.to_python(context)

        # Logical and Comparison Mappings
        if self.op == '=':
            return f"self._approx_eq({py_left}, {py_right})"
        elif self.op == '!=':
            return f"(not self._approx_eq({py_left}, {py_right}))"
        elif self.op in ['&', '∧']:
            return f"(({py_left}) and ({py_right}))"
        elif self.op in ['|', '∨']:
            return f"(({py_left}) or ({py_right}))"
        elif self.op in ['->', '→']:
            return f"((not ({py_left})) or ({py_right}))"
        elif self.op == '^':
            return f"(({py_left}) ** ({py_right}))"

        # Standard arithmetic and basic comparisons
        return f"(({py_left}) {self.op} ({py_right}))"
    def to_z3(self, z3_context: Dict[str, Any]) -> Any: # NEW
        z3_left = self.left.to_z3(z3_context)
        z3_right = self.right.to_z3(z3_context)
        z3_equality_tolerance = z3_context['z3_equality_tolerance']
        debug_mode = z3_context.get('debug_mode', False)

        if self.op == '=':
            # Approximate equality for Z3
            return z3.And(z3_left >= z3_right - z3_equality_tolerance,
                          z3_left <= z3_right + z3_equality_tolerance)
        elif self.op == '!=':
            # Negation of approximate equality
            return z3.Not(z3.And(z3_left >= z3_right - z3_equality_tolerance,
                                 z3_left <= z3_right + z3_equality_tolerance))
        elif self.op == '<=':
            return z3_left <= z3_right
        elif self.op == '>=':
            return z3_left >= z3_right
        elif self.op == '<':
            return z3_left < z3_right
        elif self.op == '>':
            return z3_left > z3_right
        elif self.op in ['&', '∧']:
            return z3.And(z3_left, z3_right)
        elif self.op in ['|', '∨']:
            return z3.Or(z3_left, z3_right)
        elif self.op in ['->', '→']:
            return z3.Implies(z3_left, z3_right)
        elif self.op == '+':
            return z3_left + z3_right
        elif self.op == '-':
            return z3_left - z3_right
        elif self.op == '*':
            return z3_left * z3_right
        elif self.op == '/':
            # Z3 division for reals
            return z3_left / z3_right
        elif self.op == '^':
            # FIX: Z3 does not have z3.Power. Use Python's ** for Z3 expressions
            # For integer exponents, Z3 expressions support the ** operator directly.
            if isinstance(self.right, ExpressionNumber):
                exponent_val = self.right.value
                if isinstance(exponent_val, int):
                    if debug_mode:
                        print(f"DEBUG_Z3_CONV: Converting '{_node_to_kx_internal(self.left)} ^ {exponent_val}' to Z3: {z3_left} ** {exponent_val}")
                    return z3_left ** exponent_val
                else:
                    raise ValueError(f"Z3 does not directly support non-integer exponents like '{exponent_val}' for real variables. Expression: '{_node_to_kx_internal(self.left)} ^ {_node_to_kx_internal(self.right)}'")
            else:
                raise ValueError(f"Z3 exponentiation requires a numeric literal exponent. Found: '{_node_to_kx_internal(self.right)}'. Expression: '{_node_to_kx_internal(self.left)} ^ {_node_to_kx_internal(self.right)}'")
        raise ValueError(f"Unsupported binary operator for Z3: {self.op}")