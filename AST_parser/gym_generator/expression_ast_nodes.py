# expression_ast_nodes.py
# This module defines the AST node classes for representing KeYmaera X expressions in a structured way.
import numpy as np # For np.abs in to_python
from typing import Dict, Any, List, Optional, Union
import pyparsing as pp # Import pyparsing to check ParseResults type
import re 

class ExpressionASTNode:
    """Base class for all Abstract Syntax Tree nodes for expressions."""
    def to_python(self, context: Dict[str, Any]) -> str:
        """Converts the AST node to its Python string representation."""
        raise NotImplementedError(f"to_python not implemented for {self.__class__.__name__}")

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

class ExpressionFunctionCall(ExpressionASTNode):
    """
    Represents a function or predicate call (e.g., f(x, y)).
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