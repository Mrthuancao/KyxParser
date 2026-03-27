# ast_config.py

from __future__ import annotations
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ASTNode:
    """Base class for all nodes in the Abstract Syntax Tree."""
    def __init__(self, node_type, value=None, children=None):
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self, level=0):
        indent = "  " * level
        value_str = ":"
        if self.value is not None:
            value_str += f" {self.value}"
        s = f"{indent}[{self.node_type}]{value_str}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class Definition(ASTNode):
    """Represents a constant, function, or predicate definition with RL metadata."""
    def __init__(self, name, definition_type, value=None, comment=None, metadata=None):
        super().__init__("Definition", name)
        self.name = name
        self.definition_type = definition_type
        self.definition_value = value 
        self.comment = comment
        # Stores RL annotations like {'anno_type': 'safety', 'penalty': '-100'}
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self, level=0):
        indent = "  " * level
        value_str = ""
        if self.definition_value:
            cleaned_value = ' '.join(self.definition_value.split())
            value_str = f" = {cleaned_value[:60]}..." if len(cleaned_value) > 60 else f" = {cleaned_value}"

        meta_str = f" [Meta: {self.metadata}]" if self.metadata else ""
        s = f"{indent}[DEF]: {self.definition_type} {self.name}{value_str}{meta_str}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class HybridProgram(Definition):
    """Represents a Hybrid Program (HP) definition."""
    def __init__(self, name, body):
        super().__init__(name, "HP", value=body)
        self.name = name
        self.body = body

    def __repr__(self, level=0):
        indent = "  " * level
        cleaned_body = ' '.join(self.body.split())
        body_display = cleaned_body[:80] + "..."
        s = f"{indent}[HP]: {self.name}\n"
        s += f"{indent}  Body (raw): {body_display}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class Variable(ASTNode):
    """Represents a variable declaration."""
    def __init__(self, name, var_type, comment=None):
        super().__init__("Variable", name)
        self.var_type = var_type
        self.comment = comment

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[VAR]: {self.var_type} {self.value}\n"

class ProblemStatement(ASTNode):
    """Represents the main Problem formula."""
    def __init__(self, formula):
        super().__init__("ProblemStatement")
        self.formula = formula

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[PROBLEM]:\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s