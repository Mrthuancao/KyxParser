# ast_node.py
from __future__ import annotations
import re
import logging

# Configure logging (you might want to customize this further in a real application)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- AST Node Definitions ---
class ASTNode:
    """Base class for all nodes in the Abstract Syntax Tree."""
    def __init__(self, node_type, value=None, children=None):
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self, level=0):
        """Generates a readable, indented representation of the tree."""
        indent = "  " * level
        value_str = ":"
        if self.value is not None:
            value_str += f" {self.value}"
        s = f"{indent}[{self.node_type}]{value_str}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class Definition(ASTNode):
    """Represents a constant, function, or predicate definition."""
    def __init__(self, name, definition_type, value=None, comment=None):
        super().__init__("Definition", name)
        self.name = name
        self.definition_type = definition_type
        self.definition_value = value # Stores the full, raw value
        self.comment = comment

    def __repr__(self, level=0):
        indent = "  " * level

        value_str = ""
        if self.definition_value is not None:
            # Clean up whitespace for display in KeYmaeraX AST
            cleaned_value = ' '.join(self.definition_value.split())

            # Truncate only if very long, otherwise display full value
            if len(cleaned_value) > 60:
                 value_str = f" = {cleaned_value[:60]}..."
            else:
                 value_str = f" = {cleaned_value}"

        comment_str = f" /* {self.comment} */" if self.comment else ""

        # Output format: [DEF] Type Name = Value
        s = f"{indent}[DEF]: {self.definition_type} {self.name}{value_str}{comment_str}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s


class HybridProgram(Definition):
    """
    Represents a Hybrid Program (HP) definition.
    It stores the raw body, and its children will be populated by the parser
    with the parsed internal structure (choices, sequences, ODEs, etc.).
    """
    def __init__(self, name, body):
        super().__init__(name, "HP", value=body)
        self.name = name # Store name for specific parsing logic
        self.body = body # Stores the raw body string

    def __repr__(self, level=0):
        indent = "  " * level

        # Clean up body display and truncate
        cleaned_body = ' '.join(self.body.split())

        # Specific truncation for controlFunc Body (raw) to match the example
        if self.name == 'controlFunc':
            body_display = cleaned_body[:80] + "..."
        # Specific truncation for stepFunc Body (raw) to match the example
        elif self.name == 'stepFunc':
            # Truncate after t' = 1
            idx = cleaned_body.find("t' = 1")
            if idx != -1:
                body_display = cleaned_body[:idx + len("t' = 1")] + " ..."
            else:
                body_display = cleaned_body[:80] + "..." if len(cleaned_body) > 80 else cleaned_body
        else:
            body_display = cleaned_body[:80] + "..." if len(cleaned_body) > 80 else cleaned_body

        if self.name:
            s = f"{indent}[HP]: {self.name}\n"
        else:
            s = f"{indent}[HP]:\n"
        s += f"{indent}  Body (raw): {body_display}\n"

        # Print children (parsed structure)
        for child in self.children:
            s += child.__repr__(level + 1)
        return s


class Variable(ASTNode):
    """Represents a variable declaration in the ProgramVariables block."""
    def __init__(self, name, var_type, comment=None):
        super().__init__("Variable", name)
        self.var_type = var_type
        self.comment = comment

    def __repr__(self, level=0):
        indent = "  " * level
        comment_str = f" /* {self.comment} */" if self.comment else ""
        return f"{indent}[VAR]: {self.var_type} {self.value}{comment_str}\n"

class ProblemStatement(ASTNode):
    """Represents the main Problem formula (P -> [HP] Q)."""
    def __init__(self, formula):
        super().__init__("ProblemStatement")
        self.formula = formula

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[PROBLEM]:\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s