# ast_config.py

from __future__ import annotations
import re
import logging
from typing import Any, Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ASTNode:
    """Base class for all nodes in the Abstract Syntax Tree."""
    def __init__(self, node_type: str, value: Any = None, children: Optional[list] = None, metadata: Optional[Dict] = None):
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []
        self.metadata = metadata if metadata is not None else {} # Ensure metadata is stored here

    def add_child(self, child: 'ASTNode'):
        self.children.append(child)

    def __repr__(self, level=0):
        indent = "  " * level
        value_str = ""
        if self.value is not None:
            value_str = f": {self.value}"
        meta_str = f" [Meta: {self.metadata}]" if self.metadata else ""
        s = f"{indent}[{self.node_type}]{value_str}{meta_str}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class Definition(ASTNode):
    """Represents a constant, function, or predicate definition with RL metadata."""
    def __init__(self, name: str, definition_type: str, value: Any = None, comment: Optional[str] = None, metadata: Optional[Dict] = None):
        super().__init__("Definition", name, metadata=metadata) # Pass metadata to ASTNode
        self.name = name
        self.definition_type = definition_type
        self.definition_value = value
        self.comment = comment
        # self.metadata is already set by super().__init__ from ASTNode

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
    def __init__(self, name: str, body: str, metadata: Optional[Dict] = None): # Added metadata parameter
        super().__init__(name, "HP", value=body, metadata=metadata) # Pass metadata to Definition
        self.name = name
        self.body = body

    def __repr__(self, level=0):
        indent = "  " * level
        cleaned_body = ' '.join(self.body.split())
        body_display = cleaned_body[:80] + "..." if len(cleaned_body) > 80 else cleaned_body
        meta_str = f" [Meta: {self.metadata}]" if self.metadata else ""
        s = f"{indent}[HP]: {self.name}{meta_str}\n"
        s += f"{indent}  Body (raw): {body_display}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class Variable(ASTNode):
    """Represents a variable declaration."""
    def __init__(self, name: str, var_type: str, comment: Optional[str] = None, metadata: Optional[Dict] = None): # Added metadata parameter
        super().__init__("Variable", name, metadata=metadata) # Pass metadata to ASTNode
        self.var_type = var_type # This is where var_type is assigned
        self.comment = comment
        # self.metadata is already set by super().__init__ from ASTNode

    def __repr__(self, level=0):
        indent = "  " * level
        meta_str = f" [Meta: {self.metadata}]" if self.metadata else ""
        return f"{indent}[VAR]: {self.var_type} {self.value}{meta_str}\n"

class ProblemStatement(ASTNode):
    """Represents the main Problem formula."""
    def __init__(self, formula: str, metadata: Optional[Dict] = None): # Added metadata parameter
        super().__init__("ProblemStatement", formula, metadata=metadata) # Pass metadata to ASTNode, formula as value
        self.formula = formula

    def __repr__(self, level=0):
        indent = "  " * level
        meta_str = f" [Meta: {self.metadata}]" if self.metadata else ""
        s = f"{indent}[PROBLEM]{meta_str}:\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

# --- NEW AST NODE TYPES FOR PROGRAM FRAGMENTS AND PROBLEM STRUCTURE ---

class IfElseStatement(ASTNode):
    def __init__(self, condition: str, metadata: Optional[Dict] = None):
        super().__init__("IfElseStatement", value=condition, metadata=metadata)

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[IfElseStatement]: if ({self.value})\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class ChoiceStatement(ASTNode):
    def __init__(self, metadata: Optional[Dict] = None):
        super().__init__("ChoiceStatement", metadata=metadata)

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[ChoiceStatement] (++)\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class SequenceStatement(ASTNode):
    def __init__(self, metadata: Optional[Dict] = None):
        super().__init__("SequenceStatement", metadata=metadata)

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[SequenceStatement] (;)\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class TestStatement(ASTNode):
    def __init__(self, formula: str, metadata: Optional[Dict] = None):
        super().__init__("TestStatement", value=formula, metadata=metadata)

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[TestStatement]: ?({self.value})\n"

class AssignmentStatement(ASTNode):
    def __init__(self, assignment: str, metadata: Optional[Dict] = None):
        super().__init__("AssignmentStatement", value=assignment, metadata=metadata)

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[AssignmentStatement]: {self.value}\n"

class HPCall(ASTNode):
    def __init__(self, hp_name: str, metadata: Optional[Dict] = None):
        super().__init__("HPCall", value=hp_name, metadata=metadata)

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[HPCall]: {self.value}\n"

class SafetyPostcondition(ASTNode):
    def __init__(self, metadata: Optional[Dict] = None):
        super().__init__("SafetyPostcondition", metadata=metadata)

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[SafetyPostcondition] ([] operator)\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class LivenessPostcondition(ASTNode):
    def __init__(self, metadata: Optional[Dict] = None):
        super().__init__("LivenessPostcondition", metadata=metadata)

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[LivenessPostcondition] (<> operator)\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s