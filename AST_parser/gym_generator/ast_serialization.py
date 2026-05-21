# ast_serialization.py
# This module defines the serialize_ast_node function, which recursively converts AST configuration objects into their Python constructor call strings. 
# This allows the full AST configuration to be embedded as a literal in the generated environment's __init__ method, 
# enabling runtime access to the original configuration for purposes like Z3 helper functions.
import textwrap
from typing import Any, List, Dict, Tuple

# Assuming pyast_config.py is in the same directory or importable
from pyast_config import (
    ASTConfigNode, # Import the base class
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

def serialize_ast_node(node: Any, indent_level: int = 0) -> str:
    """
    Recursively serializes an AST node into its Python constructor call string.
    This is used to embed the full AST configuration objects into the generated
    environment's __init__ method for later runtime access (e.g., by Z3 helpers).
    """
    indent_str = "    " * indent_level
    next_indent_str = "    " * (indent_level + 1)

    # Handle primitive types and collections
    if isinstance(node, bool):
        return str(node) # True/False
    if isinstance(node, (int, float)):
        return repr(node) # Use repr for numbers
    if node is None:
        return "None"

    if isinstance(node, str):
        # If the string contains newlines, serialize it using triple double quotes.
        # This prevents "unterminated string literal" errors when the generated
        # string is itself embedded as a literal in Python code (e.g., in a '''...''' block).
        if '\n' in node:
            # Escape any triple double quotes within the string to avoid premature termination
            # and ensure the literal is correctly parsed.
            escaped_node = node.replace('"""', '\\"\\"\\"')
            return f'"""{escaped_node}"""'
        else:
            # For single-line strings, repr() is generally sufficient and handles
            # single quotes, double quotes, and other special characters by escaping them.
            return repr(node)

    if isinstance(node, list):
        if not node:
            return "[]"
        items_str = ",\n".join([f"{next_indent_str}{serialize_ast_node(item, indent_level + 1)}" for item in node])
        return f"[\n{items_str}\n{indent_str}]"

    if isinstance(node, tuple):
        # For tuples, we need to ensure elements are serialized, then joined with ", "
        # and if it's a single element tuple, it needs a trailing comma.
        items_str = ", ".join([serialize_ast_node(item, 0) for item in node]) # Use 0 indent for tuple items for compactness
        return f"({items_str}{',' if len(node) == 1 else ''})"

    if isinstance(node, dict):
        if not node:
            return "{}"
        items_str = ",\n".join([
            f"{next_indent_str}{serialize_ast_node(k, indent_level + 1)}: {serialize_ast_node(v, indent_level + 1)}"
            for k, v in node.items()
        ])
        return f"{{\n{items_str}\n{indent_str}}}"

    # Handle ASTConfigNode instances dynamically
    if isinstance(node, ASTConfigNode):
        class_name = node.__class__.__name__

        # Use the _get_serialization_params method to get constructor arguments
        params = node._get_serialization_params()

        attributes = []
        for param_name, param_value in params.items():
            # Recursively serialize each parameter's value
            serialized_value = serialize_ast_node(param_value, indent_level + 1)
            attributes.append(f"{param_name}={serialized_value}")

        if not attributes:
            return f"{class_name}()"

        attributes_str = ",\n".join([f"{next_indent_str}{attr}" for attr in attributes])
        return f"{class_name}(\n{attributes_str}\n{indent_str})"
    else:
        # Fallback for unexpected types, or types not inheriting from ASTConfigNode
        # For example, if you have a custom object that isn't an ASTConfigNode but needs serialization
        # you might add a specific handler here, or raise an error.
        raise TypeError(f"Cannot serialize unknown or unhandled type: {type(node)}")