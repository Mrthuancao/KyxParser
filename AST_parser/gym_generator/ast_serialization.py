import textwrap
from typing import Any, List, Dict

# Assuming pyast_config.py is in the same directory or importable
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

def serialize_ast_node(node: Any, indent_level: int = 0) -> str:
    """
    Recursively serializes an AST node into its Python constructor call string.
    This is used to embed the full AST configuration objects into the generated
    environment's __init__ method for later runtime access (e.g., by Z3 helpers).
    """
    indent_str = "    " * indent_level
    next_indent_str = "    " * (indent_level + 1)

    if isinstance(node, (str, int, float, bool)):
        return repr(node) # Use repr for primitives (handles strings, numbers, bools)

    if node is None:
        return "None"
    
    if isinstance(node, list):
        if not node:
            return "[]"
        items_str = ",\n".join([f"{next_indent_str}{serialize_ast_node(item, indent_level + 1)}" for item in node])
        return f"[\n{items_str}\n{indent_str}]"

    if isinstance(node, tuple):
        items_str = ", ".join([repr(item) for item in node])
        return f"({items_str})"

    if isinstance(node, dict):
        if not node:
            return "{}"
        items_str = ",\n".join([
            f"{next_indent_str}{serialize_ast_node(k, indent_level + 1)}: {serialize_ast_node(v, indent_level + 1)}"
            for k, v in node.items()
        ])
        return f"{{\n{items_str}\n{indent_str}}}"


    class_name = node.__class__.__name__
    attributes = []
    
    # Manually map attributes for each config type to control formatting and recursion
    if isinstance(node, GymEnvironmentConfig):
        attributes.append(f"env_name={serialize_ast_node(node.env_name)}")
        attributes.append(f"definitions={serialize_ast_node(node.definitions, indent_level + 1)}")
        attributes.append(f"variables={serialize_ast_node(node.variables, indent_level + 1)}")
        attributes.append(f"problem={serialize_ast_node(node.problem, indent_level + 1)}")
    elif isinstance(node, DefinitionsConfig):
        attributes.append(f"constants={serialize_ast_node(node.constants, indent_level + 1)}")
        attributes.append(f"functions={serialize_ast_node(node.functions, indent_level + 1)}")
        attributes.append(f"predicates={serialize_ast_node(node.predicates, indent_level + 1)}")
        attributes.append(f"hybrid_programs={serialize_ast_node(node.hybrid_programs, indent_level + 1)}")
    elif isinstance(node, VariablesConfig):
        attributes.append(f"state_variables={serialize_ast_node(node.state_variables, indent_level + 1)}")
    elif isinstance(node, ProblemConfig):
        attributes.append(f"precondition={serialize_ast_node(node.precondition, indent_level + 1)}")
        attributes.append(f"loop_execution={serialize_ast_node(node.loop_execution, indent_level + 1)}")
        attributes.append(f"postcondition={serialize_ast_node(node.postcondition, indent_level + 1)}")
    elif isinstance(node, ConstantDefinitionConfig):
        value_str = repr(node.value) if node.value is not None else "None"
        attributes.append(f"name={serialize_ast_node(node.name)}")
        attributes.append(f"type={serialize_ast_node(node.type)}")
        attributes.append(f"value={value_str}")
        attributes.append(f"metadata={serialize_ast_node(node.metadata)}")
    elif isinstance(node, FunctionDefinitionConfig):
        attributes.append(f"name={serialize_ast_node(node.name)}")
        attributes.append(f"return_type={serialize_ast_node(node.return_type)}")
        attributes.append(f"parameters={serialize_ast_node(node.parameters)}")
        attributes.append(f"body={repr(node.body)}")
        attributes.append(f"metadata={serialize_ast_node(node.metadata)}")
    elif isinstance(node, PredicateDefinitionConfig):
        attributes.append(f"name={serialize_ast_node(node.name)}")
        attributes.append(f"parameters={serialize_ast_node(node.parameters)}")
        attributes.append(f"body={repr(node.body)}")
        attributes.append(f"metadata={serialize_ast_node(node.metadata)}")
    elif isinstance(node, ControlFuncConfig):
        attributes.append(f"name={serialize_ast_node(node.name)}")
        attributes.append(f"action_space_config={serialize_ast_node(node.action_space_config, indent_level + 1)}")
        attributes.append(f"metadata={serialize_ast_node(node.metadata)}")
    elif isinstance(node, StepFuncConfig):
        attributes.append(f"name={serialize_ast_node(node.name)}")
        attributes.append(f"odes={serialize_ast_node(node.odes, indent_level + 1)}")
        attributes.append(f"evolution_domain={serialize_ast_node(node.evolution_domain, indent_level + 1)}")
        attributes.append(f"duration={serialize_ast_node(node.duration)}")
        attributes.append(f"raw_body_snippet={serialize_ast_node(node.raw_body_snippet)}")
        attributes.append(f"metadata={serialize_ast_node(node.metadata)}")
    elif isinstance(node, ODEConfig):
        attributes.append(f"variable={serialize_ast_node(node.variable)}")
        attributes.append(f"equation={repr(node.equation)}")
    elif isinstance(node, EvolutionDomainConfig):
        constraints_str = ", ".join([repr(c) for c in node.constraints])
        attributes.append(f"constraints=[{constraints_str}]")
    elif isinstance(node, VariableDeclarationConfig):
        attributes.append(f"name={serialize_ast_node(node.name)}")
        attributes.append(f"type={serialize_ast_node(node.type)}")
    elif isinstance(node, PreconditionConfig):
        attributes.append(f"formula={repr(node.formula)}")
    elif isinstance(node, PostconditionConfig):
        attributes.append(f"formula={repr(node.formula)}")
    elif isinstance(node, LoopExecutionConfig):
        attributes.append(f"iterations={serialize_ast_node(node.iterations)}")
        attributes.append(f"loop_invariant_formula={repr(node.loop_invariant_formula)}")
        attributes.append(f"loop_body={serialize_ast_node(node.loop_body, indent_level + 1)}")
    elif isinstance(node, HybridProgramStepConfig):
        attributes.append(f"hp_name={serialize_ast_node(node.hp_name)}")
        attributes.append(f"invariant_formula={repr(node.invariant_formula)}")
    elif isinstance(node, ChoiceActionConfig):
        attributes.append(f"level_name={serialize_ast_node(node.level_name)}")
        attributes.append(f"choices={serialize_ast_node(node.choices, indent_level + 1)}")
    elif isinstance(node, SequenceActionConfig):
        attributes.append(f"actions={serialize_ast_node(node.actions, indent_level + 1)}")
    elif isinstance(node, DeterministicAssignmentActionConfig):
        attributes.append(f"variable={serialize_ast_node(node.variable)}")
        attributes.append(f"value={repr(node.value)}")
    elif isinstance(node, NondeterministicAssignmentActionConfig):
        attributes.append(f"variable={serialize_ast_node(node.variable)}")
    elif isinstance(node, ActionConstraintConfig):
        attributes.append(f"formula={repr(node.formula)}")
    elif isinstance(node, CallHybridProgramConfig):
        attributes.append(f"hp_name={serialize_ast_node(node.hp_name)}")
    else:
        raise TypeError(f"Cannot serialize unknown AST node type: {type(node)}")

    if not attributes:
        return f"{class_name}()"

    attributes_str = ",\n".join([f"{next_indent_str}{attr}" for attr in attributes])
    return f"{class_name}(\n{attributes_str}\n{indent_str})"