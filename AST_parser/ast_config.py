# ast_config.py
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple

# Using from __future__ import annotations for cleaner forward references
# This makes 'SequenceActionConfig' and 'ChoiceActionConfig' work without string literals
# if they were defined later in the same file, but string literals are still robust across files.


# --- Base AST Node ---
@dataclass
class PyASTNode:
    # node_type is a field with a default value and not part of __init__
    # It will be automatically set to the class name in __post_init__ if not explicitly overridden
    node_type: str = field(default="PyASTNode", init=False)

    def __post_init__(self):
        # Set node_type to the class name if it's still the default "PyASTNode"
        if self.node_type == "PyASTNode" and self.__class__.__name__ != "PyASTNode":
            self.node_type = self.__class__.__name__

# --- Definitions ---
@dataclass
class ConstantDefinitionConfig(PyASTNode):
    name: str
    value: str # Stored as string, will be evaluated

@dataclass
class FunctionDefinitionConfig(PyASTNode):
    name: str
    parameters: List[str]
    body: str # Stored as string, will be evaluated

@dataclass
class PredicateDefinitionConfig(PyASTNode):
    name: str
    parameters: List[str]
    body: str # Stored as string, will be evaluated

@dataclass
class DefinitionsConfig(PyASTNode):
    constants: List[ConstantDefinitionConfig] = field(default_factory=list)
    functions: List[FunctionDefinitionConfig] = field(default_factory=list)
    predicates: List[PredicateDefinitionConfig] = field(default_factory=list)

# --- Variables ---
@dataclass
class VariableConfig(PyASTNode):
    name: str
    type: str # e.g., 'real'
    lower_bound: Optional[str] = None # Stored as string, will be evaluated
    upper_bound: Optional[str] = None # Stored as string, will be evaluated

@dataclass
class VariablesConfig(PyASTNode):
    state_variables: List[VariableConfig] = field(default_factory=list)
    auxiliary_variables: List[VariableConfig] = field(default_factory=list)

# --- Control Function (Action Space) ---
@dataclass
class NondeterministicAssignmentActionConfig(PyASTNode):
    variable: str
    # Value is chosen by agent, so no 'value' field here.
    # Bounds for the choice might come from VariableConfig or ActionConstraintConfig.

@dataclass
class DeterministicAssignmentActionConfig(PyASTNode):
    variable: str
    value: str # Stored as string, will be evaluated

@dataclass
class ActionConstraintConfig(PyASTNode):
    formula: str # Stored as string, will be evaluated

@dataclass
class SequenceActionConfig(PyASTNode):
    actions: List[
        Union[
            NondeterministicAssignmentActionConfig,
            DeterministicAssignmentActionConfig,
            ActionConstraintConfig,
        ]
    ] = field(default_factory=list)

@dataclass
class ChoiceActionConfig(PyASTNode):
    level_name: str # e.g., 'ChoiceLevel_0'
    choices: List[SequenceActionConfig] = field(default_factory=list) # Each choice is a sequence of actions

@dataclass
class ControlFuncConfig(PyASTNode):
    # FIX: action_structure must come before any default arguments, or have a default itself.
    action_structure: Optional[Union[SequenceActionConfig, ChoiceActionConfig]] = field(default=None)
    name: str = "controlFunc"

# --- Step Function (Environment Dynamics) ---
@dataclass
class ODEConfig(PyASTNode):
    variable: str
    equation: str # Stored as string, will be evaluated

@dataclass
class EvolutionDomainConfig(PyASTNode):
    formula: str # Stored as string, will be evaluated. ODEs run while this is true.

@dataclass
class StepFuncConfig(PyASTNode):
    # CORRECTED ORDER: 'evolution_domain' (non-default) now comes before 'odes' (default)
    evolution_domain: EvolutionDomainConfig
    name: str = "stepFunc"
    odes: List[ODEConfig] = field(default_factory=list)
    duration: Optional[str] = None # Max duration of ODE integration, if specified

# --- Problem Configuration ---
@dataclass
class PreconditionConfig(PyASTNode):
    formula: str # Stored as string, will be evaluated

@dataclass
class PostconditionConfig(PyASTNode):
    formula: str # Stored as string, will be evaluated (e.g., reward, termination)

@dataclass
class LoopExecutionConfig(PyASTNode):
    iterations: str # e.g., '1' for single step, 'inf' for continuous

@dataclass
class ProblemConfig(PyASTNode):
    name: str # e.g., 'Problem'
    precondition: PreconditionConfig
    postcondition: PostconditionConfig
    loop_execution: LoopExecutionConfig

# --- Root Environment Config ---
@dataclass
class GymEnvironmentConfig(PyASTNode):
    env_name: str
    definitions: DefinitionsConfig
    variables: VariablesConfig
    control_func: ControlFuncConfig
    step_func: StepFuncConfig
    problem: ProblemConfig