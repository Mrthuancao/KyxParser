# ast_config.py
from __future__ import annotations # Enables postponed evaluation of type annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Any

# --- Base AST Node for PyAST ---
@dataclass
class PyASTNode:
    """Base class for all nodes in the intermediate Python AST."""
    node_type: str = field(default="PyASTNode", init=False)

    def __post_init__(self):
        # Automatically set node_type to the class name if not explicitly set
        if self.node_type == "PyASTNode" and self.__class__.__name__ != "PyASTNode":
            self.node_type = self.__class__.__name__

    def __repr__(self, level=0):
        """Generates a readable, indented representation of the intermediate AST."""
        indent = "  " * level
        name_field = getattr(self, 'name', None)
        name_str = f"({name_field})" if name_field else ""
        return f"{indent}[{self.node_type}]{name_str}\n"

# --- Definitions ---
@dataclass
class ConstantDefinitionConfig(PyASTNode):
    """Represents a constant definition (e.g., `Real ep;`)."""
    name: str
    type: str # e.g., 'Real', 'Bool'
    value: Optional[str] = None # The initial value if assigned, stored as string

    def __repr__(self, level=0):
        indent = "  " * level
        value_str = f" = '{self.value}'" if self.value is not None else ""
        return f"{indent}[ConstantDefinition]({self.name}): Type='{self.type}'{value_str}\n"

@dataclass
class FunctionDefinitionConfig(PyASTNode):
    """Represents a function definition (e.g., `Real stopDist(Real v) = 0;`)."""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]] # List of (type, name) tuples
    body: str # Raw string of the function body

    def __repr__(self, level=0):
        indent = "  " * level
        params_str = ", ".join([f"{t} {n}" for t, n in self.parameters])
        return f"{indent}[FunctionDefinition]({self.name}): Returns='{self.return_type}', Params=({params_str}), Body='{self.body}'\n"

@dataclass
class PredicateDefinitionConfig(PyASTNode):
    """Represents a predicate definition (e.g., `Bool isWellformedDir(Real dx, Real dy) <-> dx^2 + dy^2 = 1;`)."""
    name: str
    parameters: List[Tuple[str, str]] # List of (type, name) tuples
    body: str # Raw string of the predicate body

    def __repr__(self, level=0):
        indent = "  " * level
        params_str = ", ".join([f"{t} {n}" for t, n in self.parameters])
        return f"{indent}[PredicateDefinition]({self.name}): Params=({params_str}), Body='{self.body}'\n"

# --- Action Space Specific Nodes (for controlFunc) ---
@dataclass
class NondeterministicAssignmentActionConfig(PyASTNode):
    """Represents a nondeterministic assignment (e.g., `vxo := *`)."""
    variable: str

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[NondeterministicAssignment]: {self.variable} := *\n"

@dataclass
class DeterministicAssignmentActionConfig(PyASTNode):
    """Represents a deterministic assignment (e.g., `t := 0`)."""
    variable: str
    value: str

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[DeterministicAssignment]: {self.variable} := '{self.value}'\n"

@dataclass
class ActionConstraintConfig(PyASTNode):
    """Represents an action constraint (e.g., `?vxo^2+vyo^2<=V^2;`)."""
    formula: str

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[ActionConstraint]: '{self.formula}'\n"

@dataclass
class CallHybridProgramConfig(PyASTNode):
    """Represents a call to another hybrid program (e.g., `controlFunc;`)."""
    hp_name: str

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[CallHybridProgram]: {self.hp_name}\n"

# Actions can be a mix of assignments, constraints, or nested sequences/choices
ActionType = Union[
    NondeterministicAssignmentActionConfig,
    DeterministicAssignmentActionConfig,
    ActionConstraintConfig,
    "SequenceActionConfig", # Forward reference for recursive types
    "ChoiceActionConfig",   # Forward reference
    CallHybridProgramConfig
]

@dataclass
class SequenceActionConfig(PyASTNode):
    """Represents a sequence of actions (e.g., `{a; b; c;}`)."""
    actions: List[ActionType] = field(default_factory=list)

    def __repr__(self, level=0):
        s = f"{'  ' * level}[SequenceAction]\n"
        for act in self.actions:
            s += act.__repr__(level + 1)
        return s

@dataclass
class ChoiceActionConfig(PyASTNode):
    """Represents a choice between different sequences of actions (e.g., `{a;} ++ {b;}`)."""
    level_name: str # e.g., 'ChoiceLevel_0' for debugging/identification
    choices: List[SequenceActionConfig] = field(default_factory=list) # Each choice branch is a sequence of actions

    def __repr__(self, level=0):
        s = f"{'  ' * level}[ChoiceAction]({self.level_name})\n"
        for choice_seq in self.choices:
            s += choice_seq.__repr__(level + 1)
        return s

@dataclass
class HybridProgramConfig(PyASTNode):
    """Base class for hybrid program definitions."""
    name: str
    body_raw: str # The raw KeYmaera X body string for reference

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[HybridProgram]({self.name})\n"
        # Truncate raw body for display to avoid overly long output
        truncated_body = self.body_raw.replace('\n', ' ').strip()
        s += f"{indent}  Body (raw): '{truncated_body[:80]}{'...' if len(truncated_body) > 80 else ''}'\n"
        return s

@dataclass
class ControlFuncConfig(HybridProgramConfig):
    """Represents the `controlFunc` hybrid program, defining the discrete action space."""
    action_structure: Optional[ActionType] = field(default=None) # Root of the action tree (Choice or Sequence)

    def __repr__(self, level=0):
        s = super().__repr__(level)
        if self.action_structure:
            s += self.action_structure.__repr__(level + 1)
        return s

# --- Step Function (Environment Dynamics) ---
@dataclass
class ODEConfig(PyASTNode):
    """Represents a single ordinary differential equation (e.g., `x' = v * dx`)."""
    variable: str # The variable being differentiated (e.g., 'x')
    equation: str # The expression for its derivative (e.g., 'v * dx')

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[ODE]: '{self.variable}\' = {self.equation}'\n"

@dataclass
class EvolutionDomainConfig(PyASTNode):
    """Represents the domain constraints for the continuous evolution (e.g., `t <= ep`)."""
    constraints: List[str] = field(default_factory=list) # List of constraint strings

    def __repr__(self, level=0):
        s = f"{'  ' * level}[EvolutionDomain]\n"
        for constraint in self.constraints:
            s += f"{'  ' * (level + 1)}[Constraint]: '{constraint}'\n"
        return s

@dataclass
class StepFuncConfig(HybridProgramConfig):
    """Represents the `stepFunc` hybrid program, defining the continuous dynamics."""
    odes: List[ODEConfig] = field(default_factory=list)
    evolution_domain: Optional[EvolutionDomainConfig] = field(default=None)
    duration: Optional[str] = None # Max duration of ODE integration, if specified in the original problem

    def __repr__(self, level=0):
        s = super().__repr__(level)
        if self.odes:
            s += f"{'  ' * (level + 1)}[ODEs]:\n"
            for ode in self.odes: s += ode.__repr__(level + 2)
        if self.evolution_domain:
            s += self.evolution_domain.__repr__(level + 1)
        return s

@dataclass
class DefinitionsConfig(PyASTNode):
    """Aggregates all definitions (constants, functions, predicates, hybrid programs)."""
    constants: List[ConstantDefinitionConfig] = field(default_factory=list)
    functions: List[FunctionDefinitionConfig] = field(default_factory=list)
    predicates: List[PredicateDefinitionConfig] = field(default_factory=list)
    hybrid_programs: List[Union[ControlFuncConfig, StepFuncConfig]] = field(default_factory=list)

    def __repr__(self, level=0):
        s = f"{'  ' * level}[DefinitionsConfig]\n"
        for c in self.constants: s += c.__repr__(level + 1)
        for f in self.functions: s += f.__repr__(level + 1)
        for p in self.predicates: s += p.__repr__(level + 1)
        for hp in self.hybrid_programs: s += hp.__repr__(level + 1)
        return s

# --- Variables ---
@dataclass
class VariableConfig(PyASTNode):
    """Represents a variable declaration (e.g., `Real x;`)."""
    name: str
    type: str # e.g., 'Real', 'Bool'
    lower_bound: Optional[str] = None # Optional bounds, stored as string
    upper_bound: Optional[str] = None

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[VariableDeclaration]({self.name}): Type='{self.type}'\n"

@dataclass
class VariablesConfig(PyASTNode):
    """Aggregates all variable declarations."""
    # Reverted to explicit state_variables and auxiliary_variables for compatibility
    # with the existing gym_generator.py logic.
    state_variables: List[VariableConfig] = field(default_factory=list)
    auxiliary_variables: List[VariableConfig] = field(default_factory=list)

    def __repr__(self, level=0):
        s = f"{'  ' * level}[VariablesConfig]\n"
        if self.state_variables:
            s += f"{'  ' * (level + 1)}[State Variables]:\n"
            for var in self.state_variables: s += var.__repr__(level + 2)
        if self.auxiliary_variables:
            s += f"{'  ' * (level + 1)}[Auxiliary Variables]:\n"
            for var in self.auxiliary_variables: s += var.__repr__(level + 2)
        return s

# --- Problem Configuration ---
@dataclass
class PreconditionConfig(PyASTNode):
    """Represents the precondition of the theorem (P)."""
    formula: str

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[Precondition]: '{self.formula}'\n"

@dataclass
class PostconditionConfig(PyASTNode):
    """Represents the postcondition of the theorem (Q)."""
    formula: str

    def __repr__(self, level=0):
        indent = "  " * level
        return f"{indent}[Postcondition]: '{self.formula}'\n"

@dataclass
class HybridProgramStepConfig(PyASTNode):
    """Represents a single step in a hybrid program, often used for continuous evolution."""
    hp_name: str # Name of the hybrid program being executed (e.g., 'stepFunc')
    invariant_formula: Optional[str] = None # Invariant for differential invariants/constraints

    def __repr__(self, level=0):
        indent = "  " * level
        inv_str = f", Invariant='{self.invariant_formula}'" if self.invariant_formula else ""
        return f"{indent}[HybridProgramStep]({self.hp_name}){inv_str}\n"

@dataclass
class LoopExecutionConfig(PyASTNode):
    """Represents the main loop structure of the problem (e.g., `{...}*@invariant(...)`)."""
    iterations: str = "inf" # Default to 'inf' for continuous loops
    loop_invariant_formula: Optional[str] = None
    loop_body: SequenceActionConfig = field(default_factory=SequenceActionConfig) # The sequence of actions within the loop

    def __repr__(self, level=0):
        s = f"{'  ' * level}[LoopExecution]\n"
        s += f"{'  ' * (level + 1)}[Iterations]: '{self.iterations}'\n"
        if self.loop_invariant_formula:
            s += f"{'  ' * (level + 1)}[LoopInvariant]: '{self.loop_invariant_formula}'\n"
        s += f"{'  ' * (level + 1)}[LoopBody]:\n"
        s += self.loop_body.__repr__(level + 2)
        return s

@dataclass
class ProblemConfig(PyASTNode):
    """Aggregates the precondition, loop execution, and postcondition."""
    precondition: PreconditionConfig
    loop_execution: LoopExecutionConfig
    postcondition: PostconditionConfig

    def __repr__(self, level=0):
        s = f"{'  ' * level}[ProblemConfig]\n"
        s += self.precondition.__repr__(level + 1)
        s += self.loop_execution.__repr__(level + 1)
        s += self.postcondition.__repr__(level + 1)
        return s

# --- Root Environment Config ---
@dataclass
class GymEnvironmentConfig(PyASTNode):
    """The top-level configuration for the entire Gym-like environment."""
    env_name: str
    definitions: DefinitionsConfig
    variables: VariablesConfig
    problem: ProblemConfig

    def __repr__(self, level=0):
        s = f"{'  ' * level}[GymEnvironmentConfig]({self.env_name})\n"
        s += self.definitions.__repr__(level + 1)
        s += self.variables.__repr__(level + 1)
        s += self.problem.__repr__(level + 1)
        return s