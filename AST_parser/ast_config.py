import logging
from typing import List, Dict, Any, Optional, Union

# Configure logging for ast_config.py
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logs, INFO for less verbose

# Base AST Node for configuration
class ASTConfigNode:
    """Base class for all AST configuration nodes."""
    def __repr__(self):
        return f"[{self.__class__.__name__}]"

# --- Main Environment Configuration ---
class GymEnvironmentConfig(ASTConfigNode):
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.definitions = DefinitionsConfig()
        self.variables = VariablesConfig()
        self.problem = ProblemConfig()

    def __repr__(self):
        return f"[GymEnvironmentConfig]({self.env_name})\n  {self.definitions}\n  {self.variables}\n  {self.problem}"

# --- Definitions Block Configuration ---
class DefinitionsConfig(ASTConfigNode):
    def __init__(self):
        self.constants: List[ConstantDefinitionConfig] = []
        self.functions: List[FunctionDefinitionConfig] = []
        self.predicates: List[PredicateDefinitionConfig] = []
        self.hybrid_programs: List[HybridProgramConfig] = [] # Can contain ControlFuncConfig, StepFuncConfig

    def __repr__(self):
        return (f"[DefinitionsConfig]\n"
                f"    Constants: {len(self.constants)}\n"
                f"    Functions: {len(self.functions)}\n"
                f"    Predicates: {len(self.predicates)}\n"
                f"    Hybrid Programs: {[hp.name for hp in self.hybrid_programs]}")

class ConstantDefinitionConfig(ASTConfigNode):
    def __init__(self, name: str, kx_type: str):
        self.name = name
        self.kx_type = kx_type

    def __repr__(self):
        return f"[ConstantDefinition]({self.name}): Type='{self.kx_type}'"

class FunctionDefinitionConfig(ASTConfigNode):
    def __init__(self, name: str, return_type: str, parameters: List[tuple], body: str):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters # List of (type, name) tuples
        self.body = body

    def __repr__(self):
        params_str = ", ".join([f"{t} {n}" for t, n in self.parameters])
        return f"[FunctionDefinition]({self.name}): Returns='{self.return_type}', Params=({params_str}), Body='{self.body}'"

class PredicateDefinitionConfig(ASTConfigNode):
    def __init__(self, name: str, parameters: List[tuple], body: str):
        self.name = name
        self.parameters = parameters # List of (type, name) tuples
        self.body = body

    def __repr__(self):
        params_str = ", ".join([f"{t} {n}" for t, n in self.parameters])
        return f"[PredicateDefinition]({self.name}): Params=({params_str}), Body='{self.body}'"

class HybridProgramConfig(ASTConfigNode):
    """Base class for specific hybrid program configurations."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"[HybridProgram]({self.name})"

class ControlFuncConfig(HybridProgramConfig):
    """Configuration for a control function (discrete choices and assignments)."""
    def __init__(self, name: str, action_space_config: 'ActionConfig'):
        super().__init__(name)
        self.action_space_config = action_space_config

    def __repr__(self):
        return f"[HybridProgram]({self.name})\n      Body (raw): '{self.action_space_config.to_string_representation()}'\n      {self.action_space_config}"

class StepFuncConfig(HybridProgramConfig):
    """Configuration for a step function (differential equations)."""
    def __init__(self, name: str, differential_equations: Dict[str, str], evolution_domain: Optional[str] = None):
        super().__init__(name)
        self.differential_equations = differential_equations # Maps variable name to expression string
        self.evolution_domain = evolution_domain # The guard for the differential equations

    def __repr__(self):
        equations_str = ", ".join([f"{var}' = {expr}" for var, expr in self.differential_equations.items()])
        domain_str = f" & {self.evolution_domain}" if self.evolution_domain else ""
        return f"[HybridProgram]({self.name}): '{equations_str}{domain_str}'"


# --- Variables Block Configuration ---
class VariablesConfig(ASTConfigNode):
    def __init__(self):
        self.state_variables: List[VariableDeclarationConfig] = []
        self.auxiliary_variables: List[VariableDeclarationConfig] = [] # Variables not part of observation space

    def __repr__(self):
        state_vars_str = "\n".join([f"      {v}" for v in self.state_variables])
        aux_vars_str = "\n".join([f"      {v}" for v in self.auxiliary_variables])
        return (f"[VariablesConfig]\n"
                f"    [State Variables]:\n{state_vars_str}\n"
                f"    [Auxiliary Variables]:\n{aux_vars_str}")

class VariableDeclarationConfig(ASTConfigNode):
    def __init__(self, name: str, kx_type: str):
        self.name = name
        self.kx_type = kx_type

    def __repr__(self):
        return f"[VariableDeclaration]({self.name}): Type='{self.kx_type}'"

# --- Problem Statement Configuration ---
class ProblemConfig(ASTConfigNode):
    def __init__(self):
        self.precondition: Optional[str] = None
        self.loop_execution: Optional[LoopExecutionConfig] = None
        self.postcondition: Optional[str] = None

    def __repr__(self):
        pre_str = f"    [Precondition]: '{self.precondition}'" if self.precondition else ""
        loop_str = f"\n    {self.loop_execution}" if self.loop_execution else ""
        post_str = f"\n    [Postcondition]: '{self.postcondition}'" if self.postcondition else ""
        return f"[ProblemConfig]{pre_str}{loop_str}{post_str}"

class LoopExecutionConfig(ASTConfigNode):
    def __init__(self, iterations: str, loop_invariant: str, loop_body: 'ActionConfig'):
        self.iterations = iterations
        self.loop_invariant = loop_invariant
        self.loop_body = loop_body # This will be a SequenceActionConfig containing discrete and continuous parts

    def __repr__(self):
        return (f"[LoopExecution]\n"
                f"      [Iterations]: '{self.iterations}'\n"
                f"      [LoopInvariant]: '{self.loop_invariant}'\n"
                f"      [LoopBody]:\n{self.loop_body}")

# --- Action Space Configuration (for controlFunc and problem body) ---
class ActionConfig(ASTConfigNode):
    """Base class for action configurations."""
    def to_string_representation(self) -> str:
        """Returns a string representation suitable for raw body display."""
        raise NotImplementedError

class ChoiceActionConfig(ActionConfig):
    def __init__(self, name: str, choices: List[ActionConfig]):
        self.name = name # e.g., 'choice_ChoiceLevel_0'
        self.choices = choices

    def __repr__(self):
        choices_str = "\n".join([f"        Choice {i+1}:\n{str(choice).replace('        ', '          ')}" for i, choice in enumerate(self.choices)])
        return f"[ChoiceAction]({self.name})\n{choices_str}"

    def to_string_representation(self) -> str:
        return " ++ ".join([choice.to_string_representation() for choice in self.choices])

class SequenceActionConfig(ActionConfig):
    def __init__(self, actions: Optional[List[ActionConfig]] = None):
        self.actions = actions if actions is not None else []

    def __repr__(self):
        actions_str = "\n".join([f"          {str(action)}" for action in self.actions])
        return f"[SequenceAction]\n{actions_str}"

    def to_string_representation(self) -> str:
        return "; ".join([action.to_string_representation() for action in self.actions])

class DeterministicAssignmentActionConfig(ActionConfig):
    def __init__(self, variable: str, expression: str):
        self.variable = variable
        self.expression = expression

    def __repr__(self):
        return f"[DeterministicAssignment]: {self.variable} := '{self.expression}'"

    def to_string_representation(self) -> str:
        return f"{self.variable} := {self.expression}"

class NondeterministicAssignmentActionConfig(ActionConfig):
    def __init__(self, variable: str):
        self.variable = variable

    def __repr__(self):
        return f"[NondeterministicAssignment]: {self.variable} := *"

    def to_string_representation(self) -> str:
        return f"{self.variable} := *"

class ActionConstraintConfig(ActionConfig):
    def __init__(self, constraint_expression: str):
        self.constraint_expression = constraint_expression

    def __repr__(self):
        return f"[ActionConstraint]: '{self.constraint_expression}'"

    def to_string_representation(self) -> str:
        return f"?{self.constraint_expression}"

class CallHybridProgramConfig(ActionConfig):
    def __init__(self, name: str, invariant: Optional[str] = None):
        self.name = name
        self.invariant = invariant

    def __repr__(self):
        invariant_str = f"@invariant({self.invariant})" if self.invariant else ""
        return f"[CallHybridProgram]: {self.name}{invariant_str}"

    def to_string_representation(self) -> str:
        invariant_str = f"@invariant({self.invariant})" if self.invariant else ""
        return f"{self.name}{invariant_str}"