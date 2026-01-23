#ast_config.py
#this file provide the building block for PyAST
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure logging for ast_config.py
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logs, INFO for less verbose

# Base AST Node for configuration
class ASTConfigNode:
    """Base class for all AST configuration nodes."""
    def __repr__(self):
        # Default __repr__ calls _repr_indent with level 0
        return self._repr_indent(0)

    def _repr_indent(self, level=0):
        """Generates a readable, indented representation of the config node."""
        indent = "  " * level
        return f"{indent}[{self.__class__.__name__}]" # Default, to be overridden if needed

# --- Main Environment Configuration ---
class GymEnvironmentConfig(ASTConfigNode):
    def __init__(self, env_name: str,
                 definitions: Optional['DefinitionsConfig'] = None,
                 variables: Optional['VariablesConfig'] = None,
                 problem: Optional['ProblemConfig'] = None):
        self.env_name = env_name
        self.definitions = definitions if definitions is not None else DefinitionsConfig()
        self.variables = variables if variables is not None else VariablesConfig()
        self.problem = problem if problem is not None else ProblemConfig()

    def _repr_indent(self, level=0):
        indent = "  " * level
        return (f"{indent}[GymEnvironmentConfig]({self.env_name})\n"
                f"{self.definitions._repr_indent(level + 1)}\n"
                f"{self.variables._repr_indent(level + 1)}\n"
                f"{self.problem._repr_indent(level + 1)}")

# --- Definitions Block Configuration ---
class DefinitionsConfig(ASTConfigNode):
    def __init__(self, constants: Optional[List['ConstantDefinitionConfig']] = None,
                 functions: Optional[List['FunctionDefinitionConfig']] = None,
                 predicates: Optional[List['PredicateDefinitionConfig']] = None,
                 hybrid_programs: Optional[List['HybridProgramConfig']] = None):
        self.constants = constants if constants is not None else []
        self.functions = functions if functions is not None else []
        self.predicates = predicates if predicates is not None else []
        self.hybrid_programs = hybrid_programs if hybrid_programs is not None else [] # Can contain ControlFuncConfig, StepFuncConfig

    def _repr_indent(self, level=0):
        indent = "  " * level
        constants_str = "\n".join([c._repr_indent(level + 4) for c in self.constants])
        functions_str = "\n".join([f._repr_indent(level + 4) for f in self.functions])
        predicates_str = "\n".join([p._repr_indent(level + 4) for p in self.predicates])
        hybrid_programs_str = "\n".join([hp._repr_indent(level + 4) for hp in self.hybrid_programs])

        return (f"{indent}[DefinitionsConfig]\n"
                f"{indent}  [Constants]:\n{constants_str or f'{indent}    (None)'}\n"
                f"{indent}  [Functions]:\n{functions_str or f'{indent}    (None)'}\n"
                f"{indent}  [Predicates]:\n{predicates_str or f'{indent}    (None)'}\n"
                f"{indent}  [Hybrid Programs]:\n{hybrid_programs_str or f'{indent}    (None)'}")

class ConstantDefinitionConfig(ASTConfigNode):
    def __init__(self, name: str, type: str, value: Optional[str] = None):
        self.name = name
        self.type = type
        self.value = value

    def _repr_indent(self, level=0):
        indent = "  " * level
        value_str = f", Value='{self.value}'" if self.value is not None else ""
        return f"{indent}[ConstantDefinition]({self.name}): Type='{self.type}'{value_str}"

class FunctionDefinitionConfig(ASTConfigNode):
    def __init__(self, name: str, return_type: str, parameters: List[Tuple[str, str]], body: str):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters # List of (type, name) tuples
        self.body = body

    def _repr_indent(self, level=0):
        indent = "  " * level
        params_str = ", ".join([f"{t} {n}" for t, n in self.parameters])
        body_display = self.body
        if len(body_display) > 50:
            body_display = body_display[:47] + "..."
        return f"{indent}[FunctionDefinition]({self.name}): Returns='{self.return_type}', Params=({params_str}), Body='{body_display}'"

class PredicateDefinitionConfig(ASTConfigNode):
    def __init__(self, name: str, parameters: List[Tuple[str, str]], body: str):
        self.name = name
        self.parameters = parameters # List of (type, name) tuples
        self.body = body

    def _repr_indent(self, level=0):
        indent = "  " * level
        params_str = ", ".join([f"{t} {n}" for t, n in self.parameters])
        body_display = self.body
        if len(body_display) > 50:
            body_display = body_display[:47] + "..."
        return f"{indent}[PredicateDefinition]({self.name}): Params=({params_str}), Body='{body_display}'"

class HybridProgramConfig(ASTConfigNode):
    """Base class for specific hybrid program configurations."""
    def __init__(self, name: str):
        self.name = name

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[HybridProgram]({self.name})"

class ControlFuncConfig(HybridProgramConfig):
    """Configuration for a control function (discrete choices and assignments)."""
    def __init__(self, name: str, action_space_config: Optional['ActionConfig'] = None):
        super().__init__(name)
        self.action_space_config = action_space_config

    def _repr_indent(self, level=0):
        indent = "  " * level
        if self.action_space_config:
            raw_body_snippet = self.action_space_config.to_string_representation()
            if len(raw_body_snippet) > 80:
                raw_body_snippet = raw_body_snippet[:77] + "..."
            return (f"{indent}[HybridProgram]({self.name})\n"
                    f"{indent}  [Body (parsed)]: '{raw_body_snippet}'\n"
                    f"{self.action_space_config._repr_indent(level + 1)}")
        return f"{indent}[HybridProgram]({self.name}): No action space defined."

class ODEConfig(ASTConfigNode):
    def __init__(self, variable: str, equation: str):
        self.variable = variable
        self.equation = equation

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[ODEConfig(var='{self.variable}', eq='{self.equation}')]"

class EvolutionDomainConfig(ASTConfigNode):
    def __init__(self, constraints: Optional[List[str]] = None):
        self.constraints = constraints if constraints is not None else []

    def _repr_indent(self, level=0):
        indent = "  " * level
        constraints_str = "\n".join([f"{indent}  [Constraint]: '{c}'" for c in self.constraints])
        return f"{indent}[Evolution Domain]:\n{constraints_str or f'{indent}  (None)'}"

class StepFuncConfig(HybridProgramConfig):
    """Configuration for a step function (differential equations)."""
    def __init__(self, name: str, odes: List[ODEConfig], evolution_domain: Optional[EvolutionDomainConfig] = None, duration: Optional[str] = None):
        super().__init__(name)
        self.odes = odes
        self.evolution_domain = evolution_domain
        self.duration = duration # e.g., 'ep' for the t' = 1 & t <= ep part

    def _repr_indent(self, level=0):
        indent = "  " * level
        odes_lines = [ode._repr_indent(level + 2) for ode in self.odes]
        duration_str = f", Duration='{self.duration}'" if self.duration else ""

        ode_block_str = "\n".join(odes_lines) if odes_lines else f"{indent}  (None)"
        domain_block_str = self.evolution_domain._repr_indent(level + 1) if self.evolution_domain else f"{indent}  [Evolution Domain]:\n{indent}    (None)"

        return (f"{indent}[HybridProgram]({self.name}){duration_str}\n"
                f"{indent}  [ODEs]:\n{ode_block_str}\n"
                f"{domain_block_str}")


# --- Variables Block Configuration ---
class VariablesConfig(ASTConfigNode):
    def __init__(self, state_variables: Optional[List['VariableDeclarationConfig']] = None,
                 auxiliary_variables: Optional[List['VariableDeclarationConfig']] = None):
        self.state_variables = state_variables if state_variables is not None else []
        self.auxiliary_variables = auxiliary_variables if auxiliary_variables is not None else [] # Variables not part of observation space

    def _repr_indent(self, level=0):
        indent = "  " * level
        state_vars_str = "\n".join([v._repr_indent(level + 4) for v in self.state_variables])
        aux_vars_str = "\n".join([v._repr_indent(level + 4) for v in self.auxiliary_variables])
        return (f"{indent}[VariablesConfig]\n"
                f"{indent}  [State Variables]:\n{state_vars_str or f'{indent}    (None)'}\n"
                f"{indent}  [Auxiliary Variables]:\n{aux_vars_str or f'{indent}    (None)'}")

class VariableDeclarationConfig(ASTConfigNode):
    def __init__(self, name: str, type: str, lower_bound: Optional[str] = None, upper_bound: Optional[str] = None):
        super().__init__()
        self.name = name
        self.type = type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _repr_indent(self, level=0):
        indent = "  " * level
        bounds_str = ""
        if self.lower_bound is not None or self.upper_bound is not None:
            bounds_str = f", Bounds=({self.lower_bound or '-inf'}, {self.upper_bound or 'inf'})"
        return f"{indent}[VariableDeclaration]({self.name}): Type='{self.type}'{bounds_str}"

# --- Problem Statement Configuration ---
class PreconditionConfig(ASTConfigNode):
    def __init__(self, formula: str):
        self.formula = formula
    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}Precondition(formula='{self.formula}')"

class PostconditionConfig(ASTConfigNode):
    def __init__(self, formula: str):
        self.formula = formula
    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}Postcondition(formula='{self.formula}')"

class ProblemConfig(ASTConfigNode):
    def __init__(self, precondition: Optional[PreconditionConfig] = None,
                 loop_execution: Optional['LoopExecutionConfig'] = None,
                 postcondition: Optional[PostconditionConfig] = None):
        self.precondition = precondition if precondition is not None else PreconditionConfig(formula="")
        self.loop_execution = loop_execution if loop_execution is not None else LoopExecutionConfig()
        self.postcondition = postcondition if postcondition is not None else PostconditionConfig(formula="")

    def _repr_indent(self, level=0):
        indent = "  " * level
        pre_str = f"{self.precondition._repr_indent(level + 1)}" if self.precondition else ""
        loop_str = f"\n{self.loop_execution._repr_indent(level + 1)}" if self.loop_execution else ""
        post_str = f"\n{self.postcondition._repr_indent(level + 1)}" if self.postcondition else ""
        return (f"{indent}[ProblemConfig]\n"
                f"{indent}  [Precondition]: '{self.precondition.formula}'" # Reverted to match example format
                f"{loop_str}{post_str}")

class LoopExecutionConfig(ASTConfigNode):
    def __init__(self, iterations: str = "inf", loop_invariant_formula: str = "", loop_body: 'SequenceActionConfig' = None):
        self.iterations = iterations
        self.loop_invariant_formula = loop_invariant_formula
        self.loop_body = loop_body if loop_body is not None else SequenceActionConfig() # This will be a SequenceActionConfig containing discrete and continuous parts

    def _repr_indent(self, level=0):
        indent = "  " * level
        return (f"{indent}[LoopExecution]\n"
                f"{indent}  [Iterations]: '{self.iterations}'\n"
                f"{indent}  [LoopInvariant]: '{self.loop_invariant_formula}'\n"
                f"{indent}  [LoopBody]:\n{self.loop_body._repr_indent(level + 2)}")

class HybridProgramStepConfig(ASTConfigNode):
    """Represents a call to a hybrid program within a loop, possibly with an invariant."""
    def __init__(self, hp_name: str, invariant_formula: Optional[str] = None):
        self.hp_name = hp_name
        self.invariant_formula = invariant_formula

    def _repr_indent(self, level=0):
        indent = "  " * level
        invariant_str = f"@invariant({self.invariant_formula})" if self.invariant_formula else ""
        return f"{indent}[HybridProgramStep]: {self.hp_name}{invariant_str}"


# --- Action Space Configuration (for controlFunc and problem body) ---
class ActionConfig(ASTConfigNode):
    """Base class for action configurations."""
    def to_string_representation(self) -> str:
        """Returns a string representation suitable for raw body display."""
        raise NotImplementedError

class ChoiceActionConfig(ActionConfig):
    def __init__(self, level_name: str, choices: List[ActionConfig]):
        self.level_name = level_name # e.g., 'ChoiceLevel_0'
        self.choices = choices

    def _repr_indent(self, level=0):
        indent = "  " * level
        choices_str_list = []
        for i, choice in enumerate(self.choices):
            choices_str_list.append(f"{indent}  [Choice {i+1}]:\n{choice._repr_indent(level + 2)}")
        return f"{indent}[ChoiceAction]({self.level_name})\n" + "\n".join(choices_str_list)

    def to_string_representation(self) -> str:
        return " ++ ".join([choice.to_string_representation() for choice in self.choices])

class SequenceActionConfig(ActionConfig):
    def __init__(self, actions: Optional[List[ActionConfig]] = None):
        self.actions = actions if actions is not None else []

    def _repr_indent(self, level=0):
        indent = "  " * level
        actions_str = "\n".join([action._repr_indent(level + 1) for action in self.actions])
        return f"{indent}[SequenceAction]\n{actions_str or f'{indent}  (None)'}"

    def to_string_representation(self) -> str:
        return "; ".join([action.to_string_representation() for action in self.actions])

class DeterministicAssignmentActionConfig(ActionConfig):
    def __init__(self, variable: str, value: str):
        self.variable = variable
        self.value = value

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[DeterministicAssignment]: {self.variable} := '{self.value}'"

    def to_string_representation(self) -> str:
        return f"{self.variable} := {self.value}"

class NondeterministicAssignmentActionConfig(ActionConfig):
    def __init__(self, variable: str):
        self.variable = variable

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[NondeterministicAssignment]: {self.variable} := *"

    def to_string_representation(self) -> str:
        return f"{self.variable} := *"

class ActionConstraintConfig(ActionConfig):
    def __init__(self, formula: str):
        self.formula = formula

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[ActionConstraint]: '{self.formula}'"

    def to_string_representation(self) -> str:
        return f"?{self.formula}"

class CallHybridProgramConfig(ActionConfig):
    def __init__(self, hp_name: str):
        self.hp_name = hp_name

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[CallHybridProgram]: {self.hp_name}"

    def to_string_representation(self) -> str:
        return self.hp_name

# Define ActionType for type hinting
ActionType = Union[
    ChoiceActionConfig,
    SequenceActionConfig,
    DeterministicAssignmentActionConfig,
    NondeterministicAssignmentActionConfig,
    ActionConstraintConfig,
    CallHybridProgramConfig,
    HybridProgramStepConfig # Added for loop body
]