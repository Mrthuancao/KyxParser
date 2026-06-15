#pyast_config.py

import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

# Configure logging for ast_config.py
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logs, INFO for less verbose

# Helper function for splitting by operator outside parentheses
def _split_by_operator_outside_parens(text: str, operator: str) -> List[str]:
    parts = []
    paren_level = 0
    current_part_start = 0
    op_len = len(operator)

    i = 0
    while i < len(text):
        if text[i] == '(':
            paren_level += 1
        elif text[i] == ')':
            paren_level -= 1
        
        # Check for operator match
        if (i + op_len <= len(text) and
            paren_level == 0 and
            text[i:i+op_len] == operator):
            
            part = text[current_part_start:i].strip()
            if part:
                parts.append(part)
            current_part_start = i + op_len
            i += op_len - 1 # Adjust i to continue after the operator
        i += 1
    
    # Add the last part
    last_part = text[current_part_start:].strip()
    if last_part:
        parts.append(last_part)
    
    return [p for p in parts if p] # Filter out empty strings

# --- Formatting Helpers (Extracted from _repr_indent) ---
def _format_multiline_formula(formula: str, indent_base: str, initial_prefix: str, subsequent_prefix: str, max_len: int = 80) -> str:
    """Formats a formula with logical operators for multi-line display and truncation."""
    body_display = formula.strip()
    if not body_display:
        return "''"

    if '&' in body_display or '|' in body_display:
        and_parts = _split_by_operator_outside_parens(body_display, '&')
        
        formatted_body_parts = []
        for i, part in enumerate(and_parts):
            if i > 0:
                formatted_body_parts.append(f"\n{indent_base}{subsequent_prefix}& ")
            
            or_parts = _split_by_operator_outside_parens(part, '|')
            if len(or_parts) > 1:
                formatted_sub_part = or_parts[0]
                for sub_part_idx in range(1, len(or_parts)):
                    # Ensure we don't add an extra '|' if the split result already includes it
                    if or_parts[sub_part_idx].startswith('|'):
                        formatted_sub_part += f"\n{indent_base}{subsequent_prefix}  " + or_parts[sub_part_idx]
                    else:
                        formatted_sub_part += f"\n{indent_base}{subsequent_prefix}  | " + or_parts[sub_part_idx]
                formatted_body_parts.append(formatted_sub_part)
            else:
                formatted_body_parts.append(part)
        
        body_display = "".join(formatted_body_parts).strip()
    
    single_line_body_for_len = body_display.replace('\n', ' ').strip()
    if len(single_line_body_for_len) > max_len:
        body_display = single_line_body_for_len[:max_len - 3] + "..."

    return body_display

def _format_invariant_formula(formula: str, indent_base: str) -> str:
    """Formats an invariant formula for multi-line display."""
    if not formula:
        return ""
    
    parts = _split_by_operator_outside_parens(formula, ',')
    if len(parts) > 1:
        formatted_invariant = f"@invariant({parts[0]}"
        # 10 spaces for alignment after indent, assuming indent_base is "  " * level
        # This needs to be relative to the current line's start, not just the base indent
        current_indent_len = len(indent_base) + len("[HybridProgramStep]: ") # Assuming this prefix
        align_spaces = " " * (current_indent_len + len("@invariant(") - 1) # -1 for the opening paren
        for part in parts[1:]:
            formatted_invariant += f",\n{align_spaces}{part}"
        formatted_invariant += ")"
        return formatted_invariant
    else:
        return f"@invariant({formula})"


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
    
    def _get_serialization_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of constructor parameters and their values for serialization.
        Subclasses should override this.
        """
        # Default implementation, useful for classes with no __init__ args or simple ones
        # For more complex classes, this needs to be explicitly overridden.
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


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
    
    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "env_name": self.env_name,
            "definitions": self.definitions,
            "variables": self.variables,
            "problem": self.problem,
        }


# --- Definitions Block Configuration ---
class DefinitionsConfig(ASTConfigNode):
    def __init__(self, constants: Optional[List['ConstantDefinitionConfig']] = None,
                 functions: Optional[List['FunctionDefinitionConfig']] = None,
                 predicates: Optional[List['PredicateDefinitionConfig']] = None,
                 hybrid_programs: Optional[List['HybridProgramConfig']] = None):
        self.constants = constants or []
        self.functions = functions or []
        self.predicates = predicates or []
        self.hybrid_programs = hybrid_programs or []

    def _repr_indent(self, level=0):
        indent = "  " * level
        # Standardize indentation to level + 2 for items within lists
        constants_str = "\n".join([c._repr_indent(level + 2) for c in self.constants])
        functions_str = "\n".join([f._repr_indent(level + 2) for f in self.functions])
        predicates_str = "\n".join([p._repr_indent(level + 2) for p in self.predicates])
        hybrid_programs_str = "\n".join([hp._repr_indent(level + 2) for hp in self.hybrid_programs])

        return (f"{indent}[DefinitionsConfig]\n"
                f"{indent}  [Constants]:\n{constants_str or f'{indent}    (None)'}\n"
                f"{indent}  [Functions]:\n{functions_str or f'{indent}    (None)'}\n"
                f"{indent}  [Predicates]:\n{predicates_str or f'{indent}    (None)'}\n"
                f"{indent}  [Hybrid Programs]:\n{hybrid_programs_str or f'{indent}    (None)'}")

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "constants": self.constants,
            "functions": self.functions,
            "predicates": self.predicates,
            "hybrid_programs": self.hybrid_programs,
        }

class ConstantDefinitionConfig(ASTConfigNode):
    def __init__(self, name: str, type: str, value: Optional[str] = None, metadata: Optional[Dict] = None):
        self.name = name
        self.type = type
        self.value = value
        self.metadata = metadata or {}

    def _repr_indent(self, level=0):
        indent = "  " * level
        meta = f", Meta={self.metadata}" if self.metadata else ""
        return f"{indent}[ConstantDefinition]({self.name}): Type='{self.type}', Value='{self.value}'{meta}"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "value": self.value,
            "metadata": self.metadata,
        }

class FunctionDefinitionConfig(ASTConfigNode):
    def __init__(self, name: str, return_type: str, parameters: List[Tuple[str, str]], body: str, metadata: Optional[Dict] = None):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters # List of (type, name) tuples
        self.body = body
        self.metadata = metadata or {}

    def _repr_indent(self, level=0):
        indent = "  " * level
        params = ", ".join([f"{t} {n}" for t, n in self.parameters])
        meta = f", Meta={self.metadata}" if self.metadata else ""
        return f"{indent}[FunctionDefinition]({self.name}): Returns='{self.return_type}', Params=({params}), Body='{self.body}'{meta}"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "body": self.body,
            "metadata": self.metadata,
        }

class PredicateDefinitionConfig(ASTConfigNode):
    def __init__(self, name: str, parameters: List[Tuple[str, str]], body: str, metadata: Optional[Dict] = None):
        self.name = name
        self.parameters = parameters # List of (type, name) tuples
        self.body = body
        self.metadata = metadata or {}

    def _get_formatted_body_display(self, level: int) -> str:
        """Helper to format the predicate body for display."""
        indent_base = "  " * level
        # The subsequent prefix needs to account for the fixed string part before the body
        # e.g., "[PredicateDefinition](name): Params=(): Body='"
        # Let's approximate a good alignment for now, or refine if needed.
        # For simplicity, using a fixed offset relative to the current indent.
        return _format_multiline_formula(self.body, indent_base, "    ", "    ", max_len=77)

    def _repr_indent(self, level=0):
        indent = "  " * level
        params_str = ", ".join([f"{t} {n}" for t, n in self.parameters])
        meta = f", Meta={self.metadata}" if self.metadata else ""
        
        body_display = self._get_formatted_body_display(level)

        return f"{indent}[PredicateDefinition]({self.name}): Params=({params_str}), Body='{body_display}'{meta}"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "body": self.body,
            "metadata": self.metadata,
        }

class HybridProgramConfig(ASTConfigNode):
    """Base class for specific hybrid program configurations."""
    def __init__(self, name: str):
        self.name = name

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[HybridProgram]({self.name})"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {"name": self.name}


class ControlFuncConfig(HybridProgramConfig):
    """Configuration for a control function (discrete choices and assignments)."""
    def __init__(self, name: str, action_space_config: Optional['ActionConfig'] = None, metadata: Optional[Dict] = None):
        super().__init__(name)
        self.action_space_config = action_space_config
        self.metadata = metadata or {}

    def _get_formatted_action_space_body(self) -> str:
        if self.action_space_config:
            raw_body_snippet = self.action_space_config.to_string_representation()
            if len(raw_body_snippet) > 80:
                raw_body_snippet = raw_body_snippet[:77] + "..."
            return raw_body_snippet
        return ""

    def _repr_indent(self, level=0):
        indent = "  " * level
        meta = f" [Meta: {self.metadata}]" if self.metadata else ""

        if self.action_space_config:
            body_display = self._get_formatted_action_space_body()
            return (f"{indent}[HybridProgram]({self.name}){meta}\n"
                    f"{indent}  [Body (parsed)]: '{body_display}'\n"
                    f"{self.action_space_config._repr_indent(level + 1)}")
        return f"{indent}[HybridProgram]({self.name}): No action space defined."

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "action_space_config": self.action_space_config,
            "metadata": self.metadata,
        }

class ODEConfig(ASTConfigNode):
    def __init__(self, variable: str, equation: str):
        self.variable = variable
        self.equation = equation

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[ODEConfig(var='{self.variable}', eq='{self.equation}')]"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "equation": self.equation,
        }

class EvolutionDomainConfig(ASTConfigNode):
    def __init__(self, constraints: Optional[List[str]] = None):
        self.constraints = constraints or []

    def _repr_indent(self, level=0):
        indent = "  " * level
        constraints_str = "\n".join([f"{indent}  [Constraint]: '{c}'" for c in self.constraints])
        return f"{indent}[Evolution Domain]:\n{constraints_str or f'{indent}    (None)'}"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "constraints": self.constraints,
        }

class StepFuncConfig(HybridProgramConfig):
    """Configuration for a step function (differential equations)."""
    def __init__(self, name: str, odes: List[ODEConfig], evolution_domain: Optional[EvolutionDomainConfig] = None, duration: Optional[str] = None, raw_body_snippet: Optional[str] = None, metadata: Optional[Dict] = None):
        super().__init__(name)
        self.odes = odes
        self.evolution_domain = evolution_domain
        self.duration = duration
        self.raw_body_snippet = raw_body_snippet
        self.metadata = metadata or {}

    def _get_formatted_raw_body_snippet(self) -> str:
        if self.raw_body_snippet:
            body_display = self.raw_body_snippet
            if len(body_display) > 80: # Truncate for display
                body_display = body_display[:77] + "..."
            return body_display
        return ""

    def _repr_indent(self, level=0):
        indent = "  " * level

        meta = f" [Meta: {self.metadata}]" if self.metadata else ""
        # Standardize indentation to level + 2 for items within lists
        odes_lines = [ode._repr_indent(level + 2) for ode in self.odes]
        duration_str = f", Duration='{self.duration}'" if self.duration else ""

        ode_block_str = "\n".join(odes_lines) if odes_lines else f"{indent}    (None)"
        domain_block_str = self.evolution_domain._repr_indent(level + 1) if self.evolution_domain else f"{indent}  [Evolution Domain]:\n{indent}    (None)"

        body_parsed_str = ""
        formatted_snippet = self._get_formatted_raw_body_snippet()
        if formatted_snippet:
            body_parsed_str = f"{indent}  [Body (parsed)]: '{formatted_snippet}'\n"

        return (f"{indent}[HybridProgram]({self.name}){duration_str}{meta}\n"
                f"{body_parsed_str}"
                f"{indent}  [ODEs]:\n{ode_block_str}\n"
                f"{domain_block_str}")

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "odes": self.odes,
            "evolution_domain": self.evolution_domain,
            "duration": self.duration,
            "raw_body_snippet": self.raw_body_snippet,
            "metadata": self.metadata,
        }


# --- Variables Block Configuration ---
class VariablesConfig(ASTConfigNode):
    def __init__(self, state_variables: Optional[List['VariableDeclarationConfig']] = None,
                 auxiliary_variables: Optional[List['VariableDeclarationConfig']] = None):
        self.state_variables = state_variables or []
        self.auxiliary_variables = auxiliary_variables or []

    def _repr_indent(self, level=0):
        indent = "  " * level
        # Standardize indentation to level + 2 for items within lists
        state_vars_str = "\n".join([v._repr_indent(level + 2) for v in self.state_variables])
        aux_vars_str = "\n".join([v._repr_indent(level + 2) for v in self.auxiliary_variables])
        return (f"{indent}[VariablesConfig]\n"
                f"{indent}  [State Variables]:\n{state_vars_str or f'{indent}    (None)'}\n"
                f"{indent}  [Auxiliary Variables]:\n{aux_vars_str or f'{indent}    (None)'}")

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "state_variables": self.state_variables,
            "auxiliary_variables": self.auxiliary_variables,
        }

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

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

# --- Problem Statement Configuration ---
class PreconditionConfig(ASTConfigNode):
    def __init__(self, formula: str):
        self.formula = formula
    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[Precondition]: '{self.formula}'" # Consistent with ProblemConfig's desired output

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {"formula": self.formula}

class PostconditionConfig(ASTConfigNode):
    def __init__(self, formula: str):
        self.formula = formula
    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[Postcondition]: '{self.formula}'" # Consistent with ProblemConfig's desired output

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {"formula": self.formula}

class ProblemConfig(ASTConfigNode):
    def __init__(self, precondition: Optional[PreconditionConfig] = None,
                 loop_execution: Optional['LoopExecutionConfig'] = None,
                 postcondition: Optional[PostconditionConfig] = None):
        self.precondition = precondition if precondition is not None else PreconditionConfig(formula="")
        self.loop_execution = loop_execution if loop_execution is not None else LoopExecutionConfig()
        self.postcondition = postcondition if postcondition is not None else PostconditionConfig(formula="")

    def _repr_indent(self, level=0):
        indent = "  " * level
        # Consistently call _repr_indent for all sub-nodes
        pre_str = self.precondition._repr_indent(level + 1)
        loop_str = self.loop_execution._repr_indent(level + 1)
        post_str = self.postcondition._repr_indent(level + 1)
        
        return (f"{indent}[ProblemConfig]\n"
                f"{pre_str}\n"
                f"{loop_str}\n"
                f"{post_str}")

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "precondition": self.precondition,
            "loop_execution": self.loop_execution,
            "postcondition": self.postcondition,
        }

class LoopExecutionConfig(ASTConfigNode):
    def __init__(self, iterations: str = "inf", loop_invariant_formula: str = "", loop_body: 'SequenceActionConfig' = None):
        self.iterations = iterations
        self.loop_invariant_formula = loop_invariant_formula
        self.loop_body = loop_body if loop_body is not None else SequenceActionConfig()

    def _repr_indent(self, level=0):
        indent = "  " * level
        return (f"{indent}[LoopExecution]\n"
                f"{indent}  [Iterations]: '{self.iterations}'\n"
                f"{indent}  [LoopInvariant]: '{self.loop_invariant_formula}'\n"
                f"{indent}  [LoopBody]:\n{self.loop_body._repr_indent(level + 2)}")

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "loop_invariant_formula": self.loop_invariant_formula,
            "loop_body": self.loop_body,
        }

class HybridProgramStepConfig(ASTConfigNode):
    """Represents a call to a hybrid program within a loop, possibly with an invariant."""
    def __init__(self, hp_name: str, invariant_formula: Optional[str] = None):
        self.hp_name = hp_name
        self.invariant_formula = invariant_formula

    def _get_formatted_invariant_display(self, level: int) -> str:
        """Helper to format the invariant formula for display."""
        indent_base = "  " * level
        # The prefix for the invariant starts after "[HybridProgramStep]: hp_name"
        # We need to calculate the alignment dynamically
        return _format_invariant_formula(self.invariant_formula, indent_base)

    def _repr_indent(self, level=0):
        indent = "  " * level
        invariant_str = self._get_formatted_invariant_display(level)
        if invariant_str:
            # Adjust alignment for multiline invariant
            return f"{indent}[HybridProgramStep]: {self.hp_name} {invariant_str}"
        return f"{indent}[HybridProgramStep]: {self.hp_name}"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "hp_name": self.hp_name,
            "invariant_formula": self.invariant_formula,
        }


# --- Action Space Configuration (for controlFunc and problem body) ---
class ActionConfig(ASTConfigNode):
    """Base class for action configurations."""
    def to_string_representation(self) -> str:
        """Returns a string representation suitable for raw body display."""
        raise NotImplementedError
    
    # ActionConfig subclasses typically have simple attributes, so default serialization might work
    # but it's safer to define it explicitly or ensure _get_serialization_params is robust.

class ChoiceActionConfig(ActionConfig):
    def __init__(self, level_name: str, choices: List[ActionConfig]):
        self.level_name = level_name
        self.choices = choices

    def _repr_indent(self, level=0):
        indent = "  " * level
        choices_str_list = []
        for i, choice in enumerate(self.choices):
            # Standardize indentation to level + 2 for items within lists
            choices_str_list.append(f"{indent}  [Choice {i+1}]:\n{choice._repr_indent(level + 2)}")
        return f"{indent}[ChoiceAction]({self.level_name})\n" + "\n".join(choices_str_list)

    def to_string_representation(self) -> str:
        return " ++ ".join([choice.to_string_representation() for choice in self.choices])

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "level_name": self.level_name,
            "choices": self.choices,
        }

class SequenceActionConfig(ActionConfig):
    def __init__(self, actions: Optional[List[ActionConfig]] = None):
        self.actions = actions or []

    def _repr_indent(self, level=0):
        indent = "  " * level
        # Standardize indentation to level + 1 for items within sequences
        actions_str = "\n".join([action._repr_indent(level + 1) for action in self.actions])
        return f"{indent}[SequenceAction]\n{actions_str or f'{indent}  (None)'}"

    def to_string_representation(self) -> str:
        return "; ".join([action.to_string_representation() for action in self.actions])

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {"actions": self.actions}

class DeterministicAssignmentActionConfig(ActionConfig):
    def __init__(self, variable: str, value: str):
        self.variable = variable
        self.value = value

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[DeterministicAssignment]: {self.variable} := '{self.value}'"

    def to_string_representation(self) -> str:
        return f"{self.variable} := {self.value}"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "value": self.value,
        }

class NondeterministicAssignmentActionConfig(ActionConfig):
    def __init__(self, variable: str):
        self.variable = variable

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[NondeterministicAssignment]: {self.variable} := *"

    def to_string_representation(self) -> str:
        return f"{self.variable} := *"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {"variable": self.variable}

class ActionConstraintConfig(ActionConfig):
    def __init__(self, formula: str):
        self.formula = formula

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[ActionConstraint]: '{self.formula}'"

    def to_string_representation(self) -> str:
        return f"?{self.formula}"

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {"formula": self.formula}

class CallHybridProgramConfig(ActionConfig):
    def __init__(self, hp_name: str):
        self.hp_name = hp_name

    def _repr_indent(self, level=0):
        indent = "  " * level
        return f"{indent}[CallHybridProgram]: {self.hp_name}"

    def to_string_representation(self) -> str:
        return self.hp_name

    def _get_serialization_params(self) -> Dict[str, Any]:
        return {"hp_name": self.hp_name}

# Define ActionType for type hinting
ActionType = Union[
    ChoiceActionConfig,
    SequenceActionConfig,
    DeterministicAssignmentActionConfig,
    NondeterministicAssignmentActionConfig,
    ActionConstraintConfig,
    CallHybridProgramConfig,
    HybridProgramStepConfig
]