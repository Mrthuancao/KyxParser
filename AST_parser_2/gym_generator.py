import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import textwrap
import re
from typing import List, Dict, Any, Optional, Union
from sympy import sympify, Symbol, Eq, Ge, Le, And, Or, Not, Abs, S, solve, simplify, Function
import sympy
from collections import OrderedDict, defaultdict
import math # Added for Python eval context
# Removed: import pathlib # No longer needed as template is now a string in a .py file

# Import AST nodes from user's ast_config.py
from ast_config import (
    ASTNode, KeYmaeraModel, ArchiveEntry, DefinitionsBlock, Definition, FuncDef, PredDef, HPDef,
    ProgramVariablesBlock, ProgramVariableDecl, ProblemBlock, TacticBlock,
    Term, Literal, VarRef, BinaryTermOp, UnaryTermOp, FunctionCall,
    Formula, BooleanLiteral, BinaryFormulaOp, UnaryFormulaOp, Comparison, Quantifier,
    ModalFormula, PredicateCall, C_Formula,
    HybridProgram, Assignment, NondetAssignment, Test, Sequence, Choice, Iteration,
    ODE, ODEEquation, HPBlock
)

# Import the lexer and parser from user's files
from keymaera_lexer import KeYmaeraLexer, parse_annotations
from keymaera_parser import KeYmaeraParser

# Import pyast_config (user's provided)
from pyast_config import (
    GymEnvironmentConfig,
    VariableDeclarationConfig,
    ConstantDefinitionConfig,
    FuncDefConfig,
    HPDefConfig,
    ProblemConfig
)

# Import the template code as a string from the new Python file
from environment_template_code import ENVIRONMENT_TEMPLATE_CODE


class GymnasiumEnvironmentGenerator:
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.config = GymEnvironmentConfig(environment_name=environment_name)
        self.indent_level = 0
        self.indent_str = "    " # 4 spaces
        self.hp_definition_map: Dict[str, HPDef] = {} # To store HPDef ASTs for lookup
        self.func_definition_map: Dict[str, FuncDef] = {} # To store FuncDef ASTs for lookup
        self.pred_definition_map: Dict[str, PredDef] = {} # To store PredDef ASTs for lookup

    def _indent(self, text: str) -> str:
        lines = text.splitlines()
        indented_lines = [self.indent_str * self.indent_level + line for line in lines]
        return "\n".join(indented_lines)

    # --- Python Evaluation String Conversion ---
    _PYTHON_TERM_OPS = {'+': '+', '-': '-', '*': '*', '/': '/', '^': '**'}
    _PYTHON_FORMULA_OPS = {
        '=': '==', '!=': '!=', '>': '>', '<': '<', '>=': '>=', '<=': '<=',
        '&&': 'and', '&': 'and', '|': 'or'
    }

    def _term_to_string_python(self, term: Term) -> str:
        """Converts an ast_config.Term node to its string representation for Python eval()."""
        if isinstance(term, Literal):
            # Ensure boolean literals are correctly represented as Python booleans, not 'true'/'false' strings
            return str(term.value)
        elif isinstance(term, VarRef):
            var_name = term.name
            # Check if this VarRef refers to a FuncDef that is NOT a constant and has no arguments.
            # If it's a constant, its value is directly in self.current_state.
            # If it's a function with arguments, it would be a FunctionCall AST node.
            # This handles the case where a function *without* arguments is referenced as a simple variable.
            if var_name in self.func_definition_map:
                func_def = self.func_definition_map[var_name]
                # Only treat as a function call if it has no args AND it's not a constant.
                # Constants are already handled by being put into self.config.constants
                # and then into self.current_state.
                if not func_def.args and not func_def.annotations.get('const'):
                    return f"self._{var_name}_impl()"
            return var_name
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_string_python(term.left)
            right_str = self._term_to_string_python(term.right)
            op = self._PYTHON_TERM_OPS.get(term.op, term.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_string_python(term.operand)
            if term.op == "'":
                # Derivatives should not appear in Python eval contexts for terms
                raise ValueError(f"Derivative operator '{term.op}' not supported in Python term evaluation: {term}")
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_string_python(arg) for arg in term.args)
            # Generate a Python call to the user_defined_functions' implementation method
            # This will be eval'd in the generated environment, so `self.` is needed.
            return f"self._{term.name}_impl({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for Python eval: {type(term)}")

    def _formula_to_string_python(self, formula: Formula) -> str:
        """Converts an ast_config.Formula node to its string representation for Python eval()."""
        if isinstance(formula, BooleanLiteral):
            return str(formula.value) # Python True/False
        elif isinstance(formula, Comparison):
            left_str = self._term_to_string_python(formula.left)
            right_str = self._term_to_string_python(formula.right)
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_string_python(formula.left)
            right_str = self._formula_to_string_python(formula.right)
            if formula.op == "->":
                return f"(not ({left_str}) or ({right_str}))" # Implication A -> B is (not A or B)
            elif formula.op == "<->":
                return f"((({left_str}) and ({right_str})) or (not ({left_str}) and not ({right_str})))" # Equivalence A <-> B is ((A and B) or (not A and not B))
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_string_python(formula.operand)
            if formula.op == "!": # dL '!' maps to Python 'not'
                return f"(not ({operand_str}))"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_string_python(arg) for arg in formula.args)
            # Call generated function, `self.` is needed for eval context
            return f"self._{formula.name}_impl({args_str})"
        elif isinstance(formula, (ModalFormula, Quantifier, C_Formula)):
            # These formula types are not meant for direct Python evaluation in this context
            raise ValueError(f"Formula type {type(formula)} cannot be directly converted to Python for evaluation: {formula}")
        else:
            raise ValueError(f"Unsupported Formula type for Python eval: {type(formula)}")

    # --- SymPy String Conversion (for ODEs) ---
    def _term_to_string_sympy(self, term: Term) -> str:
        """Converts an ast_config.Term node to its string representation for SymPy."""
        if isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, VarRef):
            return term.name
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_string_sympy(term.left)
            right_str = self._term_to_string_sympy(term.right)
            # Add parentheses for correct operator precedence
            return f"({left_str} {term.op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_string_sympy(term.operand)
            if term.op == "'":
                return f"{operand_str}_prime" # SymPy can handle this directly in ODEs
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_string_sympy(arg) for arg in term.args)
            # Generate a symbolic function call for SymPy to parse
            return f"sympy.Function('{term.name}')({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for SymPy: {type(term)}")

    def _formula_to_string_sympy(self, formula: Formula) -> str:
        """Converts an ast_config.Formula node to its string representation for SymPy."""
        if isinstance(formula, BooleanLiteral):
            return str(formula.value) # Python True/False
        elif isinstance(formula, Comparison):
            left_str = self._term_to_string_sympy(formula.left)
            right_str = self._term_to_string_sympy(formula.right)
            # Map dL comparison operators to SymPy
            if formula.op == '=': return f"sympy.Eq({left_str}, {right_str})"
            if formula.op == '!=': return f"sympy.Ne({left_str}, {right_str})"
            if formula.op == '>': return f"sympy.Gt({left_str}, {right_str})"
            if formula.op == '<': return f"sympy.Lt({left_str}, {right_str})"
            if formula.op == '>=': return f"sympy.Ge({left_str}, {right_str})"
            if formula.op == '<=': return f"sympy.Le({left_str}, {right_str})"
            return f"({left_str} {formula.op} {right_str})" # Fallback for unmapped
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_string_sympy(formula.left)
            right_str = self._formula_to_string_sympy(formula.right)
            # Map dL operators to SymPy equivalents
            if formula.op == "->":
                return f"sympy.Implies({left_str}, {right_str})"
            elif formula.op == "<->":
                return f"sympy.Equivalent({left_str}, {right_str})"
            elif formula.op in ["&&", "&"]:
                return f"sympy.And({left_str}, {right_str})"
            elif formula.op == "|":
                return f"sympy.Or({left_str}, {right_str})"
            return f"({left_str} {formula.op} {right_str})" # Fallback for unmapped
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_string_sympy(formula.operand)
            if formula.op == "!": # dL '!' maps to SymPy 'Not'
                return f"sympy.Not({operand_str})"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_string_sympy(arg) for arg in formula.args)
            # Generate a symbolic predicate call for SymPy to parse
            return f"sympy.Function('{formula.name}')({args_str})"
        elif isinstance(formula, ModalFormula):
            program_str = self._hp_to_string(formula.program)
            formula_str = self._formula_to_string_sympy(formula.formula)
            return f"[{program_str}]({formula_str})"
        elif isinstance(formula, Quantifier):
            vars_str = ", ".join(v.name for v in formula.variables)
            body_str = self._formula_to_string_sympy(formula.body)
            # SymPy has Quantifier support, but for this context, just stringify
            return f"{formula.quantifier_type}({vars_str}).({body_str})"
        elif isinstance(formula, C_Formula):
            return f"C({self._formula_to_string_sympy(formula.formula)})"
        else:
            raise ValueError(f"Unsupported Formula type for SymPy: {type(formula)}")


    def _hp_to_string(self, hp: HybridProgram) -> str:
        """Converts an ast_config.HybridProgram node to its string representation."""
        if isinstance(hp, Assignment):
            var_str = self._term_to_string_python(hp.var) # Assignment var is Python, expr is Python
            expr_str = self._term_to_string_python(hp.expr)
            return f"{var_str} := {expr_str}"
        elif isinstance(hp, NondetAssignment):
            var_str = self._term_to_string_python(hp.var)
            return f"{{{var_str} := *}}"
        elif isinstance(hp, Test):
            condition_str = self._formula_to_string_python(hp.condition) # Test condition is Python
            return f"?({condition_str})"
        elif isinstance(hp, Sequence):
            programs_str = "; ".join(self._hp_to_string(p) for p in hp.programs)
            return f"({programs_str})"
        elif isinstance(hp, Choice):
            left_str = self._hp_to_string(hp.left)
            right_str = self._hp_to_string(hp.right)
            return f"({left_str} ++ {right_str})"
        elif isinstance(hp, Iteration):
            program_str = self._hp_to_string(hp.program)
            # Invariant is a formula, for iteration annotation, it's a Python string
            invariant_str = f"@invariant({self._formula_to_string_python(hp.invariant)})" if hp.invariant else ""
            return f"({program_str})*{invariant_str}"
        elif isinstance(hp, ODE):
            print(f"DEBUG: _hp_to_string (ODE): hp.equations content: {hp.equations}")
            # ODE equations and domain constraints must be SymPy compatible
            equations_str = ", ".join(f"{self._term_to_string_sympy(eq.var)}' = {self._term_to_string_sympy(eq.expr)}" for eq in hp.equations)

            # Use a distinct separator for the domain constraint, if it exists
            if hp.domain_constraint:
                domain_constraint_sympy_str = self._formula_to_string_sympy(hp.domain_constraint)
                # The domain constraint should ideally be a single SymPy.And(...) expression.
                # If it's not already wrapped in sympy.And, we might need to wrap it.
                # However, _formula_to_string_sympy for BinaryFormulaOp('&') already wraps it.
                # So, we just use the generated string directly.
                domain_str = f"@@@DOMAIN_CONSTRAINT@@@{domain_constraint_sympy_str}"
            else:
                domain_str = ""

            print(f"DEBUG: _hp_to_string (ODE): generated equations_str='{equations_str}', domain_str='{domain_str}'")
            return f"({equations_str}{domain_str})"
        elif isinstance(hp, HPBlock): # For blocks like {program}
            print(f"DEBUG: _hp_to_string (HPBlock): Inner program type: {type(hp.program)}")
            return f"{{{self._hp_to_string(hp.program)}}}"
        elif isinstance(hp, VarRef): # HP references (e.g., dwwpdl)
            return hp.name
        else:
            raise ValueError(f"Unsupported HybridProgram type: {type(hp)}")

    def _resolve_hp_to_ode(self, hp_name: str) -> Optional[ODE]:
        """
        Recursively resolves an HPDef by name to find its contained ODE block.
        This is a simplified resolver for the given dL structure.
        """
        print(f"DEBUG: _resolve_hp_to_ode: START resolving '{hp_name}'")
        if hp_name not in self.hp_definition_map:
            print(f"DEBUG: _resolve_hp_to_ode: '{hp_name}' not in hp_definition_map. Returning None.")
            return None

        current_hp_def = self.hp_definition_map[hp_name]
        current_hp_body = current_hp_def.body
        print(f"DEBUG: _resolve_hp_to_ode: Resolving '{hp_name}', body type: {type(current_hp_body)}")

        def find_ode_in_hp(hp_node: HybridProgram, path: List[str]) -> Optional[ODE]:
            current_path = path + [str(type(hp_node).__name__)]
            # print(f"DEBUG: find_ode_in_hp: Path: {' -> '.join(current_path)}. Current node type: {type(hp_node)} for {hp_node}")
            if isinstance(hp_node, ODE):
                print(f"DEBUG: find_ode_in_hp: Found ODE at path: {' -> '.join(current_path)}. Returning ODE.")
                return hp_node
            elif isinstance(hp_node, Sequence):
                # print(f"DEBUG: find_ode_in_hp: Entering Sequence at path: {' -> '.join(current_path)} with {len(hp_node.programs)} programs.")
                for i, sub_hp in enumerate(hp_node.programs):
                    # print(f"DEBUG: find_ode_in_hp: Checking Sequence sub_hp[{i}] at path: {' -> '.join(current_path)}. Type: {type(sub_hp)} for {sub_hp}")
                    ode = find_ode_in_hp(sub_hp, current_path + [f"Sequence[{i}]"])
                    if ode:
                        # print(f"DEBUG: find_ode_in_hp: Found ODE in Sequence sub_hp[{i}] at path: {' -> '.join(current_path)}. Returning ODE.")
                        return ode
                # print(f"DEBUG: find_ode_in_hp: No ODE found in Sequence at path: {' -> '.join(current_path)}. Returning None.")
            elif isinstance(hp_node, Choice):
                # print(f"DEBUG: find_ode_in_hp: Entering Choice at path: {' -> '.join(current_path)}.")
                ode = find_ode_in_hp(hp_node.left, current_path + ["Choice.left"])
                if ode:
                    # print(f"DEBUG: find_ode_in_hp: Found ODE in Choice left branch at path: {' -> '.join(current_path)}. Returning ODE.")
                    return ode
                ode = find_ode_in_hp(hp_node.right, current_path + ["Choice.right"])
                if ode:
                    # print(f"DEBUG: find_ode_in_hp: Found ODE in Choice right branch at path: {' -> '.join(current_path)}. Returning ODE.")
                    return ode
                # print(f"DEBUG: find_ode_in_hp: No ODE found in Choice at path: {' -> '.join(current_path)}. Returning None.")
            elif isinstance(hp_node, Iteration):
                # print(f"DEBUG: find_ode_in_hp: Entering Iteration at path: {' -> '.join(current_path)}. Checking program body.")
                result = find_ode_in_hp(hp_node.program, current_path + ["Iteration.program"])
                # print(f"DEBUG: find_ode_in_hp: Iteration program body search result at path: {' -> '.join(current_path)}: {result}")
                return result
            elif isinstance(hp_node, HPBlock):
                # print(f"DEBUG: find_ode_in_hp: Entering HPBlock at path: {' -> '.join(current_path)}. Checking inner program.")
                result = find_ode_in_hp(hp_node.program, current_path + ["HPBlock.program"])
                # print(f"DEBUG: find_ode_in_hp: HPBlock inner program search result at path: {' -> '.join(current_path)}: {result}")
                return result
            elif isinstance(hp_node, VarRef):
                # print(f"DEBUG: find_ode_in_hp: Encountered VarRef '{hp_node.name}' at path: {' -> '.join(current_path)}. Recursively resolving.")
                # Pass the existing path to the recursive call for better context
                resolved_ode = self._resolve_hp_to_ode(hp_node.name)
                if resolved_ode:
                    # print(f"DEBUG: find_ode_in_hp: VarRef '{hp_node.name}' resolved to ODE at path: {' -> '.join(current_path)}. Returning ODE.")
                    return resolved_ode
                # print(f"DEBUG: find_ode_in_hp: VarRef '{hp_node.name}' did NOT resolve to an ODE at path: {' -> '.join(current_path)}. Returning None.")
            # else:
                # print(f"DEBUG: find_ode_in_hp: Node type {type(hp_node)} is not an ODE container at path: {' -> '.join(current_path)}. Returning None.")
            return None

        result = find_ode_in_hp(current_hp_body, [f"_resolve_hp_to_ode('{hp_name}')"])
        print(f"DEBUG: _resolve_hp_to_ode: END resolving '{hp_name}'. Final result: {result}")
        return result

    def _extract_problem_components(self, problem_formula: Formula) -> Dict[str, Any]:
        initial_condition = "True" # Python True/False
        program_ode_str = "" # Initialize here
        goal_formula_str = "True"
        termination_condition_str = "True" # Default to True to avoid infinite loops if no goal
        reward_function_str = "0.0"
        loop_invariants = []
        differential_invariants = [] # Not used in this dL example, but keep for future
        differential_ghosts = [] # Not used in this dL example, but keep for future

        current_formula = problem_formula

        print(f"DEBUG: _extract_problem_components: Initial problem_formula type: {type(problem_formula)}")
        print(f"DEBUG: _extract_problem_components: Initial problem_formula: {problem_formula}")

        # Handle implications (e.g., initial -> ...)
        if isinstance(current_formula, BinaryFormulaOp) and current_formula.op == "->":
            initial_condition = self._formula_to_string_python(current_formula.left)
            current_formula = current_formula.right # This is the post-implication formula
            print(f"DEBUG: _extract_problem_components: After implication, current_formula type: {type(current_formula)}")
            print(f"DEBUG: _extract_problem_components: After implication, current_formula: {current_formula}")


        # The post-implication formula is expected to be:
        # BinaryFormulaOp('&&', ModalFormula('box', VarRef('dwwpdl'), safety_formula), ModalFormula('diamond', VarRef('dwwpdl'), liveness_formula))
        # However, due to parser behavior, it might be:
        # ModalFormula('box', VarRef('dwwpdl'), BinaryFormulaOp('&', actual_safety_formula, ModalFormula('diamond', VarRef('dwwpdl'), liveness_formula)))

        print(f"DEBUG: _extract_problem_components: Before safety/liveness check, current_formula type: {type(current_formula)}")
        print(f"DEBUG: _extract_problem_components: Before safety/liveness check, current_formula: {current_formula}")

        safety_modal_formula = None
        liveness_modal_formula = None

        if isinstance(current_formula, BinaryFormulaOp) and current_formula.op in ["&&", "&"]:
            print(f"DEBUG: _extract_problem_components: current_formula is BinaryFormulaOp with op '{current_formula.op}'")
            print(f"DEBUG: _extract_problem_components: current_formula.left type: {type(current_formula.left)}")
            print(f"DEBUG: _extract_problem_components: current_formula.right type: {type(current_formula.right)}")

            if isinstance(current_formula.left, ModalFormula) and current_formula.left.modality == 'box':
                safety_modal_formula = current_formula.left
                print(f"DEBUG: _extract_problem_components: Identified safety_modal_formula: {safety_modal_formula}")
            else:
                print(f"DEBUG: _extract_problem_components: current_formula.left NOT a box ModalFormula. Type: {type(current_formula.left)}, Modality: {getattr(current_formula.left, 'modality', 'N/A')}")

            if isinstance(current_formula.right, ModalFormula) and current_formula.right.modality == 'diamond':
                liveness_modal_formula = current_formula.right
                print(f"DEBUG: _extract_problem_components: Identified liveness_modal_formula: {liveness_modal_formula}")
            else:
                print(f"DEBUG: _extract_problem_components: current_formula.right NOT a diamond ModalFormula. Type: {type(current_formula.right)}, Modality: {getattr(current_formula.right, 'modality', 'N/A')}")
        elif isinstance(current_formula, ModalFormula) and current_formula.modality == 'box':
            # This handles the observed parser behavior where the diamond is nested
            print(f"DEBUG: _extract_problem_components: Detected nested ModalFormula structure.")
            safety_modal_formula = current_formula

            # Now, look inside its formula for the conjunct that contains the diamond
            if isinstance(safety_modal_formula.formula, BinaryFormulaOp) and safety_modal_formula.formula.op in ["&&", "&"]:
                # The left part is the actual safety formula for the box
                # The right part should be the diamond modal formula
                if isinstance(safety_modal_formula.formula.right, ModalFormula) and safety_modal_formula.formula.right.modality == 'diamond':
                    liveness_modal_formula = safety_modal_formula.formula.right
                    # Create a new ModalFormula object for safety_modal_formula with the correct inner formula
                    safety_modal_formula = ModalFormula(
                        modality='box',
                        program=safety_modal_formula.program,
                        formula=safety_modal_formula.formula.left # This is the actual safety formula
                    )
                    print(f"DEBUG: _extract_problem_components: Extracted nested liveness_modal_formula: {liveness_modal_formula}")
                    print(f"DEBUG: _extract_problem_components: Adjusted safety_modal_formula: {safety_modal_formula}")
                else:
                    print(f"DEBUG: _extract_problem_components: Inner formula of box is BinaryFormulaOp, but right part is not diamond modal. Type: {type(safety_modal_formula.formula.right)}")
            else:
                print(f"DEBUG: _extract_problem_components: Inner formula of box is not a BinaryFormulaOp. Type: {type(safety_modal_formula.formula)}")
        else:
            print(f"DEBUG: _extract_problem_components: current_formula is NOT a BinaryFormulaOp with op in ['&&', '&'] or is not ModalFormula. Type: {type(current_formula)}, Op: {getattr(current_formula, 'op', 'N/A')}")

        print(f"DEBUG: _extract_problem_components: Value of safety_modal_formula (after processing block): {safety_modal_formula}")
        print(f"DEBUG: _extract_problem_components: Value of liveness_modal_formula (after processing block): {liveness_modal_formula}")


        # Extract program (ODE) from the safety modal formula
        if safety_modal_formula and isinstance(safety_modal_formula.program, VarRef):
            hp_name = safety_modal_formula.program.name
            print(f"DEBUG: _extract_problem_components: Attempting to resolve ODE for HP: '{hp_name}'")
            ode_block = self._resolve_hp_to_ode(hp_name)
            if ode_block:
                _temp_program_ode_str = self._hp_to_string(ode_block)
                print(f"DEBUG: _extract_problem_components: _temp_program_ode_str: '{_temp_program_ode_str}'")
                program_ode_str = _temp_program_ode_str # Explicitly assign to outer scope variable
            else:
                raise ValueError(f"Could not resolve ODE block from hybrid program '{hp_name}'")

            # Extract loop invariants from the Iteration containing dwwpdl
            dwwpdl_hp_def = self.hp_definition_map.get(hp_name)
            if dwwpdl_hp_def and isinstance(dwwpdl_hp_def.body, Iteration) and dwwpdl_hp_def.body.invariant:
                # Store invariant as a dictionary to retain its name and string representation
                if isinstance(dwwpdl_hp_def.body.invariant, PredicateCall):
                    loop_invariants.append({
                        "name": dwwpdl_hp_def.body.invariant.name,
                        "formula_str": self._formula_to_string_python(dwwpdl_hp_def.body.invariant)
                    })
                else:
                    loop_invariants.append({
                        "name": None, # Not a named predicate
                        "formula_str": self._formula_to_string_python(dwwpdl_hp_def.body.invariant)
                    })

            # Extract safety goal
            print(f"DEBUG: _extract_problem_components: Raw safety_modal_formula.formula: {repr(safety_modal_formula.formula)}")
            try:
                _temp_goal_formula_str = self._formula_to_string_python(safety_modal_formula.formula) # Assign to temp variable
                print(f"DEBUG: _extract_problem_components: _temp_goal_formula_str: '{_temp_goal_formula_str}'")
                goal_formula_str = _temp_goal_formula_str # Explicitly assign to outer scope variable
            except Exception as e:
                print(f"ERROR: Failed to convert safety_modal_formula.formula to string: {e}")
                print(f"DEBUG: safety_modal_formula.formula type: {type(safety_modal_formula.formula)}")
                print(f"DEBUG: safety_modal_formula.formula content: {safety_modal_formula.formula}")
                goal_formula_str = "True" # Fallback to avoid breaking environment generation

            # Termination condition: if safety goal is NOT met
            termination_condition_str = f"(not ({goal_formula_str}))"

            print(f"DEBUG: _extract_problem_components: program_ode_str AFTER safety/termination logic: '{program_ode_str}'")
            print(f"DEBUG: _extract_problem_components: goal_formula_str AFTER safety/termination logic: '{goal_formula_str}'")
        else:
            print(f"DEBUG: _extract_problem_components: !!! SKIPPING safety_modal_formula processing block. safety_modal_formula: {safety_modal_formula}")
            if safety_modal_formula:
                print(f"DEBUG: type(safety_modal_formula.program): {type(safety_modal_formula.program)}")


        # Extract reward from liveness modal formula
        if liveness_modal_formula and isinstance(liveness_modal_formula.formula, PredicateCall):
            ensure_predicate_call = liveness_modal_formula.formula
            ensure_pred_def = self.pred_definition_map.get(ensure_predicate_call.name)
            print(f"DEBUG: _extract_problem_components: ensure_predicate_call: {ensure_predicate_call}")
            print(f"DEBUG: _extract_problem_components: ensure_pred_def: {ensure_pred_def}")
            ensure_goal_str = self._formula_to_string_python(ensure_predicate_call) # This is `ensure(xr, vr, xg)`
            print(f"DEBUG: _extract_problem_components: ensure_goal_str (from _formula_to_string_python): '{ensure_goal_str}'")

            if ensure_pred_def:
                progress_annotation = ensure_pred_def.annotations.get('progress')
                print(f"DEBUG: _extract_problem_components: progress_annotation: {progress_annotation}")
                if progress_annotation:
                    reward_value = progress_annotation.get('reward', 0.0)
                    print(f"DEBUG: _extract_problem_components: reward_value from annotation: {reward_value}")
                    reward_function_str = f"{reward_value} if ({ensure_goal_str}) else -0.1" # Default small penalty
                    print(f"DEBUG: _extract_problem_components: constructed reward_function_str (with annotation): '{reward_function_str}'")
                else:
                    reward_function_str = f"1.0 if ({ensure_goal_str}) else -0.01" # Default reward if no annotation
                    print(f"DEBUG: _extract_problem_components: constructed reward_function_str (no annotation fallback): '{reward_function_str}'")
            else:
                reward_function_str = f"1.0 if ({ensure_goal_str}) else -0.01" # Fallback if pred def not found
                print(f"DEBUG: _extract_problem_components: constructed reward_function_str (no pred def fallback): '{reward_function_str}'")
        else:
            # Fallback reward if liveness or ensure predicate not found/annotated
            # If the safety goal is met, provide a positive reward. Otherwise, a penalty.
            reward_function_str = f"1.0 if ({goal_formula_str}) else -0.01"
            print(f"DEBUG: _extract_problem_components: constructed reward_function_str (liveness fallback): '{reward_function_str}'")

        # Incorporate loop invariant safety into termination and reward
        for inv_data in loop_invariants:
            inv_name = inv_data["name"]
            inv_formula_str = inv_data["formula_str"]

            if inv_name: # Only process if it's a named predicate
                inv_pred_def = self.pred_definition_map.get(inv_name)
                if inv_pred_def:
                    safety_annotation = inv_pred_def.annotations.get('safety')
                    if safety_annotation and safety_annotation.get('on_failure') == 'truncate':
                        # If invariant is violated, terminate (truncate)
                        termination_condition_str = f"({termination_condition_str} or (not ({inv_formula_str})))"
                        # Also add penalty if defined
                        penalty = safety_annotation.get('penalty')
                        if penalty is not None:
                            # Ensure penalty is added correctly to the reward
                            # The reward function will become a more complex ternary expression
                            # This needs careful construction to avoid syntax errors if penalty is 0.0
                            reward_function_str = f"({reward_function_str}) + ({penalty} if (not ({inv_formula_str})) else 0.0)"
                else:
                    print(f"Warning: Could not find predicate definition for invariant '{inv_name}'. Annotations will not be applied.")
            # If inv_name is None, it's a direct formula, so no annotations to apply.

        print(f"DEBUG: _extract_problem_components: Final program_ode_str before return: '{program_ode_str}'")
        print(f"DEBUG: _extract_problem_components: Final goal_formula_str before return: '{goal_formula_str}'")
        print(f"DEBUG: _extract_problem_components: Final termination_condition_body before return: '{termination_condition_str}'")
        print(f"DEBUG: _extract_problem_components: Final reward_function_body before return: '{reward_function_str}'")


        return {
            "initial_condition": initial_condition,
            "program": program_ode_str, # This is now the ODE string
            "goal": goal_formula_str,
            "loop_invariants": [inv_data["formula_str"] for inv_data in loop_invariants], # Store just the string in config
            "differential_invariants": differential_invariants,
            "differential_ghosts": differential_ghosts,
            "termination_condition_body": termination_condition_str,
            "reward_function_body": reward_function_str
        }


    def _process_definitions(self, model_ast: KeYmaeraModel):
        if not model_ast.entries:
            return

        archive_entry = model_ast.entries[0]

        # First pass: Populate definition maps for resolution
        for definition in archive_entry.definitions.defs:
            if isinstance(definition, FuncDef):
                self.func_definition_map[definition.name] = definition
            elif isinstance(definition, PredDef):
                self.pred_definition_map[definition.name] = definition
            elif isinstance(definition, HPDef):
                self.hp_definition_map[definition.name] = definition

        # Second pass: Process for config, now that maps are populated
        # Process ProgramVariables (actual variables)
        for pv_decl in archive_entry.program_variables.vars:
            var_name = pv_decl.name
            var_type = pv_decl.type
            annotations = pv_decl.annotations # This is the dict from parse_annotations

            # Check for @const annotation
            const_annotation = annotations.get('const')
            if const_annotation:
                value = const_annotation.get('value')
                # If value is not explicitly set, try to derive from bounds if they are equal
                if value is None and const_annotation.get('bound'):
                    bounds = const_annotation['bound']
                    if isinstance(bounds, list) and len(bounds) == 2:
                        # If bounds are equal, use that value
                        if bounds[0] == bounds[1]:
                            value = str(bounds[0])
                        else:
                            # If bounds are a range, pick the lower bound as a default constant value
                            # This is a design choice; could also be midpoint or upper bound.
                            value = str(bounds[0])
                # DEBUG PRINTS START
                print(f"DEBUG: Generator._process_definitions: Adding constant config for ProgramVariable: {var_name}")
                # DEBUG PRINTS END
                self.config.constants.append(ConstantDefinitionConfig(
                    name=var_name,
                    value=str(value if value is not None else var_name) # Fallback to name if no value (should be rare now)
                ))
            else: # Regular variable, check for @var annotation
                lower_bound = None
                upper_bound = None
                initial_value = None
                var_annotation = annotations.get('var')
                if var_annotation:
                    if var_annotation.get('bound'):
                        bounds = var_annotation['bound']
                        if isinstance(bounds, list) and len(bounds) == 2:
                            lower_bound = str(bounds[0])
                            upper_bound = str(bounds[1])
                    if var_annotation.get('initial_value'):
                        initial_value = str(var_annotation['initial_value'])

                # DEBUG PRINTS START
                print(f"DEBUG: Generator._process_definitions: Adding variable config for ProgramVariable: {var_name}")
                # DEBUG PRINTS END
                self.config.variables.append(VariableDeclarationConfig(
                    name=var_name,
                    type=var_type,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    initial_value=initial_value
                ))

        # Process Definitions (Functions, Predicates, HPs)
        for definition in archive_entry.definitions.defs:
            if isinstance(definition, FuncDef):
                # Check if it's a FuncDef with a @const annotation (like 'ep', 'b', etc. in the example dL)
                const_annotation = definition.annotations.get('const')
                if const_annotation:
                    value = const_annotation.get('value')
                    if value is None and const_annotation.get('bound'):
                        bounds = const_annotation['bound']
                        if isinstance(bounds, list) and len(bounds) == 2:
                            # Use lower bound as default value for constant range if no specific value
                            value = str(bounds[0])
                    # DEBUG PRINTS START
                    print(f"DEBUG: Generator._process_definitions: Adding FuncDef as constant: {definition.name}")
                    # DEBUG PRINTS END
                    self.config.constants.append(ConstantDefinitionConfig(
                        name=definition.name,
                        value=str(value if value is not None else definition.name)
                    ))
                    # IMPORTANT: Do NOT add this to function_definitions if it's a constant
                    continue # Skip to the next definition

                # If not a constant, proceed as a regular function definition
                body_str = self._term_to_string_python(definition.body) # Use Python conversion
                print(f"DEBUG: Generator._process_definitions: Adding FuncDef as function: {definition.name}, args: {[arg.name for arg in definition.args]}, body: {body_str}")
                self.config.function_definitions.append(FuncDefConfig(
                    name=definition.name,
                    args=[arg.name for arg in definition.args],
                    body=body_str
                ))
            elif isinstance(definition, PredDef): # Explicitly handle PredDef
                body_str = self._formula_to_string_python(definition.body) # Use Python conversion
                print(f"DEBUG: Generator._process_definitions: Adding PredDef as function: {definition.name}, args: {[arg.name for arg in definition.args]}, body: {body_str}")
                # Store as FuncDefConfig for evaluation purposes in the generated env
                self.config.function_definitions.append(FuncDefConfig(
                    name=definition.name,
                    args=[arg.name for arg in definition.args],
                    body=body_str
                ))
            elif isinstance(definition, HPDef):
                print(f"DEBUG: Generator._process_definitions: Adding HPDef: {definition.name}")
                self.config.hp_definitions.append(HPDefConfig(
                    name=definition.name,
                    args=[], # HPDef in ast_config doesn't have explicit args list
                    body=self._hp_to_string(definition.body)
                ))

        # Set up initial_state_sampling_config based on variable bounds
        for var_config in self.config.variables:
            if var_config.initial_value:
                self.config.initial_state_sampling_config[var_config.name] = {"value": var_config.initial_value}
            elif var_config.lower_bound is not None and var_config.upper_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {
                    "min": var_config.lower_bound,
                    "max": var_config.upper_bound
                }
            elif var_config.lower_bound is not None:
                 self.config.initial_state_sampling_config[var_config.name] = {
                    "min": var_config.lower_bound,
                    "max": f"({var_config.lower_bound}) + 1.0" # Example: default upper bound
                }
            elif var_config.upper_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {
                    "min": f"({var_config.upper_bound}) - 1.0", # Example: default lower bound
                    "max": var_config.upper_bound
                }
            else:
                # Default for variables without explicit initial value or bounds
                self.config.initial_state_sampling_config[var_name] = {"min": "-1.0", "max": "1.0"}

        # Add constants to initial_state_sampling_config with their fixed values
        # This is important so their values are available during reset/initialization
        for const_config in self.config.constants:
            self.config.initial_state_sampling_config[const_config.name] = {"value": const_config.value}


    def _process_problem(self, model_ast: KeYmaeraModel):
        if not model_ast.entries:
            return

        archive_entry = model_ast.entries[0]
        problem_block_formula = archive_entry.problem.formula

        extracted_components = self._extract_problem_components(problem_block_formula)

        print(f"DEBUG: _process_problem: program_ode_str before config assignment: '{extracted_components['program']}'")
        print(f"DEBUG: _process_problem: extracted reward_function_body: {extracted_components['reward_function_body']}")
        print(f"DEBUG: _process_problem: extracted termination_condition_body: {extracted_components['termination_condition_body']}")


        self.config.problem = ProblemConfig(
            initial_condition=extracted_components["initial_condition"],
            program=extracted_components["program"], # This is now the ODE string
            goal=extracted_components["goal"],
            loop_invariants=extracted_components["loop_invariants"],
            differential_invariants=extracted_components["differential_invariants"],
            differential_ghosts=extracted_components["differential_ghosts"]
        )
        self.config.termination_condition_body = extracted_components["termination_condition_body"]
        self.config.reward_function_body = extracted_components["reward_function_body"]

    def _determine_action_and_observation_spaces(self):
        action_variables = []
        observation_variables = []

        # Identify action variable 'ar' (robot acceleration)
        # This is a heuristic based on the dL file where 'ar' is assigned in controlFunc
        # and used in ODEs.
        if 'ar' in [v.name for v in self.config.variables]:
            action_variables.append('ar')

        # All declared variables are potential observation variables
        for var_config in self.config.variables:
            observation_variables.append(var_config.name)

        # Also include 't' (time elapsed in current control step) and 'T' (global time remaining)
        # if they are not already explicitly in variables.
        # In this dL, 't' and 'T' are ProgramVariables, so they will be in self.config.variables.

        if action_variables:
            self.config.action_space_type = "continuous"
            self.config.action_space_bounds = {
                var.name: {"min": var.lower_bound if var.lower_bound else "-float('inf')",
                           "max": var.upper_bound if var.upper_bound else "float('inf')"}
                for var in self.config.variables if var.name in action_variables
            }
        else:
            self.config.action_space_type = "discrete"
            self.config.action_space_bounds = {"n": 1}

        if observation_variables:
            self.config.observation_space_type = "continuous"
            self.config.observation_space_bounds = {
                var.name: {"min": var.lower_bound if var.lower_bound is not None else "-float('inf')",
                           "max": var.upper_bound if var.upper_bound is not None else "float('inf')"}
                for var in self.config.variables if var.name in observation_variables
            }
        else:
            self.config.observation_space_type = "continuous"
            self.config.observation_space_bounds = {"dummy_obs": {"min": -1.0, "max": 1.0}} # Placeholder


    def generate_environment_code(self, model_ast: KeYmaeraModel) -> str:
        print("Entering generate_environment_code...") # Debug print
        try:
            # --- REMOVED PATHLIB AND FILE READING ---
            # The template is now directly imported as a string variable.
            code_template = ENVIRONMENT_TEMPLATE_CODE

            # Populate internal maps and config based on the AST
            self._process_definitions(model_ast)
            self._process_problem(model_ast)
            self._determine_action_and_observation_spaces()

            import dataclasses
            config_dict = dataclasses.asdict(self.config)
            config_json_str = json.dumps(config_dict, indent=4)

            # Use repr() to safely embed the JSON string into the generated Python code.
            # This will add quotes around the string and escape any internal quotes,
            # ensuring the outer f-string's triple quotes are not prematurely terminated.
            embedded_config_json_str = repr(textwrap.indent(config_json_str, '    '))

            # Prepare user-defined functions and predicates for the generated environment
            user_defined_methods_code_lines = []
            user_defined_function_assignments_lines = []

            for func_def_config in self.config.function_definitions:
                args_str = ", ".join(func_def_config.args)
                func_name = func_def_config.name
                # Build parameter list: omit trailing comma if no args
                params = f"self, {args_str}" if args_str else "self"
                body_expr = func_def_config.body # This is already a string

                # All generated functions will now call _evaluate_expression
                user_defined_methods_code_lines.append(
                    f"    def _{func_name}_impl({params}):\n"
                    f"        # Construct context for _evaluate_expression\n"
                    f"        context = {{}}\n"
                    f"        arg_names = {json.dumps(func_def_config.args)}\n"
                    f"        for name, val in zip(arg_names, [{args_str}]):\n"
                    f"            context[name] = val\n"
                    f"        return self._evaluate_expression({json.dumps(body_expr)}, context)\n"
                )
                user_defined_function_assignments_lines.append(f"        self.user_defined_functions['{func_name}'] = self._{func_name}_impl\n") # Indented for _setup_functions (8 spaces)

            user_defined_methods_code_str = "\n".join(user_defined_methods_code_lines)
            user_defined_function_assignments_str = "\n".join(user_defined_function_assignments_lines)

            # Now format the template with the generated strings
            # Avoid using str.format because the template contains many
            # literal braces (e.g. f-string snippets). Use simple replace
            # for the known placeholders instead.
            generated_code = code_template
            generated_code = generated_code.replace('{embedded_config_json_str}', embedded_config_json_str)
            generated_code = generated_code.replace('{metadata}', json.dumps(self.config.metadata, indent=4))
            generated_code = generated_code.replace('{user_defined_methods_code_str}', user_defined_methods_code_str)
            generated_code = generated_code.replace('{user_defined_function_assignments_str}', user_defined_function_assignments_str)
            # Collapse any doubled-brace escapes produced by the template into single braces
            # This prevents leftover '{{' or '}}' sequences from appearing in generated Python code
            generated_code = generated_code.replace('{{', '{').replace('}}', '}')
            # Collapse double-brace escapes from the template into single braces
            # The template used doubled braces to avoid accidental formatting; after substitution
            # we want actual single braces in the generated Python code.
            generated_code = generated_code.replace('{{', '{').replace('}}', '}')

            print("Exiting generate_environment_code successfully.") # Debug print
            return generated_code # Return the formatted string
        except Exception as e:
            print(f"Error during environment code generation: {e}")
            import traceback
            traceback.print_exc()
            return None

# --- New function to wrap the generation and file writing ---
def generate_gym_environment_file(dl_content: str, environment_name: str, output_filename: str):
    """
    Parses dL content, generates a Gymnasium environment, and writes it to a file.
    Assumes 'environment_template_code.py' exists in the same directory.
    """
    print(f"Starting environment generation for '{environment_name}'...")

    # Parse the dL content
    parser = KeYmaeraParser(dl_content)
    model_ast = parser.parse()
    print(f"Parsed dL content into AST: {model_ast}")

    # Generate the environment code string
    generator = GymnasiumEnvironmentGenerator(environment_name)
    generated_code = generator.generate_environment_code(model_ast)

    if generated_code:
        # Write the generated code to the specified file
        with open(output_filename, "w") as f:
            f.write(generated_code)
        print(f"Generated environment code saved to {output_filename}")
        return True
    else:
        print("Environment code generation failed.")
        return False

# Example Usage (modified __main__ block)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import textwrap
import re
from typing import List, Dict, Any, Optional, Union
from sympy import sympify, Symbol, Eq, Ge, Le, And, Or, Not, Abs, S, solve, simplify, Function
import sympy
from collections import OrderedDict, defaultdict, deque
import math 
# Removed: import pathlib # No longer needed as template is now a string in a .py file

# Import AST nodes from user's ast_config.py
from ast_config import (
    ASTNode, KeYmaeraModel, ArchiveEntry, DefinitionsBlock, Definition, FuncDef, PredDef, HPDef,
    ProgramVariablesBlock, ProgramVariableDecl, ProblemBlock, TacticBlock,
    Term, Literal, VarRef, BinaryTermOp, UnaryTermOp, FunctionCall,
    Formula, BooleanLiteral, BinaryFormulaOp, UnaryFormulaOp, Comparison, Quantifier,
    ModalFormula, PredicateCall, C_Formula,
    HybridProgram, Assignment, NondetAssignment, Test, Sequence, Choice, Iteration,
    ODE, ODEEquation, HPBlock
)

# Import the lexer and parser from user's files
from keymaera_lexer import KeYmaeraLexer, parse_annotations
from keymaera_parser import KeYmaeraParser

# Import pyast_config (user's provided)
from pyast_config import (
    GymEnvironmentConfig,
    VariableDeclarationConfig,
    ConstantDefinitionConfig,
    FuncDefConfig,
    HPDefConfig,
    ProblemConfig
)

# Import the template code as a string from the new Python file
from environment_template_code import ENVIRONMENT_TEMPLATE_CODE


class GymnasiumEnvironmentGenerator:
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.config = GymEnvironmentConfig(environment_name=environment_name)
        self.indent_level = 0
        self.indent_str = "    " # 4 spaces
        self.hp_definition_map: Dict[str, HPDef] = {} # To store HPDef ASTs for lookup
        self.func_definition_map: Dict[str, FuncDef] = {} # To store FuncDef ASTs for lookup
        self.pred_definition_map: Dict[str, PredDef] = {} # To store PredDef ASTs for lookup

    def _indent(self, text: str) -> str:
        lines = text.splitlines()
        indented_lines = [self.indent_str * self.indent_level + line for line in lines]
        return "\\n".join(indented_lines)

    # --- Python Evaluation String Conversion (for generator's internal use) ---
    _PYTHON_TERM_OPS = {'+': '+', '-': '-', '*': '*', '/': '/', '^': '**'}
    _PYTHON_FORMULA_OPS = {
        '=': '==', '!=': '!=', '>': '>', '<': '<', '>=': '>=', '<=': '<=',
        '&&': 'and', '&': 'and', '|': 'or'
    }

    def _term_to_string_python(self, term: Term) -> str:
        """Converts an ast_config.Term node to its string representation for Python eval()."""
        if isinstance(term, Literal):
            # Ensure boolean literals are correctly represented as Python booleans, not 'true'/'false' strings
            return str(term.value)
        elif isinstance(term, VarRef):
            var_name = term.name
            # Check if this VarRef refers to a FuncDef that is NOT a constant and has no arguments.
            # If it's a constant, its value is directly in self.current_state.
            # If it's a function with arguments, it would be a FunctionCall AST node.
            # This handles the case where a function *without* arguments is referenced as a simple variable.
            if var_name in self.func_definition_map:
                func_def = self.func_definition_map[var_name]
                # Only treat as a function call if it has no args AND it's not a constant.
                # Constants are already handled by being put into self.config.constants
                # and then into self.current_state.
                if not func_def.args and not func_def.annotations.get('const'):
                    return f"self._{var_name}_impl()"
            return var_name
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_string_python(term.left)
            right_str = self._term_to_string_python(term.right)
            op = self._PYTHON_TERM_OPS.get(term.op, term.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_string_python(term.operand)
            if term.op == "'":
                # Derivatives should not appear in Python eval contexts for terms
                raise ValueError(f"Derivative operator '{term.op}' not supported in Python term evaluation: {term}")
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_string_python(arg) for arg in term.args)
            # Generate a Python call to the user_defined_functions' implementation method
            # This will be eval'd in the generated environment, so `self.` is needed.
            return f"self._{term.name}_impl({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for Python eval: {type(term)}")

    def _formula_to_string_python(self, formula: Formula) -> str:
        """Converts an ast_config.Formula node to its string representation for Python eval()."""
        if isinstance(formula, BooleanLiteral):
            return str(formula.value) # Python True/False
        elif isinstance(formula, Comparison):
            left_str = self._term_to_string_python(formula.left)
            right_str = self._term_to_string_python(formula.right)
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_string_python(formula.left)
            right_str = self._formula_to_string_python(formula.right)
            if formula.op == "->":
                return f"(not ({left_str}) or ({right_str}))" # Implication A -> B is (not A or B)
            elif formula.op == "<->":
                return f"((({left_str}) and ({right_str})) or (not ({left_str}) and not ({right_str})))" # Equivalence A <-> B is ((A and B) or (not A and not B))
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_string_python(formula.operand)
            if formula.op == "!": # dL '!' maps to Python 'not'
                return f"(not ({operand_str}))"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_string_python(arg) for arg in formula.args)
            # Call generated function, `self.` is needed for eval context
            return f"self._{formula.name}_impl({args_str})"
        elif isinstance(formula, (ModalFormula, Quantifier, C_Formula)):
            # These formula types are not meant for direct Python evaluation in this context
            raise ValueError(f"Formula type {type(formula)} cannot be directly converted to Python for evaluation: {formula}")
        else:
            raise ValueError(f"Unsupported Formula type for Python eval: {type(formula)}")

    # --- SymPy String Conversion (for ODEs) ---
    def _term_to_string_sympy(self, term: Term) -> str:
        """Converts an ast_config.Term node to its string representation for SymPy."""
        if isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, VarRef):
            return term.name
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_string_sympy(term.left)
            right_str = self._term_to_string_sympy(term.right)
            # Add parentheses for correct operator precedence
            return f"({left_str} {term.op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_string_sympy(term.operand)
            if term.op == "'":
                return f"{operand_str}_prime" # SymPy can handle this directly in ODEs
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_string_sympy(arg) for arg in term.args)
            # Generate a symbolic function call for SymPy to parse
            return f"sympy.Function('{term.name}')({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for SymPy: {type(term)}")

    def _formula_to_string_sympy(self, formula: Formula) -> str:
        """Converts an ast_config.Formula node to its string representation for SymPy."""
        if isinstance(formula, BooleanLiteral):
            return str(formula.value) # Python True/False
        elif isinstance(formula, Comparison):
            left_str = self._term_to_string_sympy(formula.left)
            right_str = self._term_to_string_sympy(formula.right)
            # Map dL comparison operators to SymPy
            if formula.op == '=': return f"sympy.Eq({left_str}, {right_str})"
            if formula.op == '!=': return f"sympy.Ne({left_str}, {right_str})"
            if formula.op == '>': return f"sympy.Gt({left_str}, {right_str})"
            if formula.op == '<': return f"sympy.Lt({left_str}, {right_str})"
            if formula.op == '>=': return f"sympy.Ge({left_str}, {right_str})"
            if formula.op == '<=': return f"sympy.Le({left_str}, {right_str})"
            return f"({left_str} {formula.op} {right_str})" # Fallback for unmapped
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_string_sympy(formula.left)
            right_str = self._formula_to_string_sympy(formula.right)
            # Map dL operators to SymPy equivalents
            if formula.op == "->":
                return f"sympy.Implies({left_str}, {right_str})"
            elif formula.op == "<->":
                return f"sympy.Equivalent({left_str}, {right_str})"
            elif formula.op in ["&&", "&"]:
                return f"sympy.And({left_str}, {right_str})"
            elif formula.op == "|":
                return f"sympy.Or({left_str}, {right_str})"
            return f"({left_str} {formula.op} {right_str})" # Fallback for unmapped
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_string_sympy(formula.operand)
            if formula.op == "!": # dL '!' maps to SymPy 'Not'
                return f"sympy.Not({operand_str})"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_string_sympy(arg) for arg in formula.args)
            # Generate a symbolic predicate call for SymPy to parse
            return f"sympy.Function('{formula.name}')({args_str})"
        elif isinstance(formula, ModalFormula):
            program_str = self._hp_to_string(formula.program)
            formula_str = self._formula_to_string_sympy(formula.formula)
            return f"[{program_str}]({formula_str})"
        elif isinstance(formula, Quantifier):
            vars_str = ", ".join(v.name for v in formula.variables)
            body_str = self._formula_to_string_sympy(formula.body)
            # SymPy has Quantifier support, but for this context, just stringify
            return f"{formula.quantifier_type}({vars_str}).({body_str})"
        elif isinstance(formula, C_Formula):
            return f"C({self._formula_to_string_sympy(formula.formula)})"
        else:
            raise ValueError(f"Unsupported Formula type for SymPy: {type(formula)}")


    def _hp_to_string(self, hp: HybridProgram) -> str:
        """Converts an ast_config.HybridProgram node to its string representation."""
        if isinstance(hp, Assignment):
            var_str = self._term_to_string_python(hp.var) # Assignment var is Python, expr is Python
            expr_str = self._term_to_string_python(hp.expr)
            return f"{var_str} := {expr_str}"
        elif isinstance(hp, NondetAssignment):
            var_str = self._term_to_string_python(hp.var)
            return f"{{{var_str} := *}}"
        elif isinstance(hp, Test):
            condition_str = self._formula_to_string_python(hp.condition) # Test condition is Python
            return f"?({condition_str})"
        elif isinstance(hp, Sequence):
            programs_str = "; ".join(self._hp_to_string(p) for p in hp.programs)
            return f"({programs_str})"
        elif isinstance(hp, Choice):
            left_str = self._hp_to_string(hp.left)
            right_str = self._hp_to_string(hp.right)
            return f"({left_str} ++ {right_str})"
        elif isinstance(hp, Iteration):
            program_str = self._hp_to_string(hp.program)
            # Invariant is a formula, for iteration annotation, it's a Python string
            invariant_str = f"@invariant({self._formula_to_string_python(hp.invariant)})" if hp.invariant else ""
            return f"({program_str})*{invariant_str}"
        elif isinstance(hp, ODE):
            print(f"DEBUG: _hp_to_string (ODE): hp.equations content: {hp.equations}")
            # ODE equations and domain constraints must be SymPy compatible
            equations_str = ", ".join(f"{self._term_to_string_sympy(eq.var)}' = {self._term_to_string_sympy(eq.expr)}" for eq in hp.equations)

            # Use a distinct separator for the domain constraint, if it exists
            if hp.domain_constraint:
                domain_constraint_sympy_str = self._formula_to_string_sympy(hp.domain_constraint)
                # The domain constraint should ideally be a single SymPy.And(...) expression.
                # If it's not already wrapped in sympy.And, we might need to wrap it.
                # However, _formula_to_string_sympy for BinaryFormulaOp('&') already wraps it.
                # So, we just use the generated string directly.
                domain_str = f"@@@DOMAIN_CONSTRAINT@@@{domain_constraint_sympy_str}"
            else:
                domain_str = ""

            print(f"DEBUG: _hp_to_string (ODE): generated equations_str='{equations_str}', domain_str='{domain_str}'")
            return f"({equations_str}{domain_str})"
        elif isinstance(hp, HPBlock): # For blocks like {program}
            print(f"DEBUG: _hp_to_string (HPBlock): Inner program type: {type(hp.program)}")
            return f"{{{self._hp_to_string(hp.program)}}}"
        elif isinstance(hp, VarRef): # HP references (e.g., dwwpdl)
            return hp.name
        else:
            raise ValueError(f"Unsupported HybridProgram type: {type(hp)}")

    def _resolve_hp_to_ode(self, hp_name: str) -> Optional[ODE]:
        """
        Recursively resolves an HPDef by name to find its contained ODE block.
        This is a simplified resolver for the given dL structure.
        """
        print(f"DEBUG: _resolve_hp_to_ode: START resolving '{hp_name}'")
        if hp_name not in self.hp_definition_map:
            print(f"DEBUG: _resolve_hp_to_ode: '{hp_name}' not in hp_definition_map. Returning None.")
            return None

        current_hp_def = self.hp_definition_map[hp_name]
        current_hp_body = current_hp_def.body
        print(f"DEBUG: _resolve_hp_to_ode: Resolving '{hp_name}', body type: {type(current_hp_body)}")

        def find_ode_in_hp(hp_node: HybridProgram, path: List[str]) -> Optional[ODE]:
            current_path = path + [str(type(hp_node).__name__)]
            # print(f"DEBUG: find_ode_in_hp: Path: {' -> '.join(current_path)}. Current node type: {type(hp_node)} for {hp_node}")
            if isinstance(hp_node, ODE):
                print(f"DEBUG: find_ode_in_hp: Found ODE at path: {' -> '.join(current_path)}. Returning ODE.")
                return hp_node
            elif isinstance(hp_node, Sequence):
                # print(f"DEBUG: find_ode_in_hp: Entering Sequence at path: {' -> '.join(current_path)} with {len(hp_node.programs)} programs.")
                for i, sub_hp in enumerate(hp_node.programs):
                    # print(f"DEBUG: find_ode_in_hp: Checking Sequence sub_hp[{i}] at path: {' -> '.join(current_path)}. Type: {type(sub_hp)} for {sub_hp}")
                    ode = find_ode_in_hp(sub_hp, current_path + [f"Sequence[{i}]"])
                    if ode:
                        # print(f"DEBUG: find_ode_in_hp: Found ODE in Sequence sub_hp[{i}] at path: {' -> '.join(current_path)}. Returning ODE.")
                        return ode
                # print(f"DEBUG: find_ode_in_hp: No ODE found in Sequence at path: {' -> '.join(current_path)}. Returning None.")
            elif isinstance(hp_node, Choice):
                # print(f"DEBUG: find_ode_in_hp: Entering Choice at path: {' -> '.join(current_path)}.")
                ode = find_ode_in_hp(hp_node.left, current_path + ["Choice.left"])
                if ode:
                    # print(f"DEBUG: find_ode_in_hp: Found ODE in Choice left branch at path: {' -> '.join(current_path)}. Returning ODE.")
                    return ode
                ode = find_ode_in_hp(hp_node.right, current_path + ["Choice.right"])
                if ode:
                    # print(f"DEBUG: find_ode_in_hp: Found ODE in Choice right branch at path: {' -> '.join(current_path)}. Returning ODE.")
                    return ode
                # print(f"DEBUG: find_ode_in_hp: No ODE found in Choice at path: {' -> '.join(current_path)}. Returning None.")
            elif isinstance(hp_node, Iteration):
                # print(f"DEBUG: find_ode_in_hp: Entering Iteration at path: {' -> '.join(current_path)}. Checking program body.")
                result = find_ode_in_hp(hp_node.program, current_path + ["Iteration.program"])
                # print(f"DEBUG: find_ode_in_hp: Iteration program body search result at path: {' -> '.join(current_path)}: {result}")
                return result
            elif isinstance(hp_node, HPBlock):
                # print(f"DEBUG: find_ode_in_hp: Entering HPBlock at path: {' -> '.join(current_path)}. Checking inner program.")
                result = find_ode_in_hp(hp_node.program, current_path + ["HPBlock.program"])
                # print(f"DEBUG: find_ode_in_hp: HPBlock inner program search result at path: {' -> '.join(current_path)}: {result}")
                return result
            elif isinstance(hp_node, VarRef):
                # print(f"DEBUG: find_ode_in_hp: Encountered VarRef '{hp_node.name}' at path: {' -> '.join(current_path)}. Recursively resolving.")
                # Pass the existing path to the recursive call for better context
                resolved_ode = self._resolve_hp_to_ode(hp_node.name)
                if resolved_ode:
                    # print(f"DEBUG: find_ode_in_hp: VarRef '{hp_node.name}' resolved to ODE at path: {' -> '.join(current_path)}. Returning ODE.")
                    return resolved_ode
                # print(f"DEBUG: find_ode_in_hp: VarRef '{hp_node.name}' did NOT resolve to an ODE at path: {' -> '.join(current_path)}. Returning None.")
            # else:
                # print(f"DEBUG: find_ode_in_hp: Node type {type(hp_node)} is not an ODE container at path: {' -> '.join(current_path)}. Returning None.")
            return None

        result = find_ode_in_hp(current_hp_body, [f"_resolve_hp_to_ode('{hp_name}')"])
        print(f"DEBUG: _resolve_hp_to_ode: END resolving '{hp_name}'. Final result: {result}")
        return result

    def _extract_problem_components(self, problem_formula: Formula) -> Dict[str, Any]:
        initial_condition = "True" # Python True/False
        program_ode_str = "" # Initialize here
        goal_formula_str = "True"
        termination_condition_str = "True" # Default to True to avoid infinite loops if no goal
        reward_function_str = "0.0"
        loop_invariants = []
        differential_invariants = [] # Not used in this dL example, but keep for future
        differential_ghosts = [] # Not used in this dL example, but keep for future

        current_formula = problem_formula

        print(f"DEBUG: _extract_problem_components: Initial problem_formula type: {type(problem_formula)}")
        print(f"DEBUG: _extract_problem_components: Initial problem_formula: {problem_formula}")

        # Handle implications (e.g., initial -> ...)
        if isinstance(current_formula, BinaryFormulaOp) and current_formula.op == "->":
            initial_condition = self._formula_to_string_python(current_formula.left)
            current_formula = current_formula.right # This is the post-implication formula
            print(f"DEBUG: _extract_problem_components: After implication, current_formula type: {type(current_formula)}")
            print(f"DEBUG: _extract_problem_components: After implication, current_formula: {current_formula}")


        # The post-implication formula is expected to be:
        # BinaryFormulaOp('&&', ModalFormula('box', VarRef('dwwpdl'), safety_formula), ModalFormula('diamond', VarRef('dwwpdl'), liveness_formula))
        # However, due to parser behavior, it might be:
        # ModalFormula('box', VarRef('dwwpdl'), BinaryFormulaOp('&', actual_safety_formula, ModalFormula('diamond', VarRef('dwwpdl'), liveness_formula)))

        print(f"DEBUG: _extract_problem_components: Before safety/liveness check, current_formula type: {type(current_formula)}")
        print(f"DEBUG: _extract_problem_components: Before safety/liveness check, current_formula: {current_formula}")

        safety_modal_formula = None
        liveness_modal_formula = None

        if isinstance(current_formula, BinaryFormulaOp) and current_formula.op in ["&&", "&"]:
            print(f"DEBUG: _extract_problem_components: current_formula is BinaryFormulaOp with op '{current_formula.op}'")
            print(f"DEBUG: _extract_problem_components: current_formula.left type: {type(current_formula.left)}")
            print(f"DEBUG: _extract_problem_components: current_formula.right type: {type(current_formula.right)}")

            if isinstance(current_formula.left, ModalFormula) and current_formula.left.modality == 'box':
                safety_modal_formula = current_formula.left
                print(f"DEBUG: _extract_problem_components: Identified safety_modal_formula: {safety_modal_formula}")
            else:
                print(f"DEBUG: _extract_problem_components: current_formula.left NOT a box ModalFormula. Type: {type(current_formula.left)}, Modality: {getattr(current_formula.left, 'modality', 'N/A')}")

            if isinstance(current_formula.right, ModalFormula) and current_formula.right.modality == 'diamond':
                liveness_modal_formula = current_formula.right
                print(f"DEBUG: _extract_problem_components: Identified liveness_modal_formula: {liveness_modal_formula}")
            else:
                print(f"DEBUG: _extract_problem_components: current_formula.right NOT a diamond ModalFormula. Type: {type(current_formula.right)}, Modality: {getattr(current_formula.right, 'modality', 'N/A')}")
        elif isinstance(current_formula, ModalFormula) and current_formula.modality == 'box':
            # This handles the observed parser behavior where the diamond is nested
            print(f"DEBUG: _extract_problem_components: Detected nested ModalFormula structure.")
            safety_modal_formula = current_formula

            # Now, look inside its formula for the conjunct that contains the diamond
            if isinstance(safety_modal_formula.formula, BinaryFormulaOp) and safety_modal_formula.formula.op in ["&&", "&"]:
                # The left part is the actual safety formula for the box
                # The right part should be the diamond modal formula
                if isinstance(safety_modal_formula.formula.right, ModalFormula) and safety_modal_formula.formula.right.modality == 'diamond':
                    liveness_modal_formula = safety_modal_formula.formula.right
                    # Create a new ModalFormula object for safety_modal_formula with the correct inner formula
                    safety_modal_formula = ModalFormula(
                        modality='box',
                        program=safety_modal_formula.program,
                        formula=safety_modal_formula.formula.left # This is the actual safety formula
                    )
                    print(f"DEBUG: _extract_problem_components: Extracted nested liveness_modal_formula: {liveness_modal_formula}")
                    print(f"DEBUG: _extract_problem_components: Adjusted safety_modal_formula: {safety_modal_formula}")
                else:
                    print(f"DEBUG: _extract_problem_components: Inner formula of box is BinaryFormulaOp, but right part is not diamond modal. Type: {type(safety_modal_formula.formula.right)}")
            else:
                print(f"DEBUG: _extract_problem_components: Inner formula of box is not a BinaryFormulaOp. Type: {type(safety_modal_formula.formula)}")
        else:
            print(f"DEBUG: _extract_problem_components: current_formula is NOT a BinaryFormulaOp with op in ['&&', '&'] or is not ModalFormula. Type: {type(current_formula)}, Op: {getattr(current_formula, 'op', 'N/A')}")

        print(f"DEBUG: _extract_problem_components: Value of safety_modal_formula (after processing block): {safety_modal_formula}")
        print(f"DEBUG: _extract_problem_components: Value of liveness_modal_formula (after processing block): {liveness_modal_formula}")


        # Extract program (ODE) from the safety modal formula
        if safety_modal_formula and isinstance(safety_modal_formula.program, VarRef):
            hp_name = safety_modal_formula.program.name
            print(f"DEBUG: _extract_problem_components: Attempting to resolve ODE for HP: '{hp_name}'")
            ode_block = self._resolve_hp_to_ode(hp_name)
            if ode_block:
                _temp_program_ode_str = self._hp_to_string(ode_block)
                print(f"DEBUG: _extract_problem_components: _temp_program_ode_str: '{_temp_program_ode_str}'")
                program_ode_str = _temp_program_ode_str # Explicitly assign to outer scope variable
            else:
                raise ValueError(f"Could not resolve ODE block from hybrid program '{hp_name}'")

            # Extract loop invariants from the Iteration containing dwwpdl
            dwwpdl_hp_def = self.hp_definition_map.get(hp_name)
            if dwwpdl_hp_def and isinstance(dwwpdl_hp_def.body, Iteration) and dwwpdl_hp_def.body.invariant:
                # Store invariant as a dictionary to retain its name and string representation
                if isinstance(dwwpdl_hp_def.body.invariant, PredicateCall):
                    loop_invariants.append({
                        "name": dwwpdl_hp_def.body.invariant.name,
                        "formula_str": self._formula_to_string_python(dwwpdl_hp_def.body.invariant)
                    })
                else:
                    loop_invariants.append({
                        "name": None, # Not a named predicate
                        "formula_str": self._formula_to_string_python(dwwpdl_hp_def.body.invariant)
                    })

            # Extract safety goal
            print(f"DEBUG: _extract_problem_components: Raw safety_modal_formula.formula: {repr(safety_modal_formula.formula)}")
            try:
                _temp_goal_formula_str = self._formula_to_string_python(safety_modal_formula.formula) # Assign to temp variable
                print(f"DEBUG: _extract_problem_components: _temp_goal_formula_str: '{_temp_goal_formula_str}'")
                goal_formula_str = _temp_goal_formula_str # Explicitly assign to outer scope variable
            except Exception as e:
                print(f"ERROR: Failed to convert safety_modal_formula.formula to string: {e}")
                print(f"DEBUG: safety_modal_formula.formula type: {type(safety_modal_formula.formula)}")
                print(f"DEBUG: safety_modal_formula.formula content: {safety_modal_formula.formula}")
                goal_formula_str = "True" # Fallback to avoid breaking environment generation

            # Termination condition: if safety goal is NOT met
            termination_condition_str = f"(not ({goal_formula_str}))"

            print(f"DEBUG: _extract_problem_components: program_ode_str AFTER safety/termination logic: '{program_ode_str}'")
            print(f"DEBUG: _extract_problem_components: goal_formula_str AFTER safety/termination logic: '{goal_formula_str}'")
        else:
            print(f"DEBUG: _extract_problem_components: !!! SKIPPING safety_modal_formula processing block. safety_modal_formula: {safety_modal_formula}")
            if safety_modal_formula:
                print(f"DEBUG: type(safety_modal_formula.program): {type(safety_modal_formula.program)}")


        # Extract reward from liveness modal formula
        if liveness_modal_formula and isinstance(liveness_modal_formula.formula, PredicateCall):
            ensure_predicate_call = liveness_modal_formula.formula
            ensure_pred_def = self.pred_definition_map.get(ensure_predicate_call.name)
            print(f"DEBUG: _extract_problem_components: ensure_predicate_call: {ensure_predicate_call}")
            print(f"DEBUG: _extract_problem_components: ensure_pred_def: {ensure_pred_def}")
            ensure_goal_str = self._formula_to_string_python(ensure_predicate_call) # This is `ensure(xr, vr, xg)`
            print(f"DEBUG: _extract_problem_components: ensure_goal_str (from _formula_to_string_python): '{ensure_goal_str}'")

            if ensure_pred_def:
                progress_annotation = ensure_pred_def.annotations.get('progress')
                print(f"DEBUG: _extract_problem_components: progress_annotation: {progress_annotation}")
                if progress_annotation:
                    reward_value = progress_annotation.get('reward', 0.0)
                    print(f"DEBUG: _extract_problem_components: reward_value from annotation: {reward_value}")
                    reward_function_str = f"{reward_value} if ({ensure_goal_str}) else -0.1" # Default small penalty
                    print(f"DEBUG: _extract_problem_components: constructed reward_function_str (with annotation): '{reward_function_str}'")
                else:
                    reward_function_str = f"1.0 if ({ensure_goal_str}) else -0.01" # Default reward if no annotation
                    print(f"DEBUG: _extract_problem_components: constructed reward_function_str (no annotation fallback): '{reward_function_str}'")
            else:
                reward_function_str = f"1.0 if ({ensure_goal_str}) else -0.01" # Fallback if pred def not found
                print(f"DEBUG: _extract_problem_components: constructed reward_function_str (no pred def fallback): '{reward_function_str}'")
        else:
            # Fallback reward if liveness or ensure predicate not found/annotated
            # If the safety goal is met, provide a positive reward. Otherwise, a penalty.
            reward_function_str = f"1.0 if ({goal_formula_str}) else -0.01"
            print(f"DEBUG: _extract_problem_components: constructed reward_function_str (liveness fallback): '{reward_function_str}'")

        # Incorporate loop invariant safety into termination and reward
        for inv_data in loop_invariants:
            inv_name = inv_data["name"]
            inv_formula_str = inv_data["formula_str"]

            if inv_name: # Only process if it's a named predicate
                inv_pred_def = self.pred_definition_map.get(inv_name)
                if inv_pred_def:
                    safety_annotation = inv_pred_def.annotations.get('safety')
                    if safety_annotation and safety_annotation.get('on_failure') == 'truncate':
                        # If invariant is violated, terminate (truncate)
                        termination_condition_str = f"({termination_condition_str} or (not ({inv_formula_str})))"
                        # Also add penalty if defined
                        penalty = safety_annotation.get('penalty')
                        if penalty is not None:
                            # Ensure penalty is added correctly to the reward
                            # The reward function will become a more complex ternary expression
                            # This needs careful construction to avoid syntax errors if penalty is 0.0
                            reward_function_str = f"({reward_function_str}) + ({penalty} if (not ({inv_formula_str})) else 0.0)"
                else:
                    print(f"Warning: Could not find predicate definition for invariant '{inv_name}'. Annotations will not be applied.")
            # If inv_name is None, it's a direct formula, so no annotations to apply.

        print(f"DEBUG: _extract_problem_components: Final program_ode_str before return: '{program_ode_str}'")
        print(f"DEBUG: _extract_problem_components: Final goal_formula_str before return: '{goal_formula_str}'")
        print(f"DEBUG: _extract_problem_components: Final termination_condition_body before return: '{termination_condition_str}'")
        print(f"DEBUG: _extract_problem_components: Final reward_function_body before return: '{reward_function_str}'")


        return {
            "initial_condition": initial_condition,
            "program": program_ode_str, # This is now the ODE string
            "goal": goal_formula_str,
            "loop_invariants": [inv_data["formula_str"] for inv_data in loop_invariants], # Store just the string in config
            "differential_invariants": differential_invariants,
            "differential_ghosts": differential_ghosts,
            "termination_condition_body": termination_condition_str,
            "reward_function_body": reward_function_str
        }

    def _serialize_ast_node(self, node: ASTNode) -> Dict[str, Any]:
        """Converts an AST node into a dictionary suitable for JSON serialization."""
        if isinstance(node, Literal):
            return {"_type": "Literal", "value": node.value}
        elif isinstance(node, VarRef):
            return {"_type": "VarRef", "name": node.name}
        elif isinstance(node, BinaryTermOp):
            return {
                "_type": "BinaryTermOp",
                "op": node.op,
                "left": self._serialize_ast_node(node.left),
                "right": self._serialize_ast_node(node.right),
            }
        elif isinstance(node, UnaryTermOp):
            return {
                "_type": "UnaryTermOp",
                "op": node.op,
                "operand": self._serialize_ast_node(node.operand),
            }
        elif isinstance(node, FunctionCall):
            return {
                "_type": "FunctionCall",
                "name": node.name,
                "args": [self._serialize_ast_node(arg) for arg in node.args],
            }
        elif isinstance(node, BooleanLiteral):
            return {"_type": "BooleanLiteral", "value": node.value}
        elif isinstance(node, BinaryFormulaOp):
            return {
                "_type": "BinaryFormulaOp",
                "op": node.op,
                "left": self._serialize_ast_node(node.left),
                "right": self._serialize_ast_node(node.right),
            }
        elif isinstance(node, UnaryFormulaOp):
            return {
                "_type": "UnaryFormulaOp",
                "op": node.op,
                "operand": self._serialize_ast_node(node.operand),
            }
        elif isinstance(node, Comparison):
            return {
                "_type": "Comparison",
                "op": node.op,
                "left": self._serialize_ast_node(node.left),
                "right": self._serialize_ast_node(node.right),
            }
        elif isinstance(node, PredicateCall):
            return {
                "_type": "PredicateCall",
                "name": node.name,
                "args": [self._serialize_ast_node(arg) for arg in node.args],
            }
        elif isinstance(node, Assignment):
            return {
                "_type": "Assignment",
                "var": self._serialize_ast_node(node.var),
                "expr": self._serialize_ast_node(node.expr),
            }
        elif isinstance(node, NondetAssignment):
            return {
                "_type": "NondetAssignment",
                "var": self._serialize_ast_node(node.var),
            }
        elif isinstance(node, Test):
            return {
                "_type": "Test",
                "condition": self._serialize_ast_node(node.condition),
            }
        elif isinstance(node, Sequence):
            return {
                "_type": "Sequence",
                "programs": [self._serialize_ast_node(p) for p in node.programs],
            }
        elif isinstance(node, Choice):
            return {
                "_type": "Choice",
                "left": self._serialize_ast_node(node.left),
                "right": self._serialize_ast_node(node.right),
            }
        elif isinstance(node, Iteration):
            return {
                "_type": "Iteration",
                "program": self._serialize_ast_node(node.program),
                "invariant": self._serialize_ast_node(node.invariant) if node.invariant else None,
            }
        elif isinstance(node, ODE):
            return {
                "_type": "ODE",
                "equations": [self._serialize_ast_node(eq) for eq in node.equations],
                "domain_constraint": self._serialize_ast_node(node.domain_constraint) if node.domain_constraint else None,
            }
        elif isinstance(node, ODEEquation):
            return {
                "_type": "ODEEquation",
                "var": self._serialize_ast_node(node.var),
                "expr": self._serialize_ast_node(node.expr),
            }
        elif isinstance(node, HPBlock):
            return {
                "_type": "HPBlock",
                "program": self._serialize_ast_node(node.program),
            }
        elif isinstance(node, ModalFormula):
            return {
                "_type": "ModalFormula",
                "modality": node.modality,
                "program": self._serialize_ast_node(node.program),
                "formula": self._serialize_ast_node(node.formula),
            }
        elif isinstance(node, Quantifier):
             return {
                "_type": "Quantifier",
                "quantifier_type": node.quantifier_type,
                "variables": [self._serialize_ast_node(v) for v in node.variables],
                "body": self._serialize_ast_node(node.body),
            }
        elif isinstance(node, C_Formula):
            return {
                "_type": "C_Formula",
                "formula": self._serialize_ast_node(node.formula),
            }
        elif node is None: # Handle None for optional fields
            return None
        else:
            raise ValueError(f"Unknown AST node type for serialization: {type(node)}")

    def _deserialize_ast_node(self, data: Dict[str, Any]) -> ASTNode:
        """Converts a serialized AST dictionary back into an AST node."""
        if data is None:
            return None

        node_type = data["_type"]
        if node_type == "Literal":
            return Literal(data["value"])
        elif node_type == "VarRef":
            return VarRef(data["name"])
        elif node_type == "BinaryTermOp":
            return BinaryTermOp(data["op"], self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "UnaryTermOp":
            return UnaryTermOp(data["op"], self._deserialize_ast_node(data["operand"]))
        elif node_type == "FunctionCall":
            return FunctionCall(data["name"], [self._deserialize_ast_node(arg) for arg in data.get("args", [])])
        elif node_type == "BooleanLiteral":
            return BooleanLiteral(data["value"])
        elif node_type == "BinaryFormulaOp":
            return BinaryFormulaOp(data["op"], self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "UnaryFormulaOp":
            return UnaryFormulaOp(data["op"], self._deserialize_ast_node(data["operand"]))
        elif node_type == "Comparison":
            return Comparison(data["op"], self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "PredicateCall":
            return PredicateCall(data["name"], [self._deserialize_ast_node(arg) for arg in data.get("args", [])])
        elif node_type == "Assignment":
            return Assignment(self._deserialize_ast_node(data["var"]), self._deserialize_ast_node(data["expr"]))
        elif node_type == "NondetAssignment":
            return NondetAssignment(self._deserialize_ast_node(data["var"]))
        elif node_type == "Test":
            return Test(self._deserialize_ast_node(data["condition"]))
        elif node_type == "Sequence":
            return Sequence([self._deserialize_ast_node(p) for p in data.get("programs", [])])
        elif node_type == "Choice":
            return Choice(self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "Iteration":
            return Iteration(self._deserialize_ast_node(data.get("program")), self._deserialize_ast_node(data.get("invariant")))
        elif node_type == "ODE":
            return ODE([self._deserialize_ast_node(eq) for eq in data.get("equations", [])], self._deserialize_ast_node(data.get("domain_constraint")))
        elif node_type == "ODEEquation":
            return ODEEquation(self._deserialize_ast_node(data["var"]), self._deserialize_ast_node(data["expr"]))
        elif node_type == "HPBlock":
            return HPBlock(self._deserialize_ast_node(data["program"]))
        elif node_type == "ModalFormula":
            return ModalFormula(data["modality"], self._deserialize_ast_node(data["program"]), self._deserialize_ast_node(data["formula"]))
        elif node_type == "Quantifier":
            return Quantifier(data["quantifier_type"], [self._deserialize_ast_node(v) for v in data.get("variables", [])], self._deserialize_ast_node(data.get("body")))
        elif node_type == "C_Formula":
            return C_Formula(self._deserialize_ast_node(data["formula"]))
        elif data is None:
            return None
        else:
            raise ValueError(f"Unknown AST node type for deserialization: {node_type}")


    def _process_definitions(self, model_ast: KeYmaeraModel):
        if not model_ast.entries:
            return

        archive_entry = model_ast.entries[0]

        # First pass: Populate definition maps for resolution
        for definition in archive_entry.definitions.defs:
            if isinstance(definition, FuncDef):
                self.func_definition_map[definition.name] = definition
            elif isinstance(definition, PredDef):
                self.pred_definition_map[definition.name] = definition
            elif isinstance(definition, HPDef):
                self.hp_definition_map[definition.name] = definition

        # Second pass: Process for config, now that maps are populated
        # Process ProgramVariables (actual variables)
        for pv_decl in archive_entry.program_variables.vars:
            var_name = pv_decl.name
            var_type = pv_decl.type
            annotations = pv_decl.annotations # This is the dict from parse_annotations

            # Check for @const annotation
            const_annotation = annotations.get('const')
            if const_annotation:
                value = const_annotation.get('value')
                # If value is not explicitly set, try to derive from bounds if they are equal
                if value is None and const_annotation.get('bound'):
                    bounds = const_annotation['bound']
                    if isinstance(bounds, list) and len(bounds) == 2:
                        # If bounds are equal, use that value
                        if bounds[0] == bounds[1]:
                            value = str(bounds[0])
                        else:
                            # If bounds are a range, pick the lower bound as a default constant value
                            # This is a design choice; could also be midpoint or upper bound.
                            value = str(bounds[0])
                # DEBUG PRINTS START
                print(f"DEBUG: Generator._process_definitions: Adding constant config for ProgramVariable: {var_name}")
                # DEBUG PRINTS END
                self.config.constants.append(ConstantDefinitionConfig(
                    name=var_name,
                    value=str(value if value is not None else var_name) # Fallback to name if no value (should be rare now)
                ))
            else: # Regular variable, check for @var annotation
                lower_bound = None
                upper_bound = None
                initial_value = None
                var_annotation = annotations.get('var')
                if var_annotation:
                    if var_annotation.get('bound'):
                        bounds = var_annotation['bound']
                        if isinstance(bounds, list) and len(bounds) == 2:
                            lower_bound = str(bounds[0])
                            upper_bound = str(bounds[1])
                    if var_annotation.get('initial_value'):
                        initial_value = str(var_annotation['initial_value'])

                # DEBUG PRINTS START
                print(f"DEBUG: Generator._process_definitions: Adding variable config for ProgramVariable: {var_name}")
                # DEBUG PRINTS END
                self.config.variables.append(VariableDeclarationConfig(
                    name=var_name,
                    type=var_type,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    initial_value=initial_value
                ))

        # Process Definitions (Functions, Predicates, HPs)
        for definition in archive_entry.definitions.defs:
            if isinstance(definition, FuncDef):
                # Check if it's a FuncDef with a @const annotation (like 'ep', 'b', etc. in the example dL)
                const_annotation = definition.annotations.get('const')
                if const_annotation:
                    value = const_annotation.get('value')
                    if value is None and const_annotation.get('bound'):
                        bounds = const_annotation['bound']
                        if isinstance(bounds, list) and len(bounds) == 2:
                            # Use lower bound as default value for constant range if no specific value
                            value = str(bounds[0])
                    # DEBUG PRINTS START
                    print(f"DEBUG: Generator._process_definitions: Adding FuncDef as constant: {definition.name}")
                    # DEBUG PRINTS END
                    self.config.constants.append(ConstantDefinitionConfig(
                        name=definition.name,
                        value=str(value if value is not None else definition.name)
                    ))
                    # IMPORTANT: Do NOT add this to function_definitions if it's a constant
                    continue # Skip to the next definition

                # If not a constant, proceed as a regular function definition
                body_str = self._term_to_string_python(definition.body) # Use Python conversion
                print(f"DEBUG: Generator._process_definitions: Adding FuncDef as function: {definition.name}, args: {[arg.name for arg in definition.args]}, body: {body_str}")
                self.config.function_definitions.append(FuncDefConfig(
                    name=definition.name,
                    args=[arg.name for arg in definition.args],
                    body=body_str
                ))
            elif isinstance(definition, PredDef): # Explicitly handle PredDef
                body_str = self._formula_to_string_python(definition.body) # Use Python conversion
                print(f"DEBUG: Generator._process_definitions: Adding PredDef as function: {definition.name}, args: {[arg.name for arg in definition.args]}, body: {body_str}")
                # Store as FuncDefConfig for evaluation purposes in the generated env
                self.config.function_definitions.append(FuncDefConfig(
                    name=definition.name,
                    args=[arg.name for arg in definition.args],
                    body=body_str
                ))
            elif isinstance(definition, HPDef):
                print(f"DEBUG: Generator._process_definitions: Adding HPDef: {definition.name}")
                hp_config = HPDefConfig(
                    name=definition.name,
                    args=[], # HPDef in ast_config doesn't have explicit args list
                    body=self._hp_to_string(definition.body)
                )
                self.config.hp_definitions.append(hp_config)
                if definition.name == "controlFunc":
                    self.config.control_flow_program_ast_json = self._serialize_ast_node(definition.body)

        # Set up initial_state_sampling_config based on variable bounds
        for var_config in self.config.variables:
            if var_config.initial_value:
                self.config.initial_state_sampling_config[var_config.name] = {"value": var_config.initial_value}
            elif var_config.lower_bound is not None and var_config.upper_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {
                    "min": var_config.lower_bound,
                    "max": var_config.upper_bound
                }
            elif var_config.lower_bound is not None:
                 self.config.initial_state_sampling_config[var_config.name] = {
                    "min": var_config.lower_bound,
                    "max": f"({var_config.lower_bound}) + 1.0" # Example: default upper bound
                }
            elif var_config.upper_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {
                    "min": f"({var_config.upper_bound}) - 1.0", # Example: default lower bound
                    "max": var_config.upper_bound
                }
            else:
                # Default for variables without explicit initial value or bounds
                self.config.initial_state_sampling_config[var_name] = {"min": "-1.0", "max": "1.0"}

        # Add constants to initial_state_sampling_config with their fixed values
        # This is important so their values are available during reset/initialization
        for const_config in self.config.constants:
            self.config.initial_state_sampling_config[const_config.name] = {"value": const_config.value}


    def _process_problem(self, model_ast: KeYmaeraModel):
        if not model_ast.entries:
            return

        archive_entry = model_ast.entries[0]
        problem_block_formula = archive_entry.problem.formula

        extracted_components = self._extract_problem_components(problem_block_formula)

        print(f"DEBUG: _process_problem: program_ode_str before config assignment: '{extracted_components['program']}'")
        print(f"DEBUG: _process_problem: extracted reward_function_body: {extracted_components['reward_function_body']}")
        print(f"DEBUG: _process_problem: extracted termination_condition_body: {extracted_components['termination_condition_body']}")


        self.config.problem = ProblemConfig(
            initial_condition=extracted_components["initial_condition"],
            program=extracted_components["program"], # This is now the ODE string
            goal=extracted_components["goal"],
            loop_invariants=extracted_components["loop_invariants"],
            differential_invariants=extracted_components["differential_invariants"],
            differential_ghosts=extracted_components["differential_ghosts"]
        )
        self.config.termination_condition_body = extracted_components["termination_condition_body"]
        self.config.reward_function_body = extracted_components["reward_function_body"]

    def _determine_action_and_observation_spaces(self):
        # If controlFunc AST is present, it dictates the action space
        if self.config.control_flow_program_ast_json:
            # Deserialize temporarily to analyze structure
            control_flow_ast = self._deserialize_ast_node(self.config.control_flow_program_ast_json)
            
            if isinstance(control_flow_ast, Choice):
                num_choices = 0
                q = deque([control_flow_ast])
                while q:
                    node = q.popleft()
                    if isinstance(node, Choice):
                        q.append(node.right)
                        q.append(node.left)
                    else: # This is a leaf branch
                        num_choices += 1
                
                if num_choices > 0:
                    self.config.action_space_type = "discrete"
                    self.config.action_space_bounds = {"n": num_choices}
                    print(f"DEBUG: Action space determined as Discrete({num_choices}) from controlFunc AST.")
                else:
                    print("WARNING: controlFunc is a Choice but no valid branches found. Falling back to continuous 'ar'.")
                    self.config.action_space_type = "continuous" # Fallback
                    self.config.action_space_bounds = {'ar': {"min": "-float('inf')", "max": "float('inf')"}} # Default bounds
            else:
                # If controlFunc exists but is not a Choice (e.g., a single Sequence, Assignment, or Test)
                # it implies a single discrete action to execute that HP.
                self.config.action_space_type = "discrete"
                self.config.action_space_bounds = {"n": 1}
                print(f"DEBUG: Action space determined as Discrete(1) from controlFunc AST (not a Choice).")
        else:
            # Fallback to previous logic if no controlFunc AST
            action_variables = []
            # Identify action variable 'ar' (robot acceleration)
            if 'ar' in [v.name for v in self.config.variables]:
                action_variables.append('ar')

            if action_variables:
                self.config.action_space_type = "continuous"
                self.config.action_space_bounds = {
                    var.name: {"min": var.lower_bound if var.lower_bound else "-float('inf')",
                               "max": var.upper_bound if var.upper_bound else "float('inf')"}
                    for var in self.config.variables if var.name in action_variables
                }
            else:
                self.config.action_space_type = "discrete"
                self.config.action_space_bounds = {"n": 1}

        observation_variables = []
        # All declared variables are potential observation variables
        for var_config in self.config.variables:
            observation_variables.append(var_config.name)

        if observation_variables:
            self.config.observation_space_type = "continuous"
            self.config.observation_space_bounds = {
                var.name: {"min": var.lower_bound if var.lower_bound is not None else "-float('inf')",
                           "max": var.upper_bound if var.upper_bound is not None else "float('inf')"}
                for var in self.config.variables if var.name in observation_variables
            }
        else:
            self.config.observation_space_type = "continuous"
            self.config.observation_space_bounds = {"dummy_obs": {"min": -1.0, "max": 1.0}} # Placeholder

    def generate_environment_code(self, model_ast: KeYmaeraModel) -> str:
        print("Entering generate_environment_code...") # Debug print
        try:
            # The template is now directly imported as a string variable.
            code_template = ENVIRONMENT_TEMPLATE_CODE

            # Populate internal maps and config based on the AST
            self._process_definitions(model_ast)
            self._process_problem(model_ast)
            self._determine_action_and_observation_spaces()

            import dataclasses
            config_dict = dataclasses.asdict(self.config)
            config_json_str = json.dumps(config_dict, indent=4)

            # Use repr() to safely embed the JSON string into the generated Python code.
            # This will add quotes around the string and escape any internal quotes,
            # ensuring the outer f-string's triple quotes are not prematurely terminated.
            embedded_config_json_str = repr(textwrap.indent(config_json_str, '    '))

            # Prepare user-defined functions and predicates for the generated environment
            user_defined_methods_code_lines = []
            user_defined_function_assignments_lines = []

            for func_def_config in self.config.function_definitions:
                args_str = ", ".join(func_def_config.args)
                func_name = func_def_config.name
                body_expr = func_def_config.body # This is already a string

                # All generated functions will now call _evaluate_expression
                user_defined_methods_code_lines.append(
                    f"    def _{func_name}_impl(self, {args_str}):\n"
                    f"        # Construct context for _evaluate_expression\n"
                    f"        context = {{}}\n"
                    f"        arg_names = {json.dumps(func_def_config.args)}\n"
                    f"        for name, val in zip(arg_names, [{args_str}]):\n"
                    f"            context[name] = val\n"
                    f"        return self._evaluate_expression({json.dumps(body_expr)}, context)\n"
                )
                user_defined_function_assignments_lines.append(f"        self.user_defined_functions['{func_name}'] = self._{func_name}_impl\n") # Indented for _setup_functions (8 spaces)

            user_defined_methods_code_str = "\n".join(user_defined_methods_code_lines)
            user_defined_function_assignments_str = "\n".join(user_defined_function_assignments_lines)

            # Now substitute known placeholders without using str.format
            generated_code = code_template
            generated_code = generated_code.replace('{embedded_config_json_str}', embedded_config_json_str)
            generated_code = generated_code.replace('{metadata}', json.dumps(self.config.metadata, indent=4))
            generated_code = generated_code.replace('{user_defined_methods_code_str}', user_defined_methods_code_str)
            generated_code = generated_code.replace('{user_defined_function_assignments_str}', user_defined_function_assignments_str)
            # Collapse any doubled-brace escapes produced by the template into single braces
            generated_code = generated_code.replace('{{', '{').replace('}}', '}')

            print("Exiting generate_environment_code successfully.") # Debug print
            return generated_code # Return the formatted string
        except Exception as e:
            print(f"Error during environment code generation: {e}")
            import traceback
            traceback.print_exc()
            return None

# --- New function to wrap the generation and file writing ---
def generate_gym_environment_file(dl_content: str, environment_name: str, output_filename: str):
    """
    Parses dL content, generates a Gymnasium environment, and writes it to a file.
    Assumes 'environment_template_code.py' exists in the same directory.
    """
    print(f"Starting environment generation for '{environment_name}'...")

    # Parse the dL content
    parser = KeYmaeraParser(dl_content)
    model_ast = parser.parse()
    print(f"Parsed dL content into AST: {model_ast}")

    # Generate the environment code string
    generator = GymnasiumEnvironmentGenerator(environment_name)
    generated_code = generator.generate_environment_code(model_ast)

    if generated_code:
        # Write the generated code to the specified file
        with open(output_filename, "w") as f:
            f.write(generated_code)
        print(f"Generated environment code saved to {output_filename}")
        return True
    else:
        print("Environment code generation failed.")
        return False

# Example Usage (modified __main__ block)
if __name__ == "__main__":
    dl_content = textwrap.dedent("""
        Theorem "IJRR17/Theorem 14: Reach waypoint with deadline"

        /*
         *	Robot
         * 	- must stop within distance
         *  - can only drive straight and forward
         *  - ensures progress towards goal
         *
         */

        Definitions
            // @const: bound=[0.1:0.5]; def="time limit for control decisions"
            Real ep;      /* time limit for control decisions */
            // @const: bound=[0.5:5.0]; def="minimum braking capability of the robot"
            Real b;       /* minimum braking capability of the robot */
            // @const: bound=[1.0:10.0]; def="maximum acceleration"
            Real A;       /* maximum acceleration -b <= a <= A */
            // @const: bound=[1.0:10.0]; def="goal area size"
            Real GDelta;	 /* goal area size */
            // @const: bound=[5.0:30.0]; def="maximum robot velocity"
            Real Vmax;    /* robot cannot go faster than this */

            Real waypointStartDist(Real xg) = xg-GDelta;
            Real waypointEndDist(Real xg)   = xg+GDelta;

            Real minV = A*ep;
            Real maxTravelTime(Real g) = (g-GDelta)/minV;
            Real stopTime(Real v)      = v/b;
            Real speedUpTime(Real v)   = ep-v/A;

            Real stopDist(Real v) = v^2/(2*b);
            Real accComp(Real v)  = ( (A/b + 1)*(A/2*ep^2 + ep*v) );

            Bool bounds() <-> (
                A > 0
                & b > 0
                & ep > 0
                & Vmax >= 2*A*ep
                & GDelta > Vmax*ep + Vmax^2/(2*b)
            );

            Bool initialState(Real xr, Real vr, Real xg, Real T) <-> (
                vr = 0
                & xr < waypointStartDist(xg)
                & T > ep + maxTravelTime(xg-xr) + ep + stopTime(Vmax)
            );

            Bool assumptions(Real xr, Real vr, Real xg, Real T) <->
                bounds() & initialState(xr, vr, xg, T);

            // @safety: penalty=-1000, on_failure=truncate
            Bool loopinv(Real xr, Real vr, Real xg, Real T) <-> ( // Added T to parameters as it's used in the invariant
                0 <= vr & vr <= Vmax & xr + stopDist(vr) < waypointEndDist(xg)
                        & (waypointStartDist(xg) < xr -> (vr = 0 | T >= stopTime(vr)))
                        & (xr <= waypointStartDist(xg) ->
                            (vr >= minV & T > maxTravelTime(xg-xr) + ep + stopTime(Vmax))
                            | (vr <= minV & T > speedUpTime(vr) + maxTravelTime(xg-xr) + ep + stopTime(Vmax)))
            );

            HP controlFunc ::= {{
                // Choice 0: If in the goal area, decide to brake
                {{ ? (xr > waypointStartDist(xg));  ar := -b; }}
                ++
                // Choice 1: If in the goal area AND already stopped, decide to stay stopped
                {{ ? (xr > waypointStartDist(xg) & vr = 0);  ar := 0; }}
                ++
                // Choice 2: If outside the goal area AND can accelerate, decide to accelerate
                {{ ? (xr <= waypointStartDist(xg) & (xr + stopDist(vr) + accComp(vr) < waypointEndDist(xg) & vr+A*ep <= Vmax));  ar := A; }}
                ++
                // Choice 3: If outside the goal area AND cannot accelerate (must coast), decide to coast
                {{ ? (xr <= waypointStartDist(xg) & !(xr + stopDist(vr) + accComp(vr) < waypointEndDist(xg) & vr+A*ep <= Vmax));  ar := 0; }}
            }};

            HP stepFunc ::= {{xr' = vr, vr' = ar, t' = 1, T'=-1 & t <= ep & vr >= 0}};

            HP dwwpdl ::= {{
                {{	{{ controlFunc; t := 0; }}
                    {{ stepFunc; }}
                }}*@invariant(loopinv(xr, vr, xg, T))
            }};

            // @progress: reward=1000, on_success=terminate, on_failure=terminate
            Bool ensure(Real xr, Real vr, Real xg) <-> (
                waypointStartDist(xg) < xr & vr = 0
            );

        End.

        ProgramVariables
            // @var: bound=[0:100], def="robot position: x"
            Real xr;
            // @var: bound=[0:20], def="robot translational velocity"
            Real vr;
            // @var: bound=[-10:10], def="robot translational acceleration"
            Real ar;
            // @var: bound=[0:100], def="goal position: x"
            Real xg;
            // @var: bound=[0:1], def="time elapsed in current control step"
            Real t;
            // @var: bound=[0:1000], def="global time remaining until deadline"
            Real T;
        End.

        Problem
            assumptions(xr, vr, xg, T)
                -> [ dwwpdl; ](xr < waypointEndDist(xg) & (T <= 0 -> ensure(xr, vr, xg))) // Safety postcondition
                & < dwwpdl; > ensure(xr, vr, xg)                                         // Liveness postcondition
        End.
    """)

    environment_name = "KeYmaeraRobotWaypoint"
    env_file_name = f"{environment_name.lower()}.py"

    # Call the new wrapper function
    if generate_gym_environment_file(dl_content, environment_name, env_file_name):
        try:
            import sys
            import importlib.util
            spec = importlib.util.spec_from_file_location(environment_name.lower(), env_file_name)
            env_module = importlib.util.module_from_spec(spec)
            sys.modules[environment_name.lower()] = env_module
            spec.loader.exec_module(env_module)

            EnvClass = getattr(env_module, 'KeYmaeraGymEnv')

            # Define variable values first
            xr_val = 0.0
            vr_val = 0.0
            xg_val = 50.0

            # Constants from dL:
            ep_dl = 0.1
            b_dl = 0.5
            A_dl = 1.0
            Vmax_dl = 5.0

            # Adjust GDelta_val to satisfy bounds()
            # GDelta > Vmax*ep + Vmax^2/(2*b)
            # GDelta > 5.0*0.1 + 5.0^2/(2*0.5)
            # GDelta > 0.5 + 25.0 = 25.5
            GDelta_val = 30.0 # Adjusted GDelta to satisfy bounds()

            # Recalculate T based on the formula in initialState and adjusted constants
            # waypointStartDist(xg) = xg - GDelta
            waypoint_start_dist_val = xg_val - GDelta_val # 50 - 30 = 20

            # minV = A * ep
            minV_val = A_dl * ep_dl # 1.0 * 0.1 = 0.1

            # maxTravelTime(xg-xr) = ( (xg-xr) - GDelta ) / minV
            # ( (50.0 - 0.0) - 30.0 ) / 0.1 = (20.0) / 0.1 = 200.0
            max_travel_time_val = ( (xg_val - xr_val) - GDelta_val ) / minV_val

            # stopTime(Vmax) = Vmax / b
            stop_time_vmax_val = Vmax_dl / b_dl # 5.0 / 0.5 = 10.0

            # T > ep + maxTravelTime(xg-xr) + ep + stopTime(Vmax)
            # T > 0.1 + 200.0 + 0.1 + 10.0
            # T > 210.2
            T_val = 211.0 # Set T to satisfy the initial condition

            # --- NEW: Separate constants override from initial state sampling ---
            constants_override_list = [
                {'name': 'ep', 'value': str(ep_dl)},
                {'name': 'b', 'value': str(b_dl)},
                {'name': 'A', 'value': str(A_dl)},
                {'name': 'GDelta', 'value': str(GDelta_val)}, # This is the key change
                {'name': 'Vmax', 'value': str(Vmax_dl)},
            ]

            # Construct the initial_state_sampling_config dictionary separately
            initial_sampling_config = {
                # Constants are now overridden via 'constants' list, not here.
                # Variables for initial sampling:
                'xr': {'value': str(xr_val)},
                'vr': {'value': str(vr_val)},
                'ar': {'min': str(-b_dl), 'max': str(A_dl)}, # Ensure ar is within bounds
                'xg': {'value': str(xg_val)},
                'T': {'value': str(T_val)},
                't': {'value': '0.0'} # Explicitly set t to 0
            }

            # Override initial state sampling for specific values
            config_override = {
                'constants': constants_override_list, # Add constants override here
                'initial_state_sampling_config': initial_sampling_config,
                'max_episode_steps': 500 # Increase max steps for a longer run
            }

            print(f"DEBUG: config_override before env creation: {config_override}")
            env = EnvClass(render_mode="human", config_override=config_override)
            print(f"DEBUG: env.config['reward_function_body'] after env creation: {env.config['reward_function_body']}")
            print(f"DEBUG: env.config['termination_condition_body'] after env creation: {env.config['termination_condition_body']}")


            observation, info = env.reset(seed=42)
            print("Initial Observation:", observation)
            print("Initial Info:", info)

            # Check initial condition explicitly
            initial_cond_met = env._evaluate_expression(env.config['problem']['initial_condition'], env.current_state)
            print(f"Initial condition met: {initial_cond_met}")

            # Check bounds predicate
            bounds_met = env._evaluate_expression("self._bounds_impl()", env.current_state)
            print(f"Bounds met: {bounds_met}")

            # Example agent loop using the new discrete action space
            for i in range(env.max_episode_steps):
                current_xr = env.current_state.get('xr', 0.0)
                current_vr = env.current_state.get('vr', 0.0)
                current_xg = env.current_state.get('xg', 0.0)

                waypoint_start_dist = env.user_defined_functions['waypointStartDist'](current_xg)
                
                action_choice = 0 # Default to brake if in goal area

                # Implement a simple policy that tries to mimic the controlFunc logic
                # Choice 0: If in the goal area, decide to brake
                if current_xr > waypoint_start_dist and current_vr > 0:
                    action_choice = 0 # ar := -b
                # Choice 1: If in the goal area AND already stopped, decide to stay stopped
                elif current_xr > waypoint_start_dist and abs(current_vr) < 0.01: # Check for near-zero velocity
                    action_choice = 1 # ar := 0
                # Choice 2: If outside the goal area AND can accelerate, decide to accelerate
                elif current_xr <= waypoint_start_dist:
                    # This condition is complex, let's simplify for the agent
                    # If far from goal, accelerate
                    if (waypoint_start_dist - current_xr) > 5.0 and current_vr < env.current_state['Vmax']:
                        action_choice = 2 # ar := A
                    else:
                        # Otherwise, try to coast or brake softly to approach
                        action_choice = 3 # ar := 0 (coast)
                
                # Ensure action_choice is within the valid range for the discrete action space
                # The example controlFunc has 4 choices (0, 1, 2, 3)
                action_choice = int(np.clip(action_choice, 0, env.action_space.n - 1))

                action = action_choice # Action is now an integer for discrete choice

                observation, reward, terminated, truncated, info = env.step(action)

                # Evaluate loop invariant for debugging
                loop_inv_met = env._evaluate_expression("self._loopinv_impl(xr, vr, xg, T)", env.current_state)

                print(f"Step {i+1:3d} | Action(choice)={action} | ar={env.current_state.get('ar', 'N/A'):.2f} | Obs={observation} | Reward={reward:.2f} | Terminated={terminated} | Truncated={truncated} | LoopInv={loop_inv_met} | Info={info}")

                if terminated or truncated:
                    print("Episode finished.")
                    break
            env.close()

        except Exception as e:
            print(f"Error testing generated environment: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Environment code generation failed, skipping environment testing.")