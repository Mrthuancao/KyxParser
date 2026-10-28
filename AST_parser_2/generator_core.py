import json
import keyword
import textwrap
from typing import List, Dict, Any, Optional
import math
import sympy

# AST node imports
from ast_config import (
    ASTNode, KeYmaeraModel, ArchiveEntry, DefinitionsBlock, Definition, FuncDef, PredDef, HPDef,
    ProgramVariablesBlock, ProgramVariableDecl, ProblemBlock, TacticBlock,
    Term, Literal, VarRef, BinaryTermOp, UnaryTermOp, FunctionCall,
    Formula, BooleanLiteral, BinaryFormulaOp, UnaryFormulaOp, Comparison, Quantifier,
    ModalFormula, PredicateCall, C_Formula,
    HybridProgram, Assignment, NondetAssignment, Test, Sequence, Choice, Iteration,
    ODE, ODEEquation, HPBlock
)

# Parser/lexer imports
from keymaera_parser import KeYmaeraParser

# Configuration dataclasses
from pyast_config import (
    GymEnvironmentConfig,
    VariableDeclarationConfig,
    ConstantDefinitionConfig,
    FuncDefConfig,
    HPDefConfig,
    ProblemConfig
)

# Template string
from environment_template_code import ENVIRONMENT_TEMPLATE_CODE


class GymnasiumEnvironmentGenerator:
    """Core generator class (moved out of the CLI wrapper).

    This class contains the logic to convert parsed KeYmaera ASTs into the
    generated Python source for a Gymnasium environment. It mirrors the
    previous implementation but is placed in a dedicated module to keep the
    original CLI file small.
    """
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.config = GymEnvironmentConfig(environment_name=environment_name)
        self.indent_level = 0
        self.indent_str = "    "
        self.hp_definition_map: Dict[str, HPDef] = {}
        self.func_definition_map: Dict[str, FuncDef] = {}
        self.pred_definition_map: Dict[str, PredDef] = {}
        self.control_choice_count: Optional[int] = None
        # Reused to parse standalone dL expressions found inside annotation values
        # (e.g. `@progress: reward=xr - waypointStartDist(xg)`).
        self._expr_parser = KeYmaeraParser("")

    def _annotation_value_to_python_expr(self, value: Any) -> str:
        """Converts an `@progress`/`@safety` annotation value into a Python-evaluable expression
        string. Plain numbers (the common case, e.g. `reward=1000`) pass through unchanged; a
        value that failed numeric parsing (e.g. `reward=xr - waypointStartDist(xg)`) is a raw dL
        expression string, so it's parsed as a Term and converted the same way user-defined
        function bodies are, letting the reward/penalty be shaped by the model itself."""
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            expression = value.strip()
            if expression.startswith("{") and expression.endswith("}"):
                expression = expression[1:-1].strip()
            try:
                term = self._expr_parser.parse_term(expression=expression)
                return self._term_to_string_python(term)
            except Exception:
                return expression
        return str(value)

    def _indent(self, text: str) -> str:
        lines = text.splitlines()
        indented_lines = [self.indent_str * self.indent_level + line for line in lines]
        return "\n".join(indented_lines)

    # --- Python evaluation conversions ---
    _PYTHON_TERM_OPS = {'+': '+', '-': '-', '*': '*', '/': '/', '^': '**'}
    _PYTHON_FORMULA_OPS = {
        '=': '==', '!=': '!=', '>': '>', '<': '<', '>=': '>=', '<=': '<=',
        '&&': 'and', '&': 'and', '|': 'or'
    }

    def _term_to_string_python(self, term: Term) -> str:
        if isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, VarRef):
            var_name = term.name
            if var_name in self.func_definition_map:
                func_def = self.func_definition_map[var_name]
                if not func_def.args and not func_def.annotations.get('const'):
                    return f"self._{var_name}_impl()"
            if keyword.iskeyword(var_name):
                return f"self.current_state[{var_name!r}]"
            return var_name
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_string_python(term.left)
            right_str = self._term_to_string_python(term.right)
            op = self._PYTHON_TERM_OPS.get(term.op, term.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_string_python(term.operand)
            if term.op == "'":
                raise ValueError(f"Derivative operator '{term.op}' not supported in Python term evaluation: {term}")
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_string_python(arg) for arg in term.args)
            if term.name in {'abs', 'min', 'max'}:
                return f"{term.name}({args_str})"
            return f"self._{term.name}_impl({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for Python eval: {type(term)}")

    def _formula_to_string_python(self, formula: Formula) -> str:
        if isinstance(formula, BooleanLiteral):
            return str(formula.value)
        elif isinstance(formula, Comparison):
            left_str = self._term_to_string_python(formula.left)
            right_str = self._term_to_string_python(formula.right)
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_string_python(formula.left)
            right_str = self._formula_to_string_python(formula.right)
            if formula.op == "->":
                return f"(not ({left_str}) or ({right_str}))"
            elif formula.op == "<->":
                return f"((({left_str}) and ({right_str})) or (not ({left_str}) and not ({right_str})))"
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_string_python(formula.operand)
            if formula.op == "!":
                return f"(not ({operand_str}))"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_string_python(arg) for arg in formula.args)
            return f"self._{formula.name}_impl({args_str})"
        elif isinstance(formula, (ModalFormula, Quantifier, C_Formula)):
            raise ValueError(f"Formula type {type(formula)} cannot be directly converted to Python for evaluation: {formula}")
        else:
            raise ValueError(f"Unsupported Formula type for Python eval: {type(formula)}")

    # --- SymPy conversions ---
    def _term_to_string_sympy(self, term: Term) -> str:
        if isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, VarRef):
            return term.name
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_string_sympy(term.left)
            right_str = self._term_to_string_sympy(term.right)
            return f"({left_str} {term.op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_string_sympy(term.operand)
            if term.op == "'":
                return f"{operand_str}_prime"
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_string_sympy(arg) for arg in term.args)
            return f"sympy.Function('{term.name}')({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for SymPy: {type(term)}")

    def _formula_to_string_sympy(self, formula: Formula) -> str:
        if isinstance(formula, BooleanLiteral):
            return str(formula.value)
        elif isinstance(formula, Comparison):
            left_str = self._term_to_string_sympy(formula.left)
            right_str = self._term_to_string_sympy(formula.right)
            if formula.op == '=': return f"sympy.Eq({left_str}, {right_str})"
            if formula.op == '!=': return f"sympy.Ne({left_str}, {right_str})"
            if formula.op == '>': return f"sympy.Gt({left_str}, {right_str})"
            if formula.op == '<': return f"sympy.Lt({left_str}, {right_str})"
            if formula.op == '>=': return f"sympy.Ge({left_str}, {right_str})"
            if formula.op == '<=': return f"sympy.Le({left_str}, {right_str})"
            return f"({left_str} {formula.op} {right_str})"
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_string_sympy(formula.left)
            right_str = self._formula_to_string_sympy(formula.right)
            if formula.op == "->":
                return f"sympy.Implies({left_str}, {right_str})"
            elif formula.op == "<->":
                return f"sympy.Equivalent({left_str}, {right_str})"
            elif formula.op in ["&&", "&"]:
                return f"sympy.And({left_str}, {right_str})"
            elif formula.op == "|":
                return f"sympy.Or({left_str}, {right_str})"
            return f"({left_str} {formula.op} {right_str})"
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_string_sympy(formula.operand)
            if formula.op == "!":
                return f"sympy.Not({operand_str})"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_string_sympy(arg) for arg in formula.args)
            return f"sympy.Function('{formula.name}')({args_str})"
        elif isinstance(formula, ModalFormula):
            program_str = self._hp_to_string(formula.program)
            formula_str = self._formula_to_string_sympy(formula.formula)
            return f"[{program_str}]({formula_str})"
        elif isinstance(formula, Quantifier):
            vars_str = ", ".join(v.name for v in formula.variables)
            body_str = self._formula_to_string_sympy(formula.body)
            return f"{formula.quantifier_type}({vars_str}).({body_str})"
        elif isinstance(formula, C_Formula):
            return f"C({self._formula_to_string_sympy(formula.formula)})"
        else:
            raise ValueError(f"Unsupported Formula type for SymPy: {type(formula)}")

    def _hp_to_string(self, hp: HybridProgram) -> str:
        if isinstance(hp, Assignment):
            var_str = self._term_to_string_python(hp.var)
            expr_str = self._term_to_string_python(hp.expr)
            return f"{var_str} := {expr_str}"
        elif isinstance(hp, NondetAssignment):
            var_str = self._term_to_string_python(hp.var)
            return f"{{{var_str} := *}}"
        elif isinstance(hp, Test):
            condition_str = self._formula_to_string_python(hp.condition)
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
            invariant_str = f"@invariant({self._formula_to_string_python(hp.invariant)})" if hp.invariant else ""
            return f"({program_str})*{invariant_str}"
        elif isinstance(hp, ODE):
            equations_str = ", ".join(f"{self._term_to_string_sympy(eq.var)}' = {self._term_to_string_sympy(eq.expr)}" for eq in hp.equations)
            if hp.domain_constraint:
                domain_constraint_sympy_str = self._formula_to_string_sympy(hp.domain_constraint)
                domain_str = f"@@@DOMAIN_CONSTRAINT@@@{domain_constraint_sympy_str}"
            else:
                domain_str = ""
            return f"({equations_str}{domain_str})"
        elif isinstance(hp, HPBlock):
            return f"{{{self._hp_to_string(hp.program)}}}"
        elif isinstance(hp, VarRef):
            return hp.name
        else:
            raise ValueError(f"Unsupported HybridProgram type: {type(hp)}")

    def _resolve_hp_to_ode(self, hp_name: str) -> Optional[ODE]:
        hp_definition = self.hp_definition_map.get(hp_name)
        if hp_definition is None:
            return None
        return self._find_ode_in_hp_node(hp_definition.body)

    def _find_ode_in_hp_node(self, hp_node: HybridProgram, visited: Optional[set] = None) -> Optional[ODE]:
        if visited is None:
            visited = set()
        if isinstance(hp_node, ODE):
            return hp_node
        if isinstance(hp_node, Sequence):
            for program in hp_node.programs:
                ode = self._find_ode_in_hp_node(program, visited)
                if ode is not None:
                    return ode
        elif isinstance(hp_node, Choice):
            return (self._find_ode_in_hp_node(hp_node.left, visited)
                    or self._find_ode_in_hp_node(hp_node.right, visited))
        elif isinstance(hp_node, (Iteration, HPBlock)):
            return self._find_ode_in_hp_node(hp_node.program, visited)
        elif isinstance(hp_node, VarRef) and hp_node.name not in visited:
            visited.add(hp_node.name)
            hp_definition = self.hp_definition_map.get(hp_node.name)
            if hp_definition is not None:
                return self._find_ode_in_hp_node(hp_definition.body, visited)
        return None

    def _collect_loop_invariants_from_program(
        self, hp_node: HybridProgram, visited: Optional[set] = None
    ) -> List[Formula]:
        if visited is None:
            visited = set()
        invariants: List[Formula] = []
        if isinstance(hp_node, Iteration):
            if hp_node.invariant is not None:
                invariants.append(hp_node.invariant)
            invariants.extend(self._collect_loop_invariants_from_program(hp_node.program, visited))
        elif isinstance(hp_node, Sequence):
            for program in hp_node.programs:
                invariants.extend(self._collect_loop_invariants_from_program(program, visited))
        elif isinstance(hp_node, Choice):
            invariants.extend(self._collect_loop_invariants_from_program(hp_node.left, visited))
            invariants.extend(self._collect_loop_invariants_from_program(hp_node.right, visited))
        elif isinstance(hp_node, HPBlock):
            invariants.extend(self._collect_loop_invariants_from_program(hp_node.program, visited))
        elif isinstance(hp_node, VarRef) and hp_node.name not in visited:
            visited.add(hp_node.name)
            hp_definition = self.hp_definition_map.get(hp_node.name)
            if hp_definition is not None:
                invariants.extend(
                    self._collect_loop_invariants_from_program(hp_definition.body, visited)
                )
        return invariants

    def _serialize_ast_node(self, node: ASTNode) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {"_type": type(node).__name__}
        for name, value in vars(node).items():
            if isinstance(value, ASTNode):
                serialized[name] = self._serialize_ast_node(value)
            elif isinstance(value, list):
                serialized[name] = [
                    self._serialize_ast_node(item) if isinstance(item, ASTNode) else item
                    for item in value
                ]
            else:
                serialized[name] = value
        return serialized

    def _count_control_choice_paths(self, node: HybridProgram) -> int:
        if isinstance(node, HPBlock):
            return self._count_control_choice_paths(node.program)
        if isinstance(node, Choice):
            return (
                self._count_control_choice_paths(node.left)
                + self._count_control_choice_paths(node.right)
            )
        if isinstance(node, Sequence):
            count = 1
            for program in node.programs:
                count *= self._count_control_choice_paths(program)
            return count
        if isinstance(node, Iteration):
            return self._count_control_choice_paths(node.program)
        return 1

    def _collect_loop_invariants(self, hp_name: str, _visited: Optional[set] = None) -> List[Formula]:
        """Walks a hybrid program (resolving named HP references) and collects every
        `@invariant(...)` formula attached to an Iteration node."""
        hp_definition = self.hp_definition_map.get(hp_name)
        if hp_definition is None:
            return []
        return self._collect_loop_invariants_from_program(hp_definition.body, _visited)

    @staticmethod
    def _constant_values(
        const_annotation: Dict[str, Any],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        value = const_annotation.get('value')
        lower_bound = upper_bound = None
        bound = const_annotation.get('bound')
        if value is None and isinstance(bound, (int, float)):
            value = bound
        elif value is None and isinstance(bound, list) and len(bound) == 2:
            lower_bound, upper_bound = map(str, bound)
            if bound[0] == bound[1]:
                value = bound[0]
                lower_bound = upper_bound = None
            else:
                value = bound[0]
        return (
            str(value) if value is not None else None,
            lower_bound,
            upper_bound,
        )

    def _find_safety_penalty(self, *formulas: Formula, default: str = "-100.0") -> str:
        """Searches formulas (loop invariants, safety postcondition, ...) for a PredicateCall
        whose definition carries an `@safety: penalty=X` annotation, returning the first X found
        as a Python-evaluable expression string (X may be a plain number or a dL expression)."""
        def search(node) -> Optional[str]:
            if node is None:
                return None
            if isinstance(node, PredicateCall):
                pred_def = self.pred_definition_map.get(node.name)
                if pred_def:
                    safety_annotation = pred_def.annotations.get('safety')
                    if safety_annotation and 'penalty' in safety_annotation:
                        return self._annotation_value_to_python_expr(safety_annotation['penalty'])
                for arg in node.args:
                    found = search(arg)
                    if found is not None:
                        return found
                return None
            if isinstance(node, (BinaryFormulaOp, BinaryTermOp)):
                return search(node.left) or search(node.right)
            if isinstance(node, (UnaryFormulaOp, UnaryTermOp)):
                return search(node.operand)
            if isinstance(node, Comparison):
                return search(node.left) or search(node.right)
            if isinstance(node, FunctionCall):
                for arg in node.args:
                    found = search(arg)
                    if found is not None:
                        return found
                return None
            if isinstance(node, Quantifier):
                return search(node.body)
            return None

        for formula in formulas:
            found = search(formula)
            if found is not None:
                return found
        return default

    def _extract_reward_components_from_annotation(self, annotation: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Extracts shape, reward, and penalty from @safety, @progress, or @stability annotations.
        According to the RL-Annotation DSL spec, rewards are calculated as:
        R = shape_expr + (reward_val if φ is True, else penalty_val)
        
        Returns a dict with keys 'shape', 'reward', 'penalty', each as a Python-evaluable string.
        Defaults to '0.0' for any missing component.
        """
        components = {
            'shape': '0.0',
            'reward': '0.0',
            'penalty': '0.0'
        }
        
        if annotation:
            if 'shape' in annotation:
                components['shape'] = self._annotation_value_to_python_expr(annotation['shape'])
            if 'reward' in annotation:
                components['reward'] = self._annotation_value_to_python_expr(annotation['reward'])
            if 'penalty' in annotation:
                components['penalty'] = self._annotation_value_to_python_expr(annotation['penalty'])
        
        return components

    def _extract_problem_components(self, problem_formula: Formula) -> Dict[str, Any]:
        # Implementation mirrors previous logic; kept compact for readability
        initial_condition = "True"
        program_ode_str = ""
        goal_formula_str = "True"
        termination_condition_str = "True"
        reward_function_str = "0.0"
        loop_invariants = []

        current_formula = problem_formula

        if isinstance(current_formula, BinaryFormulaOp) and current_formula.op == "->":
            initial_condition = self._formula_to_string_python(current_formula.left)
            current_formula = current_formula.right

        safety_modal_formula = None
        liveness_modal_formula = None

        if isinstance(current_formula, ModalFormula) and current_formula.modality == 'box':
            safety_modal_formula = current_formula
        elif isinstance(current_formula, BinaryFormulaOp) and current_formula.op in ["&&", "&"]:
            if isinstance(current_formula.left, ModalFormula) and current_formula.left.modality == 'box':
                safety_modal_formula = current_formula.left
            if isinstance(current_formula.right, ModalFormula) and current_formula.right.modality == 'diamond':
                liveness_modal_formula = current_formula.right
        elif isinstance(current_formula, ModalFormula) and current_formula.modality == 'box':
            safety_modal_formula = current_formula
            if isinstance(safety_modal_formula.formula, BinaryFormulaOp) and safety_modal_formula.formula.op in ["&&", "&"]:
                if isinstance(safety_modal_formula.formula.right, ModalFormula) and safety_modal_formula.formula.right.modality == 'diamond':
                    liveness_modal_formula = safety_modal_formula.formula.right
                    safety_modal_formula = ModalFormula(modality='box', program=safety_modal_formula.program, formula=safety_modal_formula.formula.left)

        raw_invariant_formulas: List[Formula] = []

        if safety_modal_formula is not None:
            program = safety_modal_formula.program
            if isinstance(program, VarRef):
                hp_name = program.name
                ode_block = self._resolve_hp_to_ode(hp_name)
                if ode_block:
                    program_ode_str = self._hp_to_string(ode_block)
                else:
                    raise ValueError(f"Could not resolve ODE block from hybrid program '{hp_name}'")
                raw_invariant_formulas = self._collect_loop_invariants(hp_name)
            else:
                ode_block = self._find_ode_in_hp_node(program)
                if ode_block is not None:
                    program_ode_str = self._hp_to_string(ode_block)
                raw_invariant_formulas = self._collect_loop_invariants_from_program(program)

            try:
                goal_formula_str = self._formula_to_string_python(safety_modal_formula.formula)
            except Exception:
                goal_formula_str = "True"

            termination_condition_str = f"(not ({goal_formula_str}))"

        for inv_formula in raw_invariant_formulas:
            try:
                inv_formula_str = self._formula_to_string_python(inv_formula)
            except Exception:
                continue
            loop_invariants.append({"formula_str": inv_formula_str})
            termination_condition_str = f"({termination_condition_str} or (not ({inv_formula_str})))"

        safety_termination_condition_str = termination_condition_str
        progress_success_conditions = []
        for pred_name, pred_def in self.pred_definition_map.items():
            annotation = pred_def.annotations.get('progress', {})
            on_success = annotation.get('on_success', '')
            if isinstance(on_success, str) and on_success.startswith('terminate'):
                args_str = ", ".join(arg.name for arg in pred_def.args)
                args = f"({args_str})" if args_str else "()"
                progress_success_conditions.append(f"self._{pred_name}_impl{args}")
        if progress_success_conditions:
            success_condition = " or ".join(progress_success_conditions)
            termination_condition_str = f"({termination_condition_str} or ({success_condition}))"

        # `@safety: penalty=X` on a loop-invariant predicate (or the safety postcondition itself)
        # gives the model author's intended penalty for violating that condition; fall back to a
        # generic -100.0 if the model doesn't declare one.
        # NOTE: Per the RL-Annotation DSL spec, safety predicates can also have shape= and reward= fields.
        safety_penalty = self._find_safety_penalty(
            *raw_invariant_formulas,
            safety_modal_formula.formula if safety_modal_formula is not None else None
        )
        not_safe_str = f"(({safety_penalty}) if ({safety_termination_condition_str}) else 0.0)"
        # Collect reward contributions from ALL annotated predicates in the model.
        # Per RL-Annotation DSL spec: R = shape_expr + (reward_val if φ is True, else penalty_val)
        # KEY RULES:
        # 1. @progress and @safety can annotate ANY predicate regardless of name/modality
        # 2. A predicate can have ONLY ONE: either @progress or @safety, not both (mutually exclusive)
        reward_components_list = []
        
        for pred_name, pred_def in self.pred_definition_map.items():
            # Enforce mutual exclusivity: @progress OR @safety, not both
            if 'progress' in pred_def.annotations:
                progress_annotation = pred_def.annotations['progress']
                components = self._extract_reward_components_from_annotation(progress_annotation)
                args_str = ", ".join([arg.name for arg in pred_def.args])
                pred_call_str = f"self._{pred_name}_impl({args_str})" if pred_def.args else f"self._{pred_name}_impl()"
                reward_components_list.append({
                    'predicate_name': pred_name,
                    'predicate_str': pred_call_str,
                    'components': components,
                    'type': 'progress'
                })
            elif 'safety' in pred_def.annotations:
                safety_annotation = pred_def.annotations['safety']
                components = self._extract_reward_components_from_annotation(safety_annotation)
                args_str = ", ".join([arg.name for arg in pred_def.args])
                pred_call_str = f"self._{pred_name}_impl({args_str})" if pred_def.args else f"self._{pred_name}_impl()"
                reward_components_list.append({
                    'predicate_name': pred_name,
                    'predicate_str': pred_call_str,
                    'components': components,
                    'type': 'safety'
                })
        
        # Build the reward function by combining all components.
        # Formula per spec: R = shape + (reward if φ true else penalty)
        # When safe: sum up all contributions; when unsafe: apply safety_penalty override
        if reward_components_list:
            # Build a list of reward expressions, one per annotated predicate
            reward_terms = []
            for component in reward_components_list:
                pred_str = component['predicate_str']
                comps = component['components']
                # Build: shape + (reward if pred else penalty)
                term = f"(({comps['shape']}) + (({comps['reward']}) if ({pred_str}) else ({comps['penalty']})))"
                reward_terms.append(term)
            
            # Sum all components when safe; apply safety_penalty when unsafe
            combined_rewards = " + ".join(reward_terms)
            reward_function_str = (
                f"({combined_rewards}) if not ({safety_termination_condition_str}) "
                f"else ({safety_penalty})"
            )
        else:
            # Fallback to default behavior if no annotations found
            if liveness_modal_formula and isinstance(liveness_modal_formula.formula, PredicateCall):
                ensure_goal_str = self._formula_to_string_python(liveness_modal_formula.formula)
                reward_function_str = f"1.0 if ({ensure_goal_str}) else {not_safe_str}"
            else:
                reward_function_str = f"1.0 if ({goal_formula_str}) else {not_safe_str}"

        return {
            "initial_condition": initial_condition,
            "program": program_ode_str,
            "goal": goal_formula_str,
            "loop_invariants": [inv.get("formula_str") for inv in loop_invariants],
            "differential_invariants": [],
            "differential_ghosts": [],
            "termination_condition_body": termination_condition_str,
            "reward_function_body": reward_function_str
        }

    def _process_definitions(self, model_ast: KeYmaeraModel):
        if not model_ast.entries:
            return
        archive_entry = model_ast.entries[0]

        for definition in archive_entry.definitions.defs:
            if isinstance(definition, FuncDef):
                if not definition.args and isinstance(definition.body, VarRef) and definition.body.name == definition.name:
                    # Bare declarations such as `Real ep;` are symbolic constants, not zero-arg functions.
                    continue
                self.func_definition_map[definition.name] = definition
            elif isinstance(definition, PredDef):
                self.pred_definition_map[definition.name] = definition
            elif isinstance(definition, HPDef):
                self.hp_definition_map[definition.name] = definition

        for pv_decl in archive_entry.program_variables.vars:
            var_name = pv_decl.name
            annotations = pv_decl.annotations
            const_annotation = annotations.get('const')
            if const_annotation:
                value, lower_bound, upper_bound = self._constant_values(const_annotation)
                self.config.constants.append(ConstantDefinitionConfig(
                    name=var_name,
                    value=value if value is not None else var_name,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                ))
            else:
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
                    if var_annotation.get('initial_value') is not None:
                        initial_value = str(var_annotation['initial_value'])
                self.config.variables.append(VariableDeclarationConfig(name=var_name, type=pv_decl.type, lower_bound=lower_bound, upper_bound=upper_bound, initial_value=initial_value))

        for definition in archive_entry.definitions.defs:
            if isinstance(definition, FuncDef):
                if not definition.args and isinstance(definition.body, VarRef) and definition.body.name == definition.name:
                    # Treat implicit constant declarations as constants rather than generated functions.
                    const_annotation = definition.annotations.get('const', {})
                    value, lower_bound, upper_bound = self._constant_values(const_annotation)
                    self.config.constants.append(ConstantDefinitionConfig(
                        name=definition.name,
                        value=value or "1.0",
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    ))
                    continue

                const_annotation = definition.annotations.get('const')
                if const_annotation:
                    value, lower_bound, upper_bound = self._constant_values(const_annotation)
                    self.config.constants.append(ConstantDefinitionConfig(
                        name=definition.name,
                        value=value or definition.name,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    ))
                    continue
                body_str = self._term_to_string_python(definition.body)
                self.config.function_definitions.append(FuncDefConfig(name=definition.name, args=[arg.name for arg in definition.args], body=body_str))
            elif isinstance(definition, PredDef):
                body_str = self._formula_to_string_python(definition.body)
                self.config.function_definitions.append(FuncDefConfig(name=definition.name, args=[arg.name for arg in definition.args], body=body_str))
            elif isinstance(definition, HPDef):
                self.config.hp_definitions.append(HPDefConfig(name=definition.name, args=[], body=self._hp_to_string(definition.body)))

        # A named controller is the agent's action interface even when its assignments
        # target a declared program variable such as `ar`.
        controller = next(
            (definition for definition in archive_entry.definitions.defs
             if isinstance(definition, HPDef) and definition.name in {'controlFunc', 'ctrl'}),
            None
        )
        if controller is not None:
            controller_program = controller.body
            while isinstance(controller_program, HPBlock):
                controller_program = controller_program.program
            self.control_choice_count = self._count_control_choice_paths(controller_program)
            self.config.control_flow_program_ast_json = self._serialize_ast_node(controller_program)

        for var_config in self.config.variables:
            if var_config.initial_value:
                self.config.initial_state_sampling_config[var_config.name] = {"value": var_config.initial_value}
            elif var_config.lower_bound is not None and var_config.upper_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {"min": var_config.lower_bound, "max": var_config.upper_bound}
            elif var_config.lower_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {"min": var_config.lower_bound, "max": f"({var_config.lower_bound}) + 1.0"}
            elif var_config.upper_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {"min": f"({var_config.upper_bound}) - 1.0", "max": var_config.upper_bound}
            else:
                self.config.initial_state_sampling_config[var_config.name] = {"min": "-1.0", "max": "1.0"}

        for const_config in self.config.constants:
            self.config.initial_state_sampling_config[const_config.name] = {"value": const_config.value}

    def _process_problem(self, model_ast: KeYmaeraModel):
        if not model_ast.entries:
            return
        archive_entry = model_ast.entries[0]
        problem_block_formula = archive_entry.problem.formula
        extracted_components = self._extract_problem_components(problem_block_formula)
        simulation_programs = self._quantified_simulation_programs(problem_block_formula)
        if simulation_programs is not None:
            initialization_program, step_program = simulation_programs
            self.config.simulation_mode = True
            self.config.simulation_initialization_ast_json = (
                self._serialize_ast_node(initialization_program)
                if initialization_program is not None else None
            )
            self.config.simulation_step_ast_json = self._serialize_ast_node(step_program)
            self.config.verification_formula_ast_json = self._serialize_ast_node(problem_block_formula)
            self._build_simulation_annotation_metadata()
            extracted_components.update({
                "initial_condition": "True",
                "program": "",
                "goal": "True",
                "loop_invariants": [],
                "termination_condition_body": "False",
                "reward_function_body": " + ".join(
                    self.config.simulation_reward_terms
                ) or "0.0",
            })
        self.config.problem = ProblemConfig(initial_condition=extracted_components["initial_condition"], program=extracted_components["program"], goal=extracted_components["goal"], loop_invariants=extracted_components["loop_invariants"], differential_invariants=extracted_components["differential_invariants"], differential_ghosts=extracted_components["differential_ghosts"])
        self.config.termination_condition_body = extracted_components["termination_condition_body"]
        self.config.reward_function_body = extracted_components["reward_function_body"]

    @staticmethod
    def _parse_annotation_action(action_value: Any) -> tuple[str, str]:
        if not isinstance(action_value, str):
            return "continue", "none"
        action, separator, status = action_value.partition(":status=")
        if not separator:
            action, separator, status = action_value.partition(":")
        action = action.strip().lower()
        if action not in {"terminate", "truncate", "continue"}:
            raise ValueError(f"Unsupported RL annotation action: {action_value}")
        return action, status.strip() if separator else "none"

    def _build_simulation_annotation_metadata(self) -> None:
        for pred_name, pred_def in self.pred_definition_map.items():
            annotations = pred_def.annotations
            kinds = [kind for kind in ("safety", "progress") if kind in annotations]
            if not kinds:
                continue
            if len(kinds) > 1:
                raise ValueError(
                    f"Predicate '{pred_name}' cannot use both @safety and @progress."
                )
            if pred_def.args:
                raise ValueError(
                    f"Annotated predicate '{pred_name}' must have no arguments "
                    "for quantified simulation."
                )

            kind = kinds[0]
            annotation = annotations[kind]
            predicate = f"self._{pred_name}_impl()"
            components = self._extract_reward_components_from_annotation(annotation)
            shape = components["shape"]
            reward = components["reward"]
            penalty = components["penalty"]
            self.config.simulation_reward_terms.append(
                f"(({shape}) + (({reward}) if ({predicate}) else ({penalty})))"
            )

            on_success, success_status = self._parse_annotation_action(
                annotation.get("on_success", "continue")
            )
            on_failure, failure_status = self._parse_annotation_action(
                annotation.get("on_failure", "continue")
            )
            self.config.simulation_predicate_rules.append({
                "predicate": predicate,
                "kind": kind,
                "on_success": on_success,
                "success_status": success_status,
                "on_failure": on_failure,
                "failure_status": failure_status,
            })

    def _quantified_simulation_programs(
        self, formula: Formula
    ) -> Optional[tuple[Optional[HybridProgram], HybridProgram]]:
        has_quantifier = False
        modal_formulas: List[ModalFormula] = []

        def collect(node: Formula) -> None:
            nonlocal has_quantifier
            if isinstance(node, Quantifier):
                has_quantifier = True
                collect(node.body)
            elif isinstance(node, ModalFormula):
                modal_formulas.append(node)
            elif isinstance(node, BinaryFormulaOp):
                collect(node.left)
                collect(node.right)
            elif isinstance(node, UnaryFormulaOp):
                collect(node.operand)
            elif isinstance(node, C_Formula):
                collect(node.formula)

        collect(formula)
        if not has_quantifier or not modal_formulas:
            return None
        if len(modal_formulas) != 1 or modal_formulas[0].modality != "box":
            raise ValueError(
                "Quantified simulation generation currently requires exactly one box modality."
            )

        program = self._expand_hybrid_program_references(
            modal_formulas[0].program, preserve_controller=True
        )
        program, controller_references = self._remove_controller_references(program)
        if controller_references:
            if self.config.control_flow_program_ast_json is None:
                raise ValueError(
                    "The modal program references controlFunc/ctrl, but no agent "
                    "controller definition was found."
                )
            if controller_references != 1:
                raise ValueError(
                    "Quantified simulation currently requires exactly one "
                    "controlFunc/ctrl call in the modal loop."
                )
        while isinstance(program, HPBlock):
            program = program.program
        programs = program.programs if isinstance(program, Sequence) else [program]
        iteration_indices = [
            index for index, item in enumerate(programs) if isinstance(item, Iteration)
        ]
        if len(iteration_indices) != 1:
            raise ValueError(
                "Quantified simulation generation requires exactly one top-level hybrid-program loop."
            )
        iteration_index = iteration_indices[0]
        if iteration_index != len(programs) - 1:
            raise ValueError(
                "Quantified simulation generation requires initialization before its final loop."
            )
        initialization_parts = programs[:iteration_index]
        initialization = (
            Sequence(initialization_parts) if len(initialization_parts) > 1
            else initialization_parts[0] if initialization_parts
            else None
        )
        return initialization, programs[iteration_index].program

    def _expand_hybrid_program_references(
        self,
        program: HybridProgram,
        expansion_stack: Optional[set[str]] = None,
        preserve_controller: bool = False,
    ) -> HybridProgram:
        if expansion_stack is None:
            expansion_stack = set()
        if (
            preserve_controller
            and isinstance(program, VarRef)
            and program.name in {"controlFunc", "ctrl"}
        ):
            return program
        if isinstance(program, VarRef) and program.name in self.hp_definition_map:
            if program.name in expansion_stack:
                cycle = " -> ".join((*sorted(expansion_stack), program.name))
                raise ValueError(f"Recursive hybrid-program definition: {cycle}")
            definition = self.hp_definition_map[program.name]
            return self._expand_hybrid_program_references(
                definition.body,
                expansion_stack | {program.name},
                preserve_controller,
            )
        if isinstance(program, HPBlock):
            return self._expand_hybrid_program_references(
                program.program, expansion_stack, preserve_controller
            )
        if isinstance(program, Sequence):
            return Sequence([
                self._expand_hybrid_program_references(
                    child, expansion_stack, preserve_controller
                )
                for child in program.programs
            ])
        if isinstance(program, Choice):
            return Choice(
                self._expand_hybrid_program_references(
                    program.left, expansion_stack, preserve_controller
                ),
                self._expand_hybrid_program_references(
                    program.right, expansion_stack, preserve_controller
                ),
            )
        if isinstance(program, Iteration):
            return Iteration(
                self._expand_hybrid_program_references(
                    program.program, expansion_stack, preserve_controller
                ),
                program.invariant,
            )
        return program

    def _remove_controller_references(
        self, program: HybridProgram
    ) -> tuple[HybridProgram, int]:
        if isinstance(program, VarRef) and program.name in {"controlFunc", "ctrl"}:
            return Sequence([]), 1
        if isinstance(program, HPBlock):
            program.program, count = self._remove_controller_references(program.program)
            return program, count
        if isinstance(program, Sequence):
            cleaned_programs = []
            total_references = 0
            for child in program.programs:
                cleaned_child, count = self._remove_controller_references(child)
                cleaned_programs.append(cleaned_child)
                total_references += count
            program.programs = cleaned_programs
            return program, total_references
        if isinstance(program, Choice):
            program.left, left_count = self._remove_controller_references(program.left)
            program.right, right_count = self._remove_controller_references(program.right)
            return program, left_count + right_count
        if isinstance(program, Iteration):
            program.program, count = self._remove_controller_references(program.program)
            return program, count
        return program, 0

    def _determine_action_and_observation_spaces(self):
        if self.config.simulation_mode:
            self.config.action_space_type = "discrete"
            self.config.action_space_bounds = {"n": 1}
            self.config.observation_space_type = "continuous"
            self.config.observation_space_bounds = {
                var.name: {
                    "min": var.lower_bound if var.lower_bound is not None else "-float('inf')",
                    "max": var.upper_bound if var.upper_bound is not None else "float('inf')",
                }
                for var in self.config.variables
            }
            if not self.config.observation_space_bounds:
                self.config.observation_space_bounds = {
                    "dummy_obs": {"min": -1.0, "max": 1.0}
                }
            return

        action_variables = []
        observation_variables = []
        if self.config.control_flow_program_ast_json is not None:
            self.config.action_space_type = "discrete"
            self.config.action_space_bounds = {"n": self.control_choice_count or 1}
        else:
            if 'ar' in [v.name for v in self.config.variables]:
                action_variables.append('ar')
            if action_variables:
                self.config.action_space_type = "continuous"
                self.config.action_space_bounds = {var.name: {"min": var.lower_bound if var.lower_bound else "-float('inf')", "max": var.upper_bound if var.upper_bound else "float('inf')"} for var in self.config.variables if var.name in action_variables}
            else:
                self.config.action_space_type = "discrete"
                self.config.action_space_bounds = {"n": 1}
        for var_config in self.config.variables:
            observation_variables.append(var_config.name)
        if observation_variables:
            self.config.observation_space_type = "continuous"
            self.config.observation_space_bounds = {var.name: {"min": var.lower_bound if var.lower_bound is not None else "-float('inf')", "max": var.upper_bound if var.upper_bound is not None else "float('inf')"} for var in self.config.variables if var.name in observation_variables}
        else:
            self.config.observation_space_type = "continuous"
            self.config.observation_space_bounds = {"dummy_obs": {"min": -1.0, "max": 1.0}}

    def generate_environment_code(self, model_ast: KeYmaeraModel) -> str:
        code_template = ENVIRONMENT_TEMPLATE_CODE
        self._process_definitions(model_ast)
        self._process_problem(model_ast)
        self._determine_action_and_observation_spaces()
        import dataclasses
        config_dict = dataclasses.asdict(self.config)
        config_json_str = json.dumps(config_dict, indent=4)
        embedded_config_json_str = repr(textwrap.indent(config_json_str, '    '))

        user_defined_methods_code_lines = []
        user_defined_function_assignments_lines = []
        for func_def_config in self.config.function_definitions:
            args_str = ", ".join(func_def_config.args)
            func_name = func_def_config.name
            params = f"self, {args_str}" if args_str else "self"
            body_expr = func_def_config.body
            user_defined_methods_code_lines.append(
                f"    def _{func_name}_impl({params}):\n"
                f"        context = {{}}\n"
                f"        arg_names = {json.dumps(func_def_config.args)}\n"
                f"        for name, val in zip(arg_names, [{args_str}]):\n"
                f"            context[name] = val\n"
                f"        return self._evaluate_expression({json.dumps(body_expr)}, context)\n"
            )
            user_defined_function_assignments_lines.append(f"        self.user_defined_functions['{func_name}'] = self._{func_name}_impl\n")

        user_defined_methods_code_str = "\n".join(user_defined_methods_code_lines)
        user_defined_function_assignments_str = "\n".join(user_defined_function_assignments_lines)

        generated_code = code_template
        generated_code = generated_code.replace('{embedded_config_json_str}', embedded_config_json_str)
        generated_code = generated_code.replace('{metadata}', json.dumps(self.config.metadata, indent=4))
        generated_code = generated_code.replace('{user_defined_methods_code_str}', user_defined_methods_code_str)
        generated_code = generated_code.replace('{user_defined_function_assignments_str}', user_defined_function_assignments_str)
        generated_code = generated_code.replace('{{', '{').replace('}}', '}')

        return generated_code


def generate_gym_environment_file(dl_content: str, environment_name: str, output_filename: str):
    parser = KeYmaeraParser(dl_content)
    model_ast = parser.parse()
    generator = GymnasiumEnvironmentGenerator(environment_name)
    generated_code = generator.generate_environment_code(model_ast)
    if generated_code:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        return True
    return False
